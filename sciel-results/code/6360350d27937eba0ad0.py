import time
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Model the problem as a Max-Cut-like problem on the conflict graph of edges.
       Two edges conflict if they form a 2-face (C4). We want to select a set of edges
       such that no face has all 4 edges selected.
       
    2. Spectral Initialization:
       Construct a graph where vertices are the edges of Q7 (448 nodes).
       Connect two edge-nodes if they belong to the same 2-face.
       This graph is the "conflict graph".
       We want to partition these 448 nodes into "Selected" and "Not Selected" to maximize
       the number of Selected nodes, subject to the constraint that no 4-cycle in the
       original Q7 is fully selected.
       
       A standard Max-Cut formulation doesn't directly apply because the constraint is
       "at most 3 of 4", not "cut edges". However, we can use the spectral embedding
       of the conflict graph to find a good initial binary vector.
       
       Specifically, we look for a vector x in {0,1}^E that minimizes the number of
       "bad" faces. A bad face is one where all 4 edges are 1.
       
       Let's use a different spectral approach:
       The problem is equivalent to finding an independent set in a hypergraph? No.
       
       Let's stick to the prompt's inspiration: "Model the problem as a maximum cut problem 
       in the line graph of Q7... solve it using spectral methods (e.g., Ncut) on the 
       conflict graph to obtain an initial partition, followed by local refinement."
       
       Actually, the conflict graph for C4s is such that if we select a set of edges S,
       we require that for every face F, |S ∩ F| <= 3.
       
       Let's define a graph G_conflict where nodes are edges of Q7.
       Two edges are adjacent in G_conflict if they share a face.
       
       If we assign +1 to selected and -1 to not selected, this doesn't map directly to Max-Cut.
       
       Alternative Spectral Idea:
       The constraint is local. We can use the Fiedler vector of the adjacency matrix of the 
       conflict graph to determine a threshold for selection.
       
       Let's try a simpler, robust spectral initialization:
       1. Build adjacency matrix A of the conflict graph (448x448).
       2. Compute the second smallest eigenvector (Fiedler vector) of the Laplacian L = D - A.
       3. Threshold this vector at 0 to get initial partition.
       4. Run a highly optimized local search (simulated annealing or hill climbing) 
          with fast vectorized validity checks.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    d = 7
    n = 2 ** d
    E = d * (1 << (d - 1))  # 448
    
    # 1. Precompute edges and faces
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces) # 672
    
    # Map each edge to the list of faces it belongs to
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # 2. Spectral Initialization
    # Build conflict graph adjacency
    # Two edges conflict if they are in the same face.
    # Note: In a face, all 4 edges are pairwise in the same face.
    # So the conflict graph is a union of cliques of size 4?
    # No, an edge belongs to 2 faces. So it's connected to 3 other edges in face 1, and 3 in face 2.
    # Some might be shared.
    
    row_idx = []
    col_idx = []
    data = []
    
    for f_idx, face in enumerate(faces):
        # Add edges within this face
        for i in range(4):
            for j in range(i+1, 4):
                u = face[i]
                v = face[j]
                row_idx.append(u)
                col_idx.append(v)
                data.append(1.0)
                row_idx.append(v)
                col_idx.append(u)
                data.append(1.0)
    
    A_conflict = csr_matrix((data, (row_idx, col_idx)), shape=(E, E))
    
    # Compute Laplacian
    # L = D - A
    # We can use scipy.sparse.csgraph.laplacian
    from scipy.sparse.csgraph import laplacian
    L = laplacian(A_conflict, normed=False)
    
    # Find Fiedler vector (2nd smallest eigenvalue)
    # For sparse matrices, eigsh is efficient
    try:
        # k=2 to get the first two eigenvectors, we want the 2nd (index 1)
        # sigma=0 to find eigenvalues near 0
        vals, vecs = eigsh(L, k=2, sigma=0, which='LM', return_eigenvectors=True)
        # Sort by eigenvalue
        idx_sort = np.argsort(vals)
        fiedler_vec = vecs[:, idx_sort[1]]
        
        # Threshold at 0
        # We want to select edges. 
        # Usually, one side of the cut is "selected".
        # Let's try both orientations and pick the better one.
        init_1 = (fiedler_vec > 0).astype(np.float64)
        init_2 = (fiedler_vec < 0).astype(np.float64)
        
        def score_state(state):
            # state: 0/1 array
            # Check validity and count
            # Vectorized check:
            # For each face, sum of edges.
            # A_face is sparse matrix F x E
            # We can precompute A_face
            
            # Precompute A_face once
            return state.sum()
            
        # Let's just do a quick local repair on the best of the two
        best_init = None
        best_score = -1
        
        for init in [init_1, init_2]:
            # Repair: remove edges from full faces
            state = init.copy()
            full_count = np.zeros(F, dtype=np.int32)
            # Initialize counts
            for f_idx, face in enumerate(faces):
                full_count[f_idx] = state[face[0]] + state[face[1]] + state[face[2]] + state[face[3]]
            
            # Remove edges from faces with count 4
            changed = True
            while changed:
                changed = False
                # Find faces with count 4
                bad_faces = np.where(full_count == 4)[0]
                if len(bad_faces) > 0:
                    # Pick a random bad face
                    bf = bad_faces[np.random.randint(len(bad_faces))]
                    face = faces[bf]
                    # Remove one edge from this face
                    # Heuristic: remove the edge that is in the most other bad faces?
                    # For speed, just pick first one that is 1
                    for e in face:
                        if state[e] == 1:
                            state[e] = 0
                            # Update counts for all faces containing e
                            for f2 in edge_faces[e]:
                                full_count[f2] -= 1
                            changed = True
                            break
            
            # Add edges back greedily
            improved = True
            while improved:
                improved = False
                # Find edges that are 0 and can be added
                # An edge e can be added if for all faces f containing e, full_count[f] < 3
                # Vectorized approach is hard for dynamic updates, so we loop
                for e in range(E):
                    if state[e] == 0:
                        can_add = True
                        for f in edge_faces[e]:
                            if full_count[f] == 3:
                                can_add = False
                                break
                        if can_add:
                            state[e] = 1
                            for f in edge_faces[e]:
                                full_count[f] += 1
                            improved = True
                            
            s = int(np.sum(state))
            if s > best_score:
                best_score = s
                best_init = state
                
    except Exception as e:
        # Fallback to random if spectral fails
        best_init = np.random.rand(E) < 0.6
        best_score = -1
        
    if best_init is None:
        best_init = np.zeros(E, dtype=np.float64)
        best_score = 0
        
    # 3. Local Refinement (Simulated Annealing / Hill Climbing)
    # Use a fast validity check.
    # We maintain full_count array.
    
    state = best_init.copy()
    full_count = np.zeros(F, dtype=np.int32)
    for f_idx, face in enumerate(faces):
        full_count[f_idx] = state[face[0]] + state[face[1]] + state[face[2]] + state[face[3]]
    
    current_score = int(np.sum(state))
    best_state = state.copy()
    best_found_score = current_score
    
    # Simple Hill Climbing with random restarts if stuck
    # Since we have 140s, we can do a lot of moves.
    
    # Precompute neighbors for faster access?
    # edge_faces is already a list of lists.
    
    # Let's use a "Kick" strategy: if no improvement for X steps, do a random kick.
    
    steps_since_improve = 0
    max_steps_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random edge to flip
        e = np.random.randint(E)
        
        if state[e] == 1:
            # Try removing
            # Check if removing improves score? No, removing decreases score.
            # We only remove if it allows adding more?
            # In this problem, removing an edge never helps directly increase score,
            # unless it unblocks other edges.
            # So we should primarily try adding.
            # If we try flipping 1->0, score drops by 1.
            # We only accept if it leads to a state where we can add >1 edges.
            # This is complex. Let's stick to:
            # 1. Try adding 0->1. Accept if valid.
            # 2. Try removing 1->0. Accept if it allows adding >1 edges immediately?
            
            # Strategy: 
            # 1. Randomly pick an edge.
            # 2. If 0, try to set to 1. If valid, do it.
            # 3. If 1, try to set to 0. Only do it if we are stuck (hill climbing stuck).
            
            if state[e] == 0:
                # Try adding
                valid = True
                for f in edge_faces[e]:
                    if full_count[f] == 3:
                        valid = False
                        break
                if valid:
                    state[e] = 1
                    for f in edge_faces[e]:
                        full_count[f] += 1
                    current_score += 1
                    steps_since_improve = 0
                    if current_score > best_found_score:
                        best_found_score = current_score
                        best_state = state.copy()
            else:
                # state[e] == 1
                # Only remove if we are stuck
                if steps_since_improve > max_steps_no_improve:
                    # Remove
                    state[e] = 0
                    for f in edge_faces[e]:
                        full_count[f] -= 1
                    current_score -= 1
                    steps_since_improve = 0 # Reset to allow more exploration
                else:
                    steps_since_improve += 1
                    
        else:
            # state[e] == 0
            valid = True
            for f in edge_faces[e]:
                if full_count[f] == 3:
                    valid = False
                    break
            if valid:
                state[e] = 1
                for f in edge_faces[e]:
                    full_count[f] += 1
                current_score += 1
                steps_since_improve = 0
                if current_score > best_found_score:
                    best_found_score = current_score
                    best_state = state.copy()
            else:
                steps_since_improve += 1

    # Final check and return
    result = []
    for i in range(E):
        if best_state[i] == 1:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
