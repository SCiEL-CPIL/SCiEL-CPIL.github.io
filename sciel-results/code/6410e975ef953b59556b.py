import time
import random
import sys

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy: Iterated Local Search (ILS) with Greedy Add and Perturbation.
    1. Precompute edges and faces.
    2. Initialize with a dense random set.
    3. Repair: Remove edges from full faces.
    4. Local Optimize: Repeatedly add any edge that doesn't complete a face.
    5. Perturb: Remove a random subset of edges to escape local optimum.
    6. Repeat until time limit.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges
    # Edge index mapping
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edge_list) # 448
    
    # 2. Precompute faces (C4s)
    # A face is defined by two bits i, j and a base vertex u.
    # We generate unique faces by sorting the 4 edges.
    faces = [] # List of lists of edge indices
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
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
    
    # 3. Precompute adjacency: for each edge, which faces does it touch?
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # State: present[i] is True if edge i is selected
    best_score = -1
    best_state = None
    
    def greedy_add(state, face_counts):
        """
        Repeatedly add any edge that can be added without creating a conflict.
        Returns the number of edges added.
        """
        added = 0
        improved = True
        while improved:
            improved = False
            for e in range(E):
                if state[e]:
                    continue
                # Check if adding e creates a conflict
                # A conflict is created if any face containing e already has 3 other edges present.
                # i.e., face_count[f] == 3 for some f in edge_faces[e].
                can_add = True
                for f_idx in edge_faces[e]:
                    if face_counts[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    state[e] = True
                    added += 1
                    improved = True
                    # Update face counts
                    for f_idx in edge_faces[e]:
                        face_counts[f_idx] += 1
        return added

    def repair(state, face_counts):
        """
        Remove edges to eliminate all conflicts (faces with 4 edges).
        Heuristic: Remove an edge from a conflicting face.
        To be fast, we just find one conflicting face and remove a random edge from it.
        """
        while True:
            # Find a face with count 4
            target_face_idx = -1
            for f_idx in range(F):
                if face_counts[f_idx] == 4:
                    target_face_idx = f_idx
                    break
            
            if target_face_idx == -1:
                break
            
            face = faces[target_face_idx]
            # Remove one edge from this face
            # Random choice to diversify
            e_to_remove = random.choice(face)
            state[e_to_remove] = False
            
            # Update face counts for faces containing e_to_remove
            for f_idx in edge_faces[e_to_remove]:
                face_counts[f_idx] -= 1

    # Main ILS Loop
    random.seed(42)
    iteration = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        iteration += 1
        
        # Initialization
        # Mix of strategies:
        # 1. Random dense (p=0.6)
        # 2. Parity-based (structured)
        if iteration % 2 == 0:
            state = [random.random() < 0.6 for _ in range(E)]
        else:
            # Parity: include edge (u,v) if popcount(min(u,v)) % 2 == 0
            state = [False] * E
            for idx, (u, v) in enumerate(edge_list):
                if bin(min(u,v)).count('1') % 2 == 0:
                    state[idx] = True
        
        # Calculate initial face counts
        face_counts = [0] * F
        for f_idx, face in enumerate(faces):
            count = 0
            for e in face:
                if state[e]:
                    count += 1
            face_counts[f_idx] = count
        
        # Repair to get a valid state
        repair(state, face_counts)
        
        # Local Optimize: Greedy Add
        greedy_add(state, face_counts)
        
        # Record best
        current_score = sum(state)
        if current_score > best_score:
            best_score = current_score
            best_state = state[:]
            
            if best_score >= 308:
                break
                
        # Perturbation: Remove k random edges to escape local optimum
        # The number of edges to remove can vary.
        # As we improve, we might want smaller perturbations? 
        # Or larger to jump? 
        # Let's try a random perturbation size between 5 and 20.
        k = random.randint(5, 20)
        
        # Select k edges to remove. 
        # Prioritize removing edges that are "redundant"? 
        # For now, random is fine.
        edges_to_remove = random.sample([i for i in range(E) if state[i]], k)
        
        for e in edges_to_remove:
            state[e] = False
            for f_idx in edge_faces[e]:
                face_counts[f_idx] -= 1
        
        # After perturbation, we must repair again (though removing edges doesn't create conflicts, 
        # it might leave us in a state where we can't add back efficiently without some structure,
        # but repair is just in case if we implemented moves that could conflict, 
        # but here we only remove, so no new conflicts. 
        # However, we will call greedy_add which assumes a valid start.)
        
        greedy_add(state, face_counts)
        
        current_score = sum(state)
        if current_score > best_score:
            best_score = current_score
            best_state = state[:]
            
            if best_score >= 308:
                break

    if best_state is None:
        return []
        
    result = []
    for i in range(E):
        if best_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
