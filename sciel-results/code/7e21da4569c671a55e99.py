"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    # Precompute face membership
    face_members = {}
    edge_faces = [[] for _ in range(total_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for base in range(n):
                if (base & (1 << i)) or (base & (1 << j)):
                    continue
                v0 = base
                v1 = base | (1 << i)
                v2 = base | (1 << j)
                v3 = base | (1 << i) | (1 << j)
                
                e0 = (v0, v1)
                e1 = (v2, v3)
                e2 = (v0, v2)
                e3 = (v1, v3)
                
                idx0 = edge_to_idx.get((min(e0[0], e0[1]), max(e0[0], e0[1])))
                idx1 = edge_to_idx.get((min(e1[0], e1[1]), max(e1[0], e1[1])))
                idx2 = edge_to_idx.get((min(e2[0], e2[1]), max(e2[0], e2[1])))
                idx3 = edge_to_idx.get((min(e3[0], e3[1]), max(e3[0], e3[1])))
                
                if idx0 is None or idx1 is None or idx2 is None or idx3 is None:
                    continue
                
                face_id = (i, j, base)
                face_members[face_id] = [idx0, idx1, idx2, idx3]
                for idx in [idx0, idx1, idx2, idx3]:
                    edge_faces[idx].append(face_id)
    
    num_faces = len(face_members)
    face_list = list(face_members.values())
    face_ids = list(face_members.keys())
    
    # Build adjacency: for each edge, which other edges share a face with it
    # Two edges conflict if they are in the same face
    edge_adj = [[] for _ in range(total_edges)]
    for members in face_list:
        for a in range(4):
            for b in range(a + 1, 4):
                ea, eb = members[a], members[b]
                if eb not in edge_adj[ea]:
                    edge_adj[ea].append(eb)
                    edge_adj[eb].append(ea)
    
    # Spectral Fiedler-vector initialization
    def fiedler_init():
        """Use the Fiedler vector of the conflict graph to initialize edge selection."""
        # Build Laplacian of the conflict graph
        # L = D - A where A is adjacency, D is degree
        degrees = np.array([len(edge_adj[i]) for i in range(total_edges)], dtype=np.float64)
        
        # Build sparse Laplacian
        from scipy.sparse import csr_matrix
        row = []
        col = []
        data = []
        
        for i in range(total_edges):
            # Diagonal: degree
            row.append(i)
            col.append(i)
            data.append(degrees[i])
            # Off-diagonal: -1 for each neighbor
            for j in edge_adj[i]:
                row.append(i)
                col.append(j)
                data.append(-1.0)
        
        L = csr_matrix((data, (row, col)), shape=(total_edges, total_edges))
        
        # Compute Fiedler vector (second smallest eigenvalue)
        # Use scipy's eigsh for sparse eigenvalue problem
        try:
            from scipy.sparse.linalg import eigsh
            # Find smallest non-zero eigenvalue
            vals, vecs = eigsh(L, k=2, which='SM')
            # Fiedler vector is the one with smallest positive eigenvalue
            fiedler_vec = None
            min_pos_val = float('inf')
            for v, vec in zip(vals, vecs.T):
                if v > 1e-8 and v < min_pos_val:
                    min_pos_val = v
                    fiedler_vec = vec
            if fiedler_vec is None:
                # Fallback: use all-ones vector
                fiedler_vec = np.ones(total_edges)
        except Exception:
            # Fallback to random
            fiedler_vec = np.random.randn(total_edges)
        
        # Use Fiedler vector to determine edge selection
        # Select edges with positive Fiedler values, then refine
        threshold = 0.0
        selected = set()
        for i in range(total_edges):
            if fiedler_vec[i] > threshold:
                # Check if adding this edge creates a conflict
                can_add = True
                for face_id in edge_faces[i]:
                    members = face_members[face_id]
                    other_count = sum(1 for m in members if m != i and m in selected)
                    if other_count == 3:
                        can_add = False
                        break
                if can_add:
                    selected.add(i)
        
        # Now try to add more edges
        for i in range(total_edges):
            if i not in selected:
                can_add = True
                for face_id in edge_faces[i]:
                    members = face_members[face_id]
                    other_count = sum(1 for m in members if m != i and m in selected)
                    if other_count == 3:
                        can_add = False
                        break
                if can_add:
                    selected.add(i)
        
        return selected
    
    # Local search with 2-opt swaps
    def local_search_2opt(selected, time_limit):
        """Local search using 2-opt swaps (remove one edge, add another)."""
        current = set(selected)
        best = set(current)
        best_count = len(current)
        
        # Precompute for each edge: which edges can be added if this edge is removed
        # This is expensive, so we'll do it on the fly
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < time_limit and no_improve < max_no_improve:
            improved = False
            
            # Try all 2-opt moves: remove edge e1, add edge e2
            # For efficiency, only try promising moves
            for e1 in list(current):
                if time.time() >= time_limit:
                    break
                
                # Temporarily remove e1
                current.remove(e1)
                
                # Find all edges that can now be added
                addable = []
                for e2 in range(total_edges):
                    if e2 in current or e2 == e1:
                        continue
                    can_add = True
                    for face_id in edge_faces[e2]:
                        members = face_members[face_id]
                        other_count = sum(1 for m in members if m != e2 and m in current)
                        if other_count == 3:
                            can_add = False
                            break
                    if can_add:
                        addable.append(e2)
                
                # If we can add more than 1 edge, do the best swap
                if len(addable) > 1:
                    # Add all addable edges (this is a multi-add, not just 2-opt)
                    for e2 in addable:
                        current.add(e2)
                    improved = True
                    no_improve = 0
                else:
                    # Put e1 back
                    current.add(e1)
                    no_improve += 1
                
                if improved:
                    if len(current) > best_count:
                        best = set(current)
                        best_count = len(current)
                    break  # Restart after improvement
        
        return best
    
    # Simulated annealing on the conflict structure
    def simulated_annealing(selected, time_limit):
        """SA that allows temporary conflicts but penalizes them."""
        current = set(selected)
        best = set(current)
        best_count = len(current)
        
        T = 20.0
        cooling = 0.9995
        penalty = 5.0
        
        def get_conflict_count(s):
            count = 0
            for members in face_list:
                if all(m in s for m in members):
                    count += 1
            return count
        
        current_conflicts = get_conflict_count(current)
        
        while time.time() < time_limit:
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            # Random move
            move_type = random.random()
            
            if move_type < 0.6:
                # Add a random edge
                e = random.randint(0, total_edges - 1)
                if e in current:
                    continue
                new_current = current | {e}
                new_conflicts = get_conflict_count(new_current)
                delta = 1 - penalty * (new_conflicts - current_conflicts)
            elif move_type < 0.85:
                # Remove a random edge
                if not current:
                    continue
                e = random.choice(list(current))
                new_current = current - {e}
                new_conflicts = get_conflict_count(new_current)
                delta = -1 - penalty * (new_conflicts - current_conflicts)
            else:
                # Swap
                if not current:
                    continue
                e1 = random.choice(list(current))
                e2 = random.randint(0, total_edges - 1)
                if e2 in current:
                    continue
                new_current = (current - {e1}) | {e2}
                new_conflicts = get_conflict_count(new_current)
                delta = penalty * (current_conflicts - new_conflicts)
            
            # Accept
            if delta > 0 or random.random() < math.exp(delta / T):
                current = new_current
                current_conflicts = get_conflict_count(current)
                if len(current) > best_count:
                    best = set(current)
                    best_count = len(current)
        
        return best
    
    # Main search
    best_result = None
    best_count = 0
    
    # Phase 1: Fiedler initialization
    if time.time() < deadline - 60:
        selected = fiedler_init()
        selected = local_search_2opt(selected, start_time + 30)
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = set(selected)
    
    # Phase 2: Multiple random restarts with local search
    num_restarts = 0
    while time.time() < deadline - 15:
        num_restarts += 1
        
        # Random greedy init
        order = list(range(total_edges))
        random.shuffle(order)
        selected = set()
        for e in order:
            can_add = True
            for face_id in edge_faces[e]:
                members = face_members[face_id]
                other_count = sum(1 for m in members if m != e and m in selected)
                if other_count == 3:
                    can_add = False
                    break
            if can_add:
                selected.add(e)
        
        # Local search
        selected = local_search_2opt(selected, start_time + 15)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = set(selected)
        
        # SA from this solution
        if time.time() < deadline - 10:
            sa_result = simulated_annealing(selected, deadline - 5)
            if len(sa_result) > best_count:
                best_count = len(sa_result)
                best_result = set(sa_result)
        
        # If we've been doing well, try Fiedler again with different random seed
        if num_restarts % 5 == 0 and time.time() < deadline - 40:
            selected = fiedler_init()
            selected = local_search_2opt(selected, start_time + 25)
            if len(selected) > best_count:
                best_count = len(selected)
                best_result = set(selected)
    
    # Final cleanup: ensure no conflicts
    if best_result is not None:
        # Remove edges from any conflicting faces
        final = set(best_result)
        changed = True
        while changed:
            changed = False
            for members in face_list:
                if all(m in final for m in members):
                    # Remove the edge with highest degree (most conflicts)
                    to_remove = max(members, key=lambda m: len(edge_adj[m]))
                    final.remove(to_remove)
                    changed = True
        best_result = final
    
    # Convert back to list of [u, v]
    result = []
    for edge_idx in sorted(best_result):
        x, y, i = edges[edge_idx]
        result.append([x, y])
    
    return result
# EVOLVE-BLOCK-END
