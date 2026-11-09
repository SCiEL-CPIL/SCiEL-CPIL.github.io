import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to run a single greedy pass
    def run_greedy(initial_edges, seed):
        random.seed(seed)
        edges = set(initial_edges)
        
        # Precompute bitmask for each vertex
        adj_mask = [0] * 16
        for (u, v) in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        # Precompute perms24
        perms24 = list(itertools.permutations(range(4)))
        
        def contains_q3(vertices):
            if len(vertices) != 8:
                return False
            
            v_list = list(vertices)
            # Fast filter: count edges
            # If < 12 edges, cannot contain Q3
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_mask[v_list[i]] >> v_list[j]) & 1:
                        edge_count += 1
            if edge_count < 12:
                return False
                
            # Try all 70 ways to choose 4 vertices for A
            for i in range(8):
                for j in range(i+1, 8):
                    for k in range(j+1, 8):
                        for l in range(k+1, 8):
                            A = [v_list[i], v_list[j], v_list[k], v_list[l]]
                            # B is the complement
                            # Optimization: precompute B list?
                            # set(A) is slow. 
                            # Since v_list is fixed, we can just check membership or use a bitmask.
                            # Let's use a bitmask for A for fast exclusion.
                            mask_A = (1 << A[0]) | (1 << A[1]) | (1 << A[2]) | (1 << A[3])
                            B = [v for v in v_list if not (mask_A >> v) & 1]
                            
                            # Try all 24 permutations of B
                            for p in perms24:
                                # Required edges:
                                # A[0] connects to B[p[0]], B[p[1]], B[p[3]]
                                # A[1] connects to B[p[0]], B[p[1]], B[p[2]]
                                # A[2] connects to B[p[0]], B[p[2]], B[p[3]]
                                # A[3] connects to B[p[1]], B[p[2]], B[p[3]]
                                
                                # Check using bitmask
                                ok = True
                                a0 = A[0]
                                if not (adj_mask[a0] >> B[p[0]]) & 1: ok = False
                                elif not (adj_mask[a0] >> B[p[1]]) & 1: ok = False
                                elif not (adj_mask[a0] >> B[p[3]]) & 1: ok = False
                                
                                if ok:
                                    a1 = A[1]
                                    if not (adj_mask[a1] >> B[p[0]]) & 1: ok = False
                                    elif not (adj_mask[a1] >> B[p[1]]) & 1: ok = False
                                    elif not (adj_mask[a1] >> B[p[2]]) & 1: ok = False
                                
                                if ok:
                                    a2 = A[2]
                                    if not (adj_mask[a2] >> B[p[0]]) & 1: ok = False
                                    elif not (adj_mask[a2] >> B[p[2]]) & 1: ok = False
                                    elif not (adj_mask[a2] >> B[p[3]]) & 1: ok = False
                                
                                if ok:
                                    a3 = A[3]
                                    if not (adj_mask[a3] >> B[p[1]]) & 1: ok = False
                                    elif not (adj_mask[a3] >> B[p[2]]) & 1: ok = False
                                    elif not (adj_mask[a3] >> B[p[3]]) & 1: ok = False
                                
                                if ok:
                                    return True
            return False

        all_pairs = set(itertools.combinations(range(16), 2))
        non_edges = list(all_pairs - edges)
        
        # Randomize the order of non-edges
        random.shuffle(non_edges)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Re-shuffle or just iterate? 
            # Re-shuffling every pass might be good to avoid local minima in ordering.
            random.shuffle(non_edges)
            
            current_non_edges = [e for e in non_edges if e not in edges]
            
            # Sort by degree (descending) to prioritize high-impact edges
            # But keep some randomness? 
            # Let's sort by degree + noise
            def degree(v):
                return bin(adj_mask[v]).count('1')
            
            # Add small random noise to break ties and explore different paths
            current_non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]) + random.uniform(0, 1), reverse=True)
            
            for e in current_non_edges:
                if time.time() > deadline:
                    break
                u, v = e
                remaining = [x for x in range(16) if x != u and x != v]
                safe = True
                
                # Temporarily add edge
                edges.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
                for combo in itertools.combinations(remaining, 6):
                    verts = [u, v] + list(combo)
                    if contains_q3(verts):
                        safe = False
                        break
                
                if safe:
                    changed = True
                else:
                    edges.remove((u, v))
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    
        return edges

    # Initial seed: K_{3,13}
    # U = {0, 1, 2}, V = {3, ..., 15}
    initial_edges = set()
    for u in range(3):
        for v in range(3, 16):
            initial_edges.add((u, v))
            
    best_edges = set()
    best_count = -1
    
    # Run multiple restarts
    seed = 42
    while time.time() < deadline:
        current_edges = run_greedy(initial_edges, seed)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            # Optional: update initial_edges to best_edges for next run?
            # This might help if the greedy path depends on the starting point.
            # But K_{3,13} is a good structural start.
            # Let's try starting from the best found so far in subsequent runs?
            # No, that might get stuck. Let's stick to K_{3,13} or mix.
            # Actually, starting from best_edges might be risky if it's not a "seed" in the same way.
            # But it is a valid Q3-free graph.
            # Let's alternate: 1. K_{3,13}, 2. Best, 3. K_{3,13}...
            # For now, just restart from K_{3,13} with different random orders.
        seed += 1
        
        # If we've found a very high edge count, we might want to break early if we can't improve?
        # Max possible is 120 (K16). Q3-free max is likely much lower.
        # If best_count > 80, we are doing well.
        
    return [list(e) for e in best_edges]
