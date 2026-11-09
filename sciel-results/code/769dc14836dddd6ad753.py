import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute perms24 globally or inside? Inside is fine.
    perms24 = list(itertools.permutations(range(4)))
    
    def run_greedy(initial_edges, seed, deadline):
        random.seed(seed)
        edges = set(initial_edges)
        
        # Precompute bitmask for each vertex
        adj_mask = [0] * 16
        for (u, v) in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        def contains_q3(vertices):
            if len(vertices) != 8:
                return False
            
            v_list = list(vertices)
            # Fast filter: count edges
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_mask[v_list[i]] >> v_list[j]) & 1:
                        edge_count += 1
            if edge_count < 12:
                return False
            
            # Precompute adj masks for the 8 vertices relative to each other
            # adj_local[i] is a bitmask of which of the 8 vertices v_list[i] is connected to
            adj_local = [0] * 8
            for i in range(8):
                mask = 0
                for j in range(8):
                    if i != j and (adj_mask[v_list[i]] >> v_list[j]) & 1:
                        mask |= (1 << j)
                adj_local[i] = mask
                
            # Try all 70 ways to choose 4 vertices for A
            # A is indices 0..7
            for i in range(8):
                for j in range(i+1, 8):
                    for k in range(j+1, 8):
                        for l in range(k+1, 8):
                            A_idx = [i, j, k, l]
                            # B_idx is complement
                            mask_A = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                            B_idx = [m for m in range(8) if not (mask_A >> m) & 1]
                            
                            # Check if bipartite graph between A and B can support Q3
                            # Each vertex in A must have >= 3 neighbors in B
                            # Each vertex in B must have >= 3 neighbors in A
                            valid_split = True
                            for a in A_idx:
                                # Count neighbors in B
                                # adj_local[a] has bits set for neighbors in 0..7
                                # We want to count bits that are in B_idx
                                b_mask = 0
                                for b in B_idx:
                                    b_mask |= (1 << b)
                                if (adj_local[a] & b_mask).bit_count() < 3:
                                    valid_split = False
                                    break
                            if not valid_split: continue
                            for b in B_idx:
                                if (adj_local[b] & mask_A).bit_count() < 3:
                                    valid_split = False
                                    break
                            if not valid_split: continue

                            # Try all 24 permutations of B relative to A
                            # Q3 structure:
                            # A: 0,1,2,3
                            # B: 0,1,2,3
                            # Edges:
                            # A0-B0, A0-B1, A0-B3
                            # A1-B0, A1-B1, A1-B2
                            # A2-B0, A2-B2, A2-B3
                            # A3-B1, A3-B2, A3-B3
                            
                            # We map A_idx[0..3] to A0..A3
                            # We map B_idx[0..3] permuted to B0..B3
                            
                            a0, a1, a2, a3 = A_idx
                            
                            for p in perms24:
                                b0 = B_idx[p[0]]
                                b1 = B_idx[p[1]]
                                b2 = B_idx[p[2]]
                                b3 = B_idx[p[3]]
                                
                                # Check edges using adj_local
                                # A0 connected to B0, B1, B3
                                if not ((adj_local[a0] >> b0) & 1): continue
                                if not ((adj_local[a0] >> b1) & 1): continue
                                if not ((adj_local[a0] >> b3) & 1): continue
                                
                                # A1 connected to B0, B1, B2
                                if not ((adj_local[a1] >> b0) & 1): continue
                                if not ((adj_local[a1] >> b1) & 1): continue
                                if not ((adj_local[a1] >> b2) & 1): continue
                                
                                # A2 connected to B0, B2, B3
                                if not ((adj_local[a2] >> b0) & 1): continue
                                if not ((adj_local[a2] >> b2) & 1): continue
                                if not ((adj_local[a2] >> b3) & 1): continue
                                
                                # A3 connected to B1, B2, B3
                                if not ((adj_local[a3] >> b1) & 1): continue
                                if not ((adj_local[a3] >> b2) & 1): continue
                                if not ((adj_local[a3] >> b3) & 1): continue
                                
                                return True
            return False

        all_pairs = set(itertools.combinations(range(16), 2))
        non_edges = list(all_pairs - edges)
        
        # Randomize the order of non-edges
        random.shuffle(non_edges)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            random.shuffle(non_edges)
            
            current_non_edges = [e for e in non_edges if e not in edges]
            
            def degree(v):
                return bin(adj_mask[v]).count('1')
            
            # Sort by degree sum, descending
            current_non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
            
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
                
                # Check for Q3
                # Optimization: Only check combinations involving u or v?
                # No, adding an edge can only create new Q3s that include that edge.
                # A Q3 containing edge (u,v) must have u and v in the 8 vertices.
                # So we only need to check 6-subsets of remaining 14 vertices.
                # This is C(14, 6) = 3003 checks.
                # The old code checked C(14, 6) as well (remaining was 14).
                
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
    initial_edges = set()
    for u in range(3):
        for v in range(3, 16):
            initial_edges.add((u, v))
            
    best_edges = set()
    best_count = -1
    
    seed = 42
    # First pass: standard greedy
    current_edges = run_greedy(initial_edges, seed, deadline)
    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = set(current_edges)
    seed += 1
    
    # Second phase: Kicks
    # Try to improve by removing some edges and re-greedy
    kick_seed = 100
    while time.time() < deadline - 5:
        # Create a perturbed version of best_edges
        perturbed_edges = set(best_edges)
        # Remove 2-4 random edges
        num_remove = random.randint(2, 4)
        edges_list = list(perturbed_edges)
        random.shuffle(edges_list)
        for e in edges_list[:num_remove]:
            perturbed_edges.remove(e)
            
        # Run greedy on perturbed
        current_edges = run_greedy(perturbed_edges, kick_seed, deadline)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
        kick_seed += 1
        
        # If we have plenty of time, try smaller kicks
        if time.time() < deadline - 20:
             # Try removing 1 edge?
             perturbed_edges = set(best_edges)
             edges_list = list(perturbed_edges)
             random.shuffle(edges_list)
             if edges_list:
                 perturbed_edges.remove(edges_list[0])
                 current_edges = run_greedy(perturbed_edges, kick_seed, deadline)
                 if len(current_edges) > best_count:
                     best_count = len(current_edges)
                     best_edges = set(current_edges)
             kick_seed += 1

    return [list(e) for e in best_edges]
