import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute perms24
    perms24 = list(itertools.permutations(range(4)))
    
    # Precompute all 8-vertex subsets and their edge index pairs for fast Q3 check
    # For each 8-subset, precompute the list of (i,j) pairs that form the 12 edges of K8
    # Actually, we'll use a different approach: precompute for each 8-vertex set the
    # 70 possible A-splits and for each split the 24 B-permutations
    
    def contains_q3_fast(vertices, edge_matrix):
        """Check if 8 vertices contain Q3 as subgraph. edge_matrix is 16x16 bool list."""
        if len(vertices) != 8:
            return False
        
        v_list = list(vertices)
        
        # Fast filter: count edges among the 8 vertices
        edge_count = 0
        for i in range(8):
            vi = v_list[i]
            row = edge_matrix[vi]
            for j in range(i+1, 8):
                if row[v_list[j]]:
                    edge_count += 1
        if edge_count < 12:
            return False
        
        # Build local adjacency: adj_local[i] is bitmask of which of the 8 vertices v_list[i] connects to
        adj_local = [0] * 8
        for i in range(8):
            vi = v_list[i]
            row = edge_matrix[vi]
            mask = 0
            for j in range(8):
                if i != j and row[v_list[j]]:
                    mask |= (1 << j)
            adj_local[i] = mask
            
        # Try all 70 ways to choose 4 vertices for A
        for i in range(8):
            for j in range(i+1, 8):
                for k in range(j+1, 8):
                    for l in range(k+1, 8):
                        A_idx = [i, j, k, l]
                        mask_A = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        B_idx = [m for m in range(8) if not (mask_A >> m) & 1]
                        
                        # Quick degree check: each vertex in A must have >=3 neighbors in B
                        b_mask = 0
                        for b in B_idx:
                            b_mask |= (1 << b)
                        
                        valid_split = True
                        for a in A_idx:
                            if (adj_local[a] & b_mask).bit_count() < 3:
                                valid_split = False
                                break
                        if not valid_split:
                            continue
                        for b in B_idx:
                            if (adj_local[b] & mask_A).bit_count() < 3:
                                valid_split = False
                                break
                        if not valid_split:
                            continue

                        a0, a1, a2, a3 = A_idx
                        
                        for p in perms24:
                            b0 = B_idx[p[0]]
                            b1 = B_idx[p[1]]
                            b2 = B_idx[p[2]]
                            b3 = B_idx[p[3]]
                            
                            # Check the 12 required edges
                            if not (adj_local[a0] >> b0) & 1: continue
                            if not (adj_local[a0] >> b1) & 1: continue
                            if not (adj_local[a0] >> b3) & 1: continue
                            if not (adj_local[a1] >> b0) & 1: continue
                            if not (adj_local[a1] >> b1) & 1: continue
                            if not (adj_local[a1] >> b2) & 1: continue
                            if not (adj_local[a2] >> b0) & 1: continue
                            if not (adj_local[a2] >> b2) & 1: continue
                            if not (adj_local[a2] >> b3) & 1: continue
                            if not (adj_local[a3] >> b1) & 1: continue
                            if not (adj_local[a3] >> b2) & 1: continue
                            if not (adj_local[a3] >> b3) & 1: continue
                            
                            return True
        return False

    def run_greedy(initial_edges, seed, deadline):
        random.seed(seed)
        edges = set(initial_edges)
        
        # Build edge_matrix: 16x16 boolean list
        edge_matrix = [[False] * 16 for _ in range(16)]
        for (u, v) in edges:
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
            
        def degree(v):
            return sum(1 for x in range(16) if x != v and edge_matrix[v][x])
            
        all_pairs = list(itertools.combinations(range(16), 2))
        non_edges = [e for e in all_pairs if e not in edges]
        
        random.shuffle(non_edges)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            random.shuffle(non_edges)
            
            # Filter to current non-edges and sort by degree sum
            current_non_edges = [e for e in non_edges if not edge_matrix[e[0]][e[1]]]
            current_non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
            
            for e in current_non_edges:
                if time.time() > deadline:
                    break
                u, v = e
                remaining = [x for x in range(16) if x != u and x != v]
                safe = True
                
                # Temporarily add edge
                edge_matrix[u][v] = True
                edge_matrix[v][u] = True
                edges.add((u, v))
                
                # Check for Q3: only need to check 8-sets containing both u and v
                for combo in itertools.combinations(remaining, 6):
                    verts = [u, v] + list(combo)
                    if contains_q3_fast(verts, edge_matrix):
                        safe = False
                        break
                
                if not safe:
                    edge_matrix[u][v] = False
                    edge_matrix[v][u] = False
                    edges.remove((u, v))
                else:
                    changed = True
                    
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
    
    # Second phase: Kicks - more aggressive with smaller kicks
    kick_seed = 100
    while time.time() < deadline - 5:
        # Alternate between removing 1, 2, and 3 edges
        kick_size = random.choice([1, 1, 2, 2, 3])
        
        perturbed_edges = set(best_edges)
        edges_list = list(perturbed_edges)
        random.shuffle(edges_list)
        for e in edges_list[:kick_size]:
            perturbed_edges.remove(e)
            
        current_edges = run_greedy(perturbed_edges, kick_seed, deadline)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
        kick_seed += 1

    return [list(e) for e in best_edges]
