import numpy as np
import itertools
import random
import time

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    # 1. Generate all norm-2 integer vectors
    # Type A: one +/- 2
    # Type B: four +/- 1
    
    vectors = []
    ids = {}
    id_to_vec = {}
    
    # Type A
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            idx = len(vectors)
            vectors.append(v)
            ids[tuple(v)] = idx
            id_to_vec[idx] = v
    
    # Type B
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            idx = len(vectors)
            vectors.append(v)
            ids[tuple(v)] = idx
            id_to_vec[idx] = v
            
    N = len(vectors)
    
    # 2. Build adjacency list for conflicts
    # Conflict if dot product > 2
    # We only need to check Type B vs Type B for potential conflicts > 2?
    # Type A vs Type B: max dot 2. No conflict.
    # Type A vs Type A: max dot 0 (distinct). No conflict.
    # So only Type B vs Type B can conflict.
    
    adj = [[] for _ in range(N)]
    
    # To speed up, we can just check all pairs?
    # N=5302. N^2/2 = 14M.
    # Dot product calculation is fast.
    # But we can be smarter.
    
    # Let's just do O(N^2) with numpy for simplicity, as 14M ops is < 1s.
    vec_arr = np.array(vectors, dtype=np.int8)
    # Calculate all pairwise dot products?
    # dot_matrix = vec_arr @ vec_arr.T
    # This is 5302x5302 matrix. 28M entries. ~200MB memory.
    # Might be tight but okay.
    
    # Alternatively, iterate and check.
    # Given 120s limit, we can afford a bit of computation.
    
    # Let's use the property:
    # Conflict if dot > 2.
    
    # We will build the graph only for Type B vertices (indices 22 to N-1)
    # But for the MIS, we can include Type A freely.
    # So we find MIS on Type B subgraph, then add all Type A.
    
    type_b_start = 22
    N_b = N - type_b_start
    
    # Build adjacency for Type B
    # Map global index to local B index
    # local_idx = global_idx - type_b_start
    
    adj_b = [[] for _ in range(N_b)]
    
    # Optimization: Group by support?
    # Or just brute force with early termination?
    # 5280 vertices.
    # For each u, check v > u.
    
    # Let's use numpy to speed up dot products in batches?
    # Or just python loops?
    # 5280 * 5280 / 2 = 14M iterations.
    # Python loop 14M iterations might take 5-10s.
    # Acceptable.
    
    for i in range(N_b):
        ui = type_b_start + i
        vi_global = vec_arr[ui]
        for j in range(i + 1, N_b):
            vj_global = vec_arr[type_b_start + j]
            # Dot product
            # Since they are sparse, we can use the support intersection logic
            # But simple dot is fine.
            dp = np.dot(vi_global, vj_global)
            if dp > 2:
                adj_b[i].append(j)
                adj_b[j].append(i)
                
    # 3. Find Maximum Independent Set on adj_b
    # Heuristic: Greedy + Local Search
    
    def greedy_mis(adj, order):
        in_set = [False] * len(adj)
        set_size = 0
        for v in order:
            # Check if any neighbor is in set
            is_conflict = False
            for u in adj[v]:
                if in_set[u]:
                    is_conflict = True
                    break
            if not is_conflict:
                in_set[v] = True
                set_size += 1
        return in_set, set_size

    def local_search(in_set, adj, n_iter=1000):
        # Try to add vertices
        improved = True
        while improved:
            improved = False
            for v in range(len(adj)):
                if not in_set[v]:
                    is_conflict = False
                    for u in adj[v]:
                        if in_set[u]:
                            is_conflict = True
                            break
                    if not is_conflict:
                        in_set[v] = True
                        improved = True
        # Try to swap?
        # If no vertex can be added, try removing one and adding others?
        # Simple kick: remove a random vertex, then try to add again.
        # We do this in the outer loop.
        return in_set

    best_size = 0
    best_set = None
    
    # Run multiple iterations
    for _ in range(50): # 50 restarts
        order = list(range(N_b))
        random.shuffle(order)
        in_set, size = greedy_mis(adj_b, order)
        
        # Local search improvement
        in_set = local_search(in_set, adj_b)
        size = sum(in_set)
        
        if size > best_size:
            best_size = size
            best_set = in_set.copy()
            
        # If we found a very large set, we can break early?
        # Kissing number 1656.
        # If we get > 1000, it's good.
        
    # Construct final vectors
    final_vectors = []
    # Add all Type A
    for i in range(type_b_start):
        final_vectors.append(vectors[i])
        
    # Add selected Type B
    for i in range(N_b):
        if best_set[i]:
            final_vectors.append(vectors[type_b_start + i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
