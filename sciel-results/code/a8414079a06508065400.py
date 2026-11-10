import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 18
    
    # Precompute all subsets of size 12 as bitmasks
    subsets_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_12.append(mask)
    
    def is_triangle_free(mask, adj):
        """Check if subgraph induced by mask is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True
    
    def find_all_triangle_free_12(adj):
        """Return list of masks of size-12 subsets that are triangle-free."""
        result = []
        for mask in subsets_12:
            if is_triangle_free(mask, adj):
                result.append(mask)
        return result
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            if (adj[w] & common) != 0:
                return False
        return True
    
    def generate_random_max_k4_free():
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    # Seed: complement of K6,6,6 (i.e., 3 disjoint K6's)
    # This has triangle-free induced subgraph of size 12 (one part from two K6's)
    seed_adj = [0] * n
    for i in range(6):
        for j in range(i+1, 6):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(i+1, 12):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(12, 18):
        for j in range(i+1, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    
    best_edges = adj_to_edges(seed_adj)
    best_score = 12
    
    def local_search(adj):
        """Try to eliminate all triangle-free 12-subsets by adding edges."""
        adj = list(adj)  # copy
        max_iter = 200
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            tf_masks = find_all_triangle_free_12(adj)
            if not tf_masks:
                return adj
            # Pick a random triangle-free 12-subset
            mask = random.choice(tf_masks)
            # Find non-edges within this subset
            non_edges = []
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                for v in range(u+1, n):
                    if (mask & (1 << v)) and not (adj[u] & (1 << v)):
                        non_edges.append((u, v))
            if not non_edges:
                break
            # Try to add a random non-edge that doesn't create K4
            random.shuffle(non_edges)
            added = False
            for u, v in non_edges:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            if not added:
                # Try removing an edge to allow future additions? Or just break
                break
        return adj
    
    # Phase 1: Try random graphs with local search
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        adj = generate_random_max_k4_free()
        
        # Quick check: does it have any triangle-free 12-subset?
        tf_count = 0
        for mask in subsets_12:
            if is_triangle_free(mask, adj):
                tf_count += 1
                if tf_count > 50:  # Too many, skip
                    break
        
        if tf_count == 0:
            return adj_to_edges(adj)
        
        if tf_count <= 50:
            # Promising: run local search
            adj_ls = local_search(adj)
            tf_final = find_all_triangle_free_12(adj_ls)
            if not tf_final:
                return adj_to_edges(adj_ls)
    
    # Phase 2: Try local search from seed
    adj_seed = list(seed_adj)
    adj_ls = local_search(adj_seed)
    tf_final = find_all_triangle_free_12(adj_ls)
    if not tf_final:
        return adj_to_edges(adj_ls)
    
    # Phase 3: More random restarts with local search
    while time.time() < deadline - 2:
        adj = generate_random_max_k4_free()
        adj_ls = local_search(adj)
        tf_final = find_all_triangle_free_12(adj_ls)
        if not tf_final:
            return adj_to_edges(adj_ls)
    
    return best_edges

# EVOLVE-BLOCK-END
