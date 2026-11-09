import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n_base = 29
    exponent = (n_base - 1) // 2
    
    # Precompute Paley graph adjacency for 29 vertices
    # adj[i][j] = 1 if (j-i) is QR mod 29, else 0
    # Note: (i,j) is symmetric.
    adj = [[0] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % n_base
            res = pow(diff, exponent, n_base)
            if res == 1:
                adj[i][j] = 1
                adj[j][i] = 1
            else:
                adj[i][j] = 0
                adj[j][i] = 0

    # Convert to numpy arrays for vectorized operations
    adj_np = np.array(adj, dtype=np.int8)
    # adj_comp[i][j] = 1 - adj[i][j] for i != j, 0 for i == j
    adj_comp_np = np.zeros((n_base, n_base), dtype=np.int8)
    np.fill_diagonal(adj_comp_np, 0)
    adj_comp_np = 1 - adj_np
    np.fill_diagonal(adj_comp_np, 0)

    # Precompute bitmasks for vertices
    # bit_mask[v] is an int with bit v set
    bit_masks = [1 << v for v in range(n_base)]
    
    # Function to count K2,2 in a graph induced by subset S (mask)
    # Using vectorized numpy operations where possible, but mask filtering is key.
    # For n=29, we can precompute common neighbors for all pairs? 
    # common_neighbors[u][v] = bitmask of w such that adj[u][w] and adj[v][w]
    # Then count_k22(mask) = sum_{u<v in mask} C(popcount(common_neighbors[u][v] & mask), 2)
    
    common_adj = [0] * (n_base * n_base)
    common_comp = [0] * (n_base * n_base)
    
    for u in range(n_base):
        for v in range(u + 1, n_base):
            mask_common = 0
            for w in range(n_base):
                if w != u and w != v:
                    if adj[u][w] and adj[v][w]:
                        mask_common |= bit_masks[w]
            common_adj[u * n_base + v] = mask_common
            
            mask_common_comp = 0
            for w in range(n_base):
                if w != u and w != v:
                    if adj_comp_np[u][w] and adj_comp_np[v][w]:
                        mask_common_comp |= bit_masks[w]
            common_comp[u * n_base + v] = mask_common_comp

    def count_k22_fast(mask, common_arr):
        count = 0
        # Iterate over pairs in mask
        # Optimized: iterate u, v
        for u in range(n_base):
            if not (mask & bit_masks[u]):
                continue
            for v in range(u + 1, n_base):
                if not (mask & bit_masks[v]):
                    continue
                # Get common neighbors in mask
                cn_mask = common_arr[u * n_base + v] & mask
                # Count bits
                c = bin(cn_mask).count('1')
                if c >= 2:
                    count += c * (c - 1) // 2
        return count

    def get_cost(mask):
        cost1 = count_k22_fast(mask, common_adj)
        mask_inv = ((1 << n_base) - 1) ^ mask
        cost2 = count_k22_fast(mask_inv, common_comp)
        return cost1 + cost2

    # Local Search
    # Start with random or specific
    # Try Q U {0}
    # Q = {1, 2, 4, 8, 16, 7, 14, 28, 27, 25, 21, 13, 9, 18}
    q_set = {1, 2, 4, 8, 16, 7, 14, 28, 27, 25, 21, 13, 9, 18}
    init_mask = 1 # vertex 0
    for v in q_set:
        init_mask |= bit_masks[v]
        
    best_mask = init_mask
    best_cost = get_cost(best_mask)
    
    if best_cost > 0:
        # Hill Climbing
        current_mask = init_mask
        current_cost = best_cost
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping each bit
            # Shuffle order to break symmetry?
            bits = list(range(n_base))
            random.shuffle(bits)
            for b in bits:
                new_mask = current_mask ^ bit_masks[b]
                new_cost = get_cost(new_mask)
                if new_cost < current_cost:
                    current_mask = new_mask
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        break
            if current_cost == 0:
                break
            if not improved:
                # Random restart or kick
                # Just pick a random mask
                current_mask = random.getrandbits(n_base)
                current_cost = get_cost(current_mask)
                
        if current_cost < best_cost:
            best_mask = current_mask
            best_cost = current_cost

    if best_cost == 0:
        n = 30
        colors = []
        # Edges 0..28
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(adj[i][j])
        # Edges 29..i
        for i in range(n_base):
            if (best_mask >> i) & 1:
                colors.append(0)
            else:
                colors.append(1)
        return {'n': n, 'colors': colors}
    else:
        # Fallback to n=29
        n = 29
        colors = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(adj[i][j])
        return {'n': n, 'colors': colors}
