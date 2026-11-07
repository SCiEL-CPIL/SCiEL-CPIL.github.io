import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Pre-compute initial adjacency as bitmasks
    init_adj = [0] * N
    for u, v in P['edges']:
        init_adj[u] |= (1 << v)
        init_adj[v] |= (1 << u)
        
    best_bicliques = None
    best_count = float('inf')
    
    # Helper to convert bitmask to list of ints
    def bits_to_list(mask):
        res = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res.append(idx)
            m ^= lsb
        return res

    # Run multiple restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Random seed for this restart
        random.seed(restart)
        
        adj = init_adj[:]
        bicliques = []
        
        # Greedy loop
        while True:
            if time.time() > deadline:
                break
                
            # Check if graph is empty
            is_empty = True
            for u in range(N):
                if adj[u]:
                    is_empty = False
                    break
            if is_empty:
                break
                
            best_val = -1
            best_L = None
            best_R_mask = 0
            candidates = []
            
            # 1. Single vertices (Stars)
            for u in range(N):
                if adj[u]:
                    r_mask = adj[u]
                    val = r_mask.bit_count()
                    if val > 0:
                        candidates.append((val, [u], r_mask))
                        
            # 2. Pairs
            for u in range(N):
                if not adj[u]: continue
                for v in range(u + 1, N):
                    if not adj[v]: continue
                    r_mask = adj[u] & adj[v]
                    if r_mask:
                        val = 2 * r_mask.bit_count()
                        candidates.append((val, [u, v], r_mask))
                        
            # 3. Triples (Optional, might be too slow if many, but N=30 is small)
            # Pruning: only consider if potential is high
            for u in range(N):
                if not adj[u]: continue
                for v in range(u + 1, N):
                    if not adj[v]: continue
                    common_uv = adj[u] & adj[v]
                    if not common_uv: continue
                    # Pruning: max possible val is 3 * bit_count(common_uv)
                    if 3 * common_uv.bit_count() <= best_val:
                        continue
                    for w in range(v + 1, N):
                        if not adj[w]: continue
                        r_mask = common_uv & adj[w]
                        if r_mask:
                            val = 3 * r_mask.bit_count()
                            candidates.append((val, [u, v, w], r_mask))
                            
            if not candidates:
                # Should not happen if is_empty is False, but just in case
                break
                
            # Find max value to filter top candidates
            max_val = max(c[0] for c in candidates)
            
            # Keep top K candidates (e.g., top 20 or those within 10% of max)
            # Randomization: pick randomly from top candidates
            top_candidates = [c for c in candidates if c[0] >= max_val * 0.9]
            
            # If no top candidates (shouldn't happen), take max
            if not top_candidates:
                top_candidates = [max(candidates, key=lambda x: x[0])]
                
            # Randomly select one
            val, L, R_mask = random.choice(top_candidates)
            
            # Add biclique
            bicliques.append([L, bits_to_list(R_mask)])
            
            # Remove edges from adj
            for u in L:
                # Clear bits in adj[u] corresponding to R_mask
                adj[u] &= ~R_mask
                
                # Clear corresponding bits in adj[r] for all r in R_mask
                m = R_mask
                while m:
                    lsb = m & -m
                    r_idx = lsb.bit_length() - 1
                    adj[r_idx] &= ~(1 << u)
                    m ^= lsb
            
            # Early termination if we already have more bicliques than best found
            if len(bicliques) >= best_count:
                break
                
        # Check if we completed the partition
        if not any(adj):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques
                
    # If no valid solution found (should not happen), fallback to stars
    if best_bicliques is None:
        # Simple star decomposition
        adj = init_adj[:]
        bicliques = []
        for u in range(N):
            if adj[u]:
                bicliques.append([[u], bits_to_list(adj[u])])
                # Clear edges
                m = adj[u]
                while m:
                    lsb = m & -m
                    r_idx = lsb.bit_length() - 1
                    adj[r_idx] &= ~(1 << u)
                    adj[u] &= ~lsb
                    m ^= lsb
        best_bicliques = bicliques
        
    # Final type conversion
    final_bicliques = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        final_bicliques.append([left, right])
        
    return final_bicliques

# EVOLVE-BLOCK-END
