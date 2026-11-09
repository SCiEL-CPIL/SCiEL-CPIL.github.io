import time
import sys
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build original adjacency as bitmasks for fast intersection
    orig_adj_mask = [0] * n
    for u, v in edges:
        orig_adj_mask[u] |= (1 << v)
        orig_adj_mask[v] |= (1 << u)
    
    def mask_to_list(mask):
        """Convert a bitmask to a list of vertex indices."""
        result = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            result.append(idx)
            m ^= lsb
        return result
    
    def greedy_solve(seed):
        random.seed(seed)
        
        # Track uncovered edges as bitmasks
        uncovered_mask = [0] * n
        for u, v in edges:
            uncovered_mask[u] |= (1 << v)
            uncovered_mask[v] |= (1 << u)
            
        bicliques = []
        
        while any(uncovered_mask[u] for u in range(n)):
            if time.time() > deadline:
                return None  # Timeout in this run
                
            active = [u for u in range(n) if uncovered_mask[u]]
            if not active:
                break
                
            # Sort by uncovered degree
            active_sorted = sorted(active, key=lambda x: bin(uncovered_mask[x]).count('1'), reverse=True)
            
            candidates = []
            
            # Size 1: star from each active vertex
            for u in active_sorted:
                R_mask = uncovered_mask[u] & ~(1 << u)
                covered = bin(R_mask).count('1')
                candidates.append((covered, [u], mask_to_list(R_mask)))
                
            # Size 2: pairs of active vertices
            K = min(25, len(active_sorted))
            top_active = active_sorted[:K]
            for i in range(len(top_active)):
                u = top_active[i]
                for j in range(i + 1, len(top_active)):
                    v = top_active[j]
                    R_mask = uncovered_mask[u] & uncovered_mask[v]
                    R_mask &= ~(1 << u)
                    R_mask &= ~(1 << v)
                    if R_mask:
                        covered = 2 * bin(R_mask).count('1')
                        candidates.append((covered, [u, v], mask_to_list(R_mask)))
                        
            # Size 3: triples of active vertices
            M = min(12, len(active_sorted))
            top_active_m = active_sorted[:M]
            for i in range(len(top_active_m)):
                u = top_active_m[i]
                for j in range(i + 1, len(top_active_m)):
                    v = top_active_m[j]
                    common_uv = uncovered_mask[u] & uncovered_mask[v]
                    common_uv &= ~(1 << u)
                    common_uv &= ~(1 << v)
                    if not common_uv:
                        continue
                    # Candidate w's from common neighbors
                    w_list = mask_to_list(common_uv)
                    if len(w_list) > 10:
                        w_list = sorted(w_list, key=lambda x: bin(uncovered_mask[x]).count('1'), reverse=True)[:10]
                    for w in w_list:
                        R_mask = common_uv & uncovered_mask[w]
                        R_mask &= ~(1 << u)
                        R_mask &= ~(1 << v)
                        R_mask &= ~(1 << w)
                        if R_mask:
                            covered = 3 * bin(R_mask).count('1')
                            candidates.append((covered, [u, v, w], mask_to_list(R_mask)))
                            
            if not candidates:
                for u in active:
                    R_list = mask_to_list(uncovered_mask[u] & ~(1 << u))
                    if R_list:
                        candidates.append((1, [u], R_list[:1]))
                    break
            
            # Sort candidates by coverage, then randomize ties
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from top candidates to introduce randomness
            top_k = min(5, len(candidates))
            chosen = random.choice(candidates[:top_k])
            
            _, best_L, best_R = chosen
            
            # Validate: every cross pair must be an original edge
            valid = True
            for u in best_L:
                for v in best_R:
                    if not (orig_adj_mask[u] & (1 << v)):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                for u in active:
                    R_list = mask_to_list(uncovered_mask[u] & ~(1 << u))
                    if R_list:
                        best_L = [u]
                        best_R = R_list[:1]
                        break
            
            bicliques.append([best_L, best_R])
            
            # Update uncovered masks
            for u in best_L:
                for v in best_R:
                    uncovered_mask[u] &= ~(1 << v)
                    uncovered_mask[v] &= ~(1 << u)
                        
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts
    seed = 42
    while time.time() < deadline:
        result = greedy_solve(seed)
        if result is not None:
            count = len(result)
            if count < best_count:
                best_count = count
                best_bicliques = result
        seed += 1
        if time.time() > deadline - 5:
            break

    if best_bicliques is None:
        trivial = []
        for u, v in edges:
            trivial.append([[u], [v]])
        return trivial

    result = []
    for L, R in best_bicliques:
        result.append([[int(x) for x in L], [int(y) for y in R]])
        
    return result
