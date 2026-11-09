def construct():
    import random
    import numpy as np
    
    N = 16
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = N * N
    
    # Precompute squared distances
    dist = [[0]*n_points for _ in range(n_points)]
    for i in range(n_points):
        xi, yi = all_points[i]
        for j in range(n_points):
            xj, yj = all_points[j]
            dist[i][j] = (xi-xj)**2 + (yi-yj)**2
            
    # Precompute isosceles blocking bitsets
    # C[i][j] is bitset of k such that (i, j, k) is isosceles
    C = [[0]*n_points for _ in range(n_points)]
    
    for i in range(n_points):
        for j in range(n_points):
            if i == j: continue
            mask = 0
            di_j = dist[i][j]
            for k in range(n_points):
                if k == i or k == j: continue
                di_k = dist[i][k]
                dj_k = dist[j][k]
                if di_j == di_k or di_j == dj_k or di_k == dj_k:
                    mask |= (1 << k)
            C[i][j] = mask

    def greedy_construct(seed):
        random.seed(seed)
        S = []
        available = (1 << n_points) - 1
        
        while available:
            best_p = -1
            best_score = -1
            best_new_blocked = 0
            
            # Get available indices
            avail_indices = []
            temp_avail = available
            while temp_avail:
                lsb = temp_avail & -temp_avail
                idx = lsb.bit_length() - 1
                avail_indices.append(idx)
                temp_avail ^= lsb
                
            random.shuffle(avail_indices)
            
            for idx in avail_indices:
                # Calculate new blocked
                new_blocked = 0
                for a_idx in S:
                    new_blocked |= C[a_idx][idx]
                
                removed = (1 << idx) | new_blocked
                count_removed = (available & removed).bit_count()
                
                if count_removed > best_score:
                    best_score = count_removed
                    best_p = idx
                    best_new_blocked = new_blocked
                    if best_score == available.bit_count():
                        break
                        
            if best_p == -1:
                break
                
            S.append(best_p)
            available &= ~((1 << best_p) | best_new_blocked)
            
        return S

    best_set = None
    best_size = n_points + 1
    
    for seed in range(50):
        s = greedy_construct(seed)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    return best_set if best_set else all_points
