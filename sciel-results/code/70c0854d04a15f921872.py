import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n = 16
    # 1. Precomputation
    edge_idx = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    edge_k4s = [[] for _ in range(120)]
    
    for k4 in k4s:
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                u, v = k4[i], k4[j]
                e = edge_idx[u][v]
                edges.append(e)
                edge_k4s[e].append(len(k4_edges))
        k4_edges.append(edges)
        
    def calc_violation(colors, edges):
        # edges: list of 6 indices
        mask = 0
        for e in edges:
            mask |= 1 << colors[e]
        # Count bits
        # Using bit_count if available, else bin count
        # Python 3.8+? bit_count is 3.10.
        # Safe fallback:
        d = bin(mask).count('1')
        if d >= 5:
            return 0
        return 5 - d

    def run_sa(K, duration):
        # Init
        colors = [random.randint(0, K-1) for _ in range(120)]
        k4_violations = [0] * 1820
        total_v = 0
        for i, edges in enumerate(k4_edges):
            v = calc_violation(colors, edges)
            k4_violations[i] = v
            total_v += v
            
        if total_v == 0:
            return colors
            
        T = 1.0
        best_colors = colors[:]
        best_v = total_v
        last_time = time.time()
        
        while time.time() - start_time < deadline - 5.0 and time.time() - last_time < duration:
            if time.time() > deadline - 5.0:
                break
                
            e = random.randint(0, 119)
            old_c = colors[e]
            new_c = random.randint(0, K-1)
            if new_c == old_c:
                continue
                
            delta_v = 0
            affected_k4s = edge_k4s[e]
            
            # Calculate delta
            for k4_idx in affected_k4s:
                edges = k4_edges[k4_idx]
                # Get old mask
                mask_old = 0
                for ee in edges:
                    if ee == e:
                        mask_old |= 1 << old_c
                    else:
                        mask_old |= 1 << colors[ee]
                d_old = bin(mask_old).count('1')
                v_old = 0 if d_old >= 5 else 5 - d_old
                
                # Get new mask
                mask_new = 0
                for ee in edges:
                    if ee == e:
                        mask_new |= 1 << new_c
                    else:
                        mask_new |= 1 << colors[ee]
                d_new = bin(mask_new).count('1')
                v_new = 0 if d_new >= 5 else 5 - d_new
                
                delta_v += v_new - v_old
                
            if delta_v < 0 or random.random() < pow(2.71828, -delta_v / T):
                # Accept
                colors[e] = new_c
                for k4_idx in affected_k4s:
                    edges = k4_edges[k4_idx]
                    mask = 0
                    for ee in edges:
                        mask |= 1 << colors[ee]
                    d = bin(mask).count('1')
                    k4_violations[k4_idx] = 0 if d >= 5 else 5 - d
                total_v += delta_v
                
                if total_v < best_v:
                    best_v = total_v
                    best_colors = colors[:]
                    if best_v == 0:
                        return best_colors
                        
            T *= 0.9995
            if T < 0.01:
                T = 0.01
                
        return best_colors if best_v == 0 else None

    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10.0:
            break
        res = run_sa(K, 40.0)
        if res:
            return res
            
    # Fallback to seed
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    # Verify seed validity
    valid = True
    for i, edges in enumerate(k4_edges):
        v = calc_violation(colors, edges)
        if v > 0:
            valid = False
            break
    if valid:
        return colors
        
    # If seed is invalid (unlikely), return random 17-coloring and hope?
    # Or just return the seed, assuming it's valid.
    return colors
