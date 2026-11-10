import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    def solve_for_n(n, time_budget):
        if n < 4:
            return {'n': n, 'colors': [0] * (n*(n-1)*(n-2)//6)}
        
        # 1. Generate n distinct non-zero vectors in GF(3)^4
        vectors = []
        seen = set()
        # Shuffle to avoid bias? Or just random.
        # To ensure we get distinct vectors quickly, we can generate all and pick n
        # But 80 is small.
        all_vectors = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if i==0 and j==0 and k==0 and l==0:
                            continue
                        all_vectors.append((i,j,k,l))
        
        random.shuffle(all_vectors)
        vectors = all_vectors[:n]
        
        # 2. Initialize colors
        m = n*(n-1)*(n-2)//6
        colors = [0] * m
        
        idx_map = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx_map[(a,b,c)] = idx
                    va = vectors[a]
                    vb = vectors[b]
                    vc = vectors[c]
                    s0 = (va[0] + vb[0] + vc[0]) % 3
                    s1 = (va[1] + vb[1] + vc[1]) % 3
                    s2 = (va[2] + vb[2] + vc[2]) % 3
                    s3 = (va[3] + vb[3] + vc[3]) % 3
                    colors[idx] = 0 if (s0==0 and s1==0 and s2==0 and s3==0) else 1
                    idx += 1
                    
        # 3. Precompute K4s and K5s
        k4s = []
        k5s = []
        tri_to_k4 = [[] for _ in range(m)]
        tri_to_k5 = [[] for _ in range(m)]
        
        # K4s
        k4_idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        t1 = idx_map[(a,b,c)]
                        t2 = idx_map[(a,b,d)]
                        t3 = idx_map[(a,c,d)]
                        t4 = idx_map[(b,c,d)]
                        k4s.append((t1, t2, t3, t4))
                        tri_to_k4[t1].append(k4_idx)
                        tri_to_k4[t2].append(k4_idx)
                        tri_to_k4[t3].append(k4_idx)
                        tri_to_k4[t4].append(k4_idx)
                        k4_idx += 1
                        
        # K5s
        k5_idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            tris = [
                                idx_map[(a,b,c)], idx_map[(a,b,d)], idx_map[(a,b,e)],
                                idx_map[(a,c,d)], idx_map[(a,c,e)], idx_map[(a,d,e)],
                                idx_map[(b,c,d)], idx_map[(b,c,e)], idx_map[(b,d,e)],
                                idx_map[(c,d,e)]
                            ]
                            k5s.append(tris)
                            for t in tris:
                                tri_to_k5[t].append(k5_idx)
                            k5_idx += 1
                            
        # 4. Initialize counts
        red_counts = [0] * len(k4s)
        blue_counts = [0] * len(k5s)
        
        for i, k4 in enumerate(k4s):
            cnt = 0
            for t in k4:
                if colors[t] == 0:
                    cnt += 1
            red_counts[i] = cnt
            
        for i, k5 in enumerate(k5s):
            cnt = 0
            for t in k5:
                if colors[t] == 1:
                    cnt += 1
            blue_counts[i] = cnt
            
        cost = sum(1 for c in red_counts if c == 4) + sum(1 for c in blue_counts if c == 10)
        
        best_cost = cost
        best_colors = colors.copy()
        
        # 5. Simulated Annealing
        T = 1.0
        cooling = 0.99995
        min_T = 0.005
        
        # Local variables for speed
        r_counts = red_counts
        b_counts = blue_counts
        t2k4 = tri_to_k4
        t2k5 = tri_to_k5
        cols = colors
        
        while time.time() < time_budget:
            if cost == 0:
                return {'n': n, 'colors': cols}
                
            t_idx = random.randint(0, m-1)
            
            old_color = cols[t_idx]
            new_color = 1 - old_color
            
            delta_cost = 0
            
            # Update K4s
            for k4_i in t2k4[t_idx]:
                old_c = r_counts[k4_i]
                if old_color == 0:
                    # Red -> Blue
                    new_c = old_c - 1
                    if old_c == 4:
                        delta_cost -= 1
                    # new_c == 4 is impossible (3 != 4)
                    r_counts[k4_i] = new_c
                else:
                    # Blue -> Red
                    new_c = old_c + 1
                    if new_c == 4:
                        delta_cost += 1
                    r_counts[k4_i] = new_c
                    
            # Update K5s
            for k5_i in t2k5[t_idx]:
                old_c = b_counts[k5_i]
                if old_color == 1:
                    # Blue -> Red
                    new_c = old_c - 1
                    if old_c == 10:
                        delta_cost -= 1
                    b_counts[k5_i] = new_c
                else:
                    # Red -> Blue
                    new_c = old_c + 1
                    if new_c == 10:
                        delta_cost += 1
                    b_counts[k5_i] = new_c
                    
            # Accept?
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                cols[t_idx] = new_color
                cost += delta_cost
                if cost < best_cost:
                    best_cost = cost
                    best_colors = cols.copy()
            else:
                # Revert
                for k4_i in t2k4[t_idx]:
                    if old_color == 0:
                        r_counts[k4_i] += 1
                    else:
                        r_counts[k4_i] -= 1
                for k5_i in t2k5[t_idx]:
                    if old_color == 1:
                        b_counts[k5_i] += 1
                    else:
                        b_counts[k5_i] -= 1
                        
            T *= cooling
            if T < min_T:
                T = min_T
                
        if best_cost == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # Try n values in descending order
    # 35 is likely impossible, but worth a quick try if time permits?
    # Actually, 34 is the known bound.
    # Let's try 34 first to ensure we get a high score quickly.
    # Then if time remains, try 35.
    
    best_result = None
    
    # Try 34
    if time.time() < deadline - 20:
        res = solve_for_n(34, deadline - 10)
        if res:
            best_result = res
    
    # Try 35 if 34 succeeded and we have time? Or if 34 failed?
    # If 34 succeeded, we have a score of 2.0 (assuming seed 17).
    # Trying 35 might yield 2.05.
    # But 35 is likely impossible.
    # Let's try 35 ONLY if we have plenty of time and 34 succeeded.
    # Or just try 35 first with a short time limit?
    
    # Strategy:
    # 1. Try 35 with 20s limit.
    # 2. If 35 fails, try 34 with remaining time.
    # 3. If 35 succeeds, return 35.
    
    # Let's adjust the order.
    # If we try 35 first and it takes 20s and fails, we have 120s for 34.
    # If we try 34 first and it takes 5s and succeeds, we have 135s for 35.
    # Since 34 is much more likely to succeed, we should prioritize it.
    # However, if we find 34 quickly, we can spend the rest on 35.
    
    # Revised Plan:
    # 1. Solve 34 with a generous budget (e.g., 60s).
    # 2. If 34 succeeds, save it.
    # 3. If time remains (> 20s), try 35.
    # 4. If 35 succeeds, return 35. Else return 34.
    # 5. If 34 fails, try 33, 32...
    
    current_best = None
    
    # Try 34
    if time.time() < deadline - 10:
        res34 = solve_for_n(34, min(deadline - 20, time.time() + 60)) # Max 60s
        if res34:
            current_best = res34
    
    # Try 35 if 34 worked and time is left
    if current_best and time.time() < deadline - 5:
        res35 = solve_for_n(35, deadline - 5)
        if res35:
            current_best = res35
    elif not current_best:
        # 34 failed, try smaller
        for n in [33, 32, 31, 30]:
            if time.time() > deadline - 5:
                break
            res = solve_for_n(n, deadline - 5)
            if res:
                current_best = res
                break
                
    if current_best:
        return current_best
        
    # Fallback: n=9
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in __import__('itertools').combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
