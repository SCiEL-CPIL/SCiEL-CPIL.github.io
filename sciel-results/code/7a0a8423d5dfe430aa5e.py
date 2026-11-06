import time
import random
import math

def construct():
    N = 128
    start_time = time.time()
    deadline = 55.0

    # Precompute 4-APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a, d))
    
    pos_to_aps = [[] for _ in range(N + 1)]
    for idx, (a, d) in enumerate(aps):
        for j in range(4):
            pos = a + j * d
            pos_to_aps[pos].append((idx, j))

    def count_violations(colors):
        v = 0
        for a, d in aps:
            c1 = colors[a-1]
            c2 = colors[a+d-1]
            c3 = colors[a+2*d-1]
            c4 = colors[a+3*d-1]
            if len({c1, c2, c3, c4}) < 3:
                v += 1
        return v

    def solve_with_k(K, time_limit):
        end_time = time.time() + time_limit
        best_viol = float('inf')
        best_colors = None
        
        while time.time() < end_time:
            colors = [random.randint(0, K - 1) for _ in range(N)]
            viol = count_violations(colors)
            
            if viol == 0:
                return colors
            
            if viol < best_viol:
                best_viol = viol
                best_colors = colors[:]
                
            T = 5.0
            T_min = 0.01
            cooling_rate = 0.9995
            
            while T > T_min and time.time() < end_time:
                i = random.randint(1, N)
                new_c = random.randint(0, K - 1)
                old_c = colors[i - 1]
                
                if new_c == old_c:
                    continue
                
                delta_viol = 0
                for ap_idx, j in pos_to_aps[i]:
                    a, d = aps[ap_idx]
                    c1 = colors[a-1]
                    c2 = colors[a+d-1]
                    c3 = colors[a+2*d-1]
                    c4 = colors[a+3*d-1]
                    
                    old_dist = len({c1, c2, c3, c4})
                    
                    if j == 0: c1 = new_c
                    elif j == 1: c2 = new_c
                    elif j == 2: c3 = new_c
                    else: c4 = new_c
                    
                    new_dist = len({c1, c2, c3, c4})
                    
                    if old_dist < 3 and new_dist >= 3:
                        delta_viol -= 1
                    elif old_dist >= 3 and new_dist < 3:
                        delta_viol += 1
                    
                if delta_viol < 0 or (random.random() < math.exp(-delta_viol / T)):
                    colors[i - 1] = new_c
                    viol += delta_viol
                    if viol == 0:
                        return colors
                    if viol < best_viol:
                        best_viol = viol
                        best_colors = colors[:]
                        
                T *= cooling_rate
                
            # Restart with new random init
            if time.time() > end_time - 2.0:
                break

        return best_colors

    # Try K=3
    time_for_k3 = 25.0
    sol3 = solve_with_k(3, time_for_k3)
    if sol3:
        return sol3

    # Try K=4
    remaining_time = deadline - (time.time() - start_time) - 5.0
    if remaining_time > 1.0:
        sol4 = solve_with_k(4, remaining_time)
        if sol4:
            return sol4

    # Fallback
    return [random.randint(0, 3) for _ in range(N)]
