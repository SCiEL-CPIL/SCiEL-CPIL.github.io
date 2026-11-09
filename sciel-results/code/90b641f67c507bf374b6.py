import numpy as np
import random
import time

def construct():
    n = 128
    
    # Precompute 4-APs
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for i in range(1, n - 3 * d + 1):
            i0 = i - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    
    aps = np.array(aps, dtype=np.int32)
    M = len(aps)
    
    idx0 = aps[:, 0]
    idx1 = aps[:, 1]
    idx2 = aps[:, 2]
    idx3 = aps[:, 3]
    
    def is_invalid_ap(c1, c2, c3, c4):
        if c1 == c2 and c2 == c3 and c3 == c4:
            return True
        if c1 == c2 and c2 == c3 and c3 != c4:
            return True
        if c1 == c2 and c2 == c4 and c1 != c3:
            return True
        if c1 == c3 and c3 == c4 and c1 != c2:
            return True
        if c2 == c3 and c3 == c4 and c2 != c1:
            return True
        if c1 == c2 and c3 == c4 and c1 != c3:
            return True
        if c1 == c3 and c2 == c4 and c1 != c2:
            return True
        if c1 == c4 and c2 == c3 and c1 != c2:
            return True
        return False

    def calculate_cost(colors):
        c1 = colors[idx0]
        c2 = colors[idx1]
        c3 = colors[idx2]
        c4 = colors[idx3]
        
        all_same = (c1 == c2) & (c2 == c3) & (c3 == c4)
        case1 = (c1 == c2) & (c2 == c3) & (c3 != c4)
        case2 = (c1 == c2) & (c2 == c4) & (c1 != c3)
        case3 = (c1 == c3) & (c3 == c4) & (c1 != c2)
        case4 = (c2 == c3) & (c3 == c4) & (c2 != c1)
        case5 = (c1 == c2) & (c3 == c4) & (c1 != c3)
        case6 = (c1 == c3) & (c2 == c4) & (c1 != c2)
        case7 = (c1 == c4) & (c2 == c3) & (c1 != c2)
        
        invalid = all_same | case1 | case2 | case3 | case4 | case5 | case6 | case7
        return np.sum(invalid)

    pos_to_aps = [[] for _ in range(n)]
    for ap_idx in range(M):
        for j in range(4):
            pos = aps[ap_idx, j]
            pos_to_aps[pos].append(ap_idx)

    def get_cost_delta(colors, pos, new_color):
        old_color = colors[pos]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ap_idx in pos_to_aps[pos]:
            p0, p1, p2, p3 = aps[ap_idx]
            
            c0 = colors[p0]
            c1 = colors[p1]
            c2 = colors[p2]
            c3 = colors[p3]
            
            was_invalid = is_invalid_ap(c0, c1, c2, c3)
            
            if pos == p0:
                nc0, nc1, nc2, nc3 = new_color, c1, c2, c3
            elif pos == p1:
                nc0, nc1, nc2, nc3 = c0, new_color, c2, c3
            elif pos == p2:
                nc0, nc1, nc2, nc3 = c0, c1, new_color, c3
            else:
                nc0, nc1, nc2, nc3 = c0, c1, c2, new_color
            
            is_invalid_now = is_invalid_ap(nc0, nc1, nc2, nc3)
            
            if was_invalid and not is_invalid_now:
                delta -= 1
            elif not was_invalid and is_invalid_now:
                delta += 1
                
        return delta

    def sa(k, max_iters, start_temp=1.0, end_temp=0.001):
        best_solution = None
        best_cost = float('inf')
        
        for restart in range(50): # More restarts
            current = np.random.randint(0, k, n)
            current_cost = calculate_cost(current)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current.copy()
            
            if current_cost == 0:
                return current
            
            best_in_run = current.copy()
            best_cost_in_run = current_cost
            
            T = start_temp
            cooling_rate = (end_temp / start_temp) ** (1.0 / max_iters)
            
            for i in range(max_iters):
                pos = np.random.randint(0, n)
                old_color = current[pos]
                
                new_color = np.random.randint(0, k)
                while new_color == old_color:
                    new_color = np.random.randint(0, k)
                
                delta = get_cost_delta(current, pos, new_color)
                
                if delta <= 0:
                    current[pos] = new_color
                    current_cost += delta
                    if current_cost < best_cost_in_run:
                        best_cost_in_run = current_cost
                        best_in_run = current.copy()
                        if best_cost_in_run == 0:
                            return best_in_run
                else:
                    if np.random.rand() < np.exp(-delta / T):
                        current[pos] = new_color
                        current_cost += delta
                
                T *= cooling_rate
            
            if best_cost_in_run < best_cost:
                best_cost = best_cost_in_run
                best_solution = best_in_run.copy()
                
        return best_solution

    start_time = time.time()
    time_limit = 55.0
    global_best_solution = None
    global_best_cost = float('inf')
    
    for k in range(3, 129):
        if time.time() - start_time > time_limit:
            break
            
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        
        # Estimate iterations
        # Each iteration involves ~100-200 AP checks.
        # Let's assume 1000 iters/sec is safe.
        est_iters = int(time_remaining * 1000)
        if est_iters < 1000:
            est_iters = 1000
            
        if k == 3:
            max_iters = min(est_iters, 100000)
        else:
            max_iters = min(est_iters, 50000)
            
        solution = sa(k, max_iters=max_iters)
        
        cost = calculate_cost(solution)
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_solution = solution
            
        if cost == 0:
            return solution.tolist()
            
    if global_best_solution is not None:
        return global_best_solution.tolist()
    else:
        return np.random.randint(0, 3, n).tolist()
