import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # We will attempt to find a valid coloring for n=34, then 33, etc.
    # The known bound is 34.
    # Strategy:
    # 1. Precompute structures for n=34.
    # 2. Run a robust simulated annealing / local search with restarts.
    # 3. If successful, return n=34.
    # 4. If not, try n=33 with remaining time.
    # 5. Fallback to small n if all else fails.

    def solve_for_n(n, time_budget):
        if n < 4:
            M = n * (n - 1) * (n - 2) // 6
            return [random.randint(0, 1) for _ in range(M)]
        
        # Precompute indices for fast lookup
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # K4s: 4 vertices, 4 triples. Violation if all 4 red.
        # K5s: 5 vertices, 10 triples. Violation if all 10 blue.
        
        k4_list = list(combinations(range(n), 4))
        num_k4 = len(k4_list)
        k5_list = list(combinations(range(n), 5)) if n >= 5 else []
        num_k5 = len(k5_list)
        
        k4_data = []
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, k4 in enumerate(k4_list):
            ts = []
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                idx = triple_to_idx[t]
                ts.append(idx)
                triple_to_k4s[idx].append(i)
            k4_data.append(ts)
            
        k5_data = []
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, k5 in enumerate(k5_list):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                ts.append(idx)
                triple_to_k5s[idx].append(i)
            k5_data.append(ts)

        k4_arr = np.array(k4_data, dtype=np.int32) if num_k4 > 0 else np.array([], dtype=np.int32).reshape(0,4)
        k5_arr = np.array(k5_data, dtype=np.int32) if num_k5 > 0 else np.array([], dtype=np.int32).reshape(0,10)
        
        # State
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        # Counts:
        # k4_counts[i] = number of RED triples in K4 i. Violation if 4.
        # k5_counts[i] = number of BLUE triples in K5 i. Violation if 10.
        
        # Initial counts
        k4_counts = np.zeros(num_k4, dtype=np.int32)
        k5_counts = np.zeros(num_k5, dtype=np.int32)
        
        # Vectorized init
        if num_k4 > 0:
            k4_counts = np.sum(colors[k4_arr] == 0, axis=1)
        if num_k5 > 0:
            k5_counts = np.sum(colors[k5_arr] == 1, axis=1)
            
        red_viol = np.sum(k4_counts == 4)
        blue_viol = np.sum(k5_counts == 10)
        total_viol = red_viol + blue_viol
        
        if total_viol == 0:
            return colors.tolist()
            
        best_viol = total_viol
        best_colors = colors.copy()
        
        # Simulated Annealing
        # We use a restart strategy with increasing temperature or random restarts.
        
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() - start_time > start_time + time_budget:
                break
                
            # Reset or Perturb
            if r == 0:
                pass # Already initialized
            elif random.random() < 0.5:
                # Random restart
                colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
                if num_k4 > 0:
                    k4_counts = np.sum(colors[k4_arr] == 0, axis=1)
                if num_k5 > 0:
                    k5_counts = np.sum(colors[k5_arr] == 1, axis=1)
                red_viol = np.sum(k4_counts == 4)
                blue_viol = np.sum(k5_counts == 10)
                total_viol = red_viol + blue_viol
                if total_viol == 0:
                    return colors.tolist()
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
            else:
                # Perturb best
                colors = best_colors.copy()
                if num_k4 > 0:
                    k4_counts = np.sum(colors[k4_arr] == 0, axis=1)
                if num_k5 > 0:
                    k5_counts = np.sum(colors[k5_arr] == 1, axis=1)
                red_viol = np.sum(k4_counts == 4)
                blue_viol = np.sum(k5_counts == 10)
                total_viol = red_viol + blue_viol
                
                # Flip some random edges
                num_flips = random.randint(5, 20)
                for _ in range(num_flips):
                    idx = random.randint(0, num_triples - 1)
                    # Inline flip for speed in setup
                    old_val = colors[idx]
                    new_val = 1 - old_val
                    
                    if old_val == 0: # Red -> Blue
                        for k4_idx in triple_to_k4s[idx]:
                            if k4_counts[k4_idx] == 4:
                                red_viol -= 1
                            k4_counts[k4_idx] -= 1
                        for k5_idx in triple_to_k5s[idx]:
                            if k5_counts[k5_idx] == 9:
                                blue_viol += 1
                            k5_counts[k5_idx] += 1
                    else: # Blue -> Red
                        for k4_idx in triple_to_k4s[idx]:
                            if k4_counts[k4_idx] == 3:
                                red_viol += 1
                            k4_counts[k4_idx] += 1
                        for k5_idx in triple_to_k5s[idx]:
                            if k5_counts[k5_idx] == 10:
                                blue_viol -= 1
                            k5_counts[k5_idx] -= 1
                    colors[idx] = new_val
                    total_viol = red_viol + blue_viol
                    if total_viol == 0:
                        return colors.tolist()

            if total_viol < best_viol:
                best_viol = total_viol
                best_colors = colors.copy()
                
            # SA Loop
            T = 2.0
            T_min = 0.05
            cooling = 0.9995
            no_improve = 0
            max_no_improve = 2000
            
            run_start = time.time()
            while time.time() - run_start < 2.0: # 2s per restart phase
                if total_viol == 0:
                    return colors.tolist()
                    
                idx = random.randint(0, num_triples - 1)
                old_val = colors[idx]
                new_val = 1 - old_val
                
                # Calculate delta
                delta_red = 0
                delta_blue = 0
                
                if old_val == 0: # Red -> Blue
                    for k4_idx in triple_to_k4s[idx]:
                        if k4_counts[k4_idx] == 4:
                            delta_red -= 1
                        k4_counts[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[idx]:
                        if k5_counts[k5_idx] == 9:
                            delta_blue += 1
                        k5_counts[k5_idx] += 1
                else: # Blue -> Red
                    for k4_idx in triple_to_k4s[idx]:
                        if k4_counts[k4_idx] == 3:
                            delta_red += 1
                        k4_counts[k4_idx] += 1
                    for k5_idx in triple_to_k5s[idx]:
                        if k5_counts[k5_idx] == 10:
                            delta_blue -= 1
                        k5_counts[k5_idx] -= 1
                
                colors[idx] = new_val
                red_viol += delta_red
                blue_viol += delta_blue
                total_viol = red_viol + blue_viol
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    no_improve = 0
                    if total_viol == 0:
                        return colors.tolist()
                else:
                    # Accept worse with prob exp(-delta/T)
                    # delta = total_viol - (total_viol - delta_red - delta_blue)
                    # old_total = total_viol - delta_red - delta_blue
                    delta = delta_red + delta_blue
                    if delta > 0:
                        if T > T_min:
                            prob = np.exp(-delta / T)
                            if random.random() < prob:
                                no_improve += 1
                            else:
                                # Revert
                                # Flip back
                                old_val_rev = colors[idx]
                                new_val_rev = 1 - old_val_rev
                                
                                if new_val_rev == 0: # Blue -> Red (reverting Red->Blue)
                                    for k4_idx in triple_to_k4s[idx]:
                                        if k4_counts[k4_idx] == 3:
                                            red_viol += 1
                                        k4_counts[k4_idx] += 1
                                    for k5_idx in triple_to_k5s[idx]:
                                        if k5_counts[k5_idx] == 10:
                                            blue_viol -= 1
                                        k5_counts[k5_idx] -= 1
                                else: # Red -> Blue (reverting Blue->Red)
                                    for k4_idx in triple_to_k4s[idx]:
                                        if k4_counts[k4_idx] == 4:
                                            red_viol -= 1
                                        k4_counts[k4_idx] -= 1
                                    for k5_idx in triple_to_k5s[idx]:
                                        if k5_counts[k5_idx] == 9:
                                            blue_viol += 1
                                        k5_counts[k5_idx] += 1
                                colors[idx] = new_val_rev
                                red_viol -= delta_red
                                blue_viol -= delta_blue
                                total_viol = red_viol + blue_viol
                                no_improve += 1
                        else:
                            # Revert
                            old_val_rev = colors[idx]
                            new_val_rev = 1 - old_val_rev
                            
                            if new_val_rev == 0:
                                for k4_idx in triple_to_k4s[idx]:
                                    if k4_counts[k4_idx] == 3:
                                        red_viol += 1
                                    k4_counts[k4_idx] += 1
                                for k5_idx in triple_to_k5s[idx]:
                                    if k5_counts[k5_idx] == 10:
                                        blue_viol -= 1
                                    k5_counts[k5_idx] -= 1
                            else:
                                for k4_idx in triple_to_k4s[idx]:
                                    if k4_counts[k4_idx] == 4:
                                        red_viol -= 1
                                    k4_counts[k4_idx] -= 1
                                for k5_idx in triple_to_k5s[idx]:
                                    if k5_counts[k5_idx] == 9:
                                        blue_viol += 1
                                    k5_counts[k5_idx] += 1
                            colors[idx] = new_val_rev
                            red_viol -= delta_red
                            blue_viol -= delta_blue
                            total_viol = red_viol + blue_viol
                            no_improve += 1
                    else:
                        no_improve += 1
                
                T *= cooling
                
                if no_improve > max_no_improve:
                    break

        if best_viol == 0:
            return best_colors.tolist()
        return None

    # Try n=34
    remaining = deadline - time.time()
    if remaining > 10:
        result = solve_for_n(34, remaining - 5)
        if result is not None:
            return {'n': 34, 'colors': result}

    # Try n=33
    remaining = deadline - time.time()
    if remaining > 5:
        result = solve_for_n(33, remaining - 2)
        if result is not None:
            return {'n': 33, 'colors': result}

    # Try n=32
    remaining = deadline - time.time()
    if remaining > 3:
        result = solve_for_n(32, remaining - 1)
        if result is not None:
            return {'n': 32, 'colors': result}
            
    # Fallback: Valid small n
    # n=1 is trivially valid
    return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
