import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def build_data(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        k4_data = []
        k5_data = []
        
        for i, k4 in enumerate(k4_list):
            ts = []
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                idx = triple_to_idx[t]
                ts.append(idx)
                triple_to_k4s[idx].append(i)
            k4_data.append(ts)
            
        for i, k5 in enumerate(k5_list):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                ts.append(idx)
                triple_to_k5s[idx].append(i)
            k5_data.append(ts)
            
        return num_triples, triple_to_k4s, triple_to_k5s, k4_data, k5_data

    def solve_n(n, time_limit):
        # Build data
        num_triples, t_to_k4, t_to_k5, k4_data, k5_data = build_data(n)
        
        # Initialize colors
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        # k4_counts[i] = number of red triples in K4 i
        # k5_counts[i] = number of blue triples in K5 i
        num_k4 = len(k4_data)
        num_k5 = len(k5_data)
        k4_counts = [0] * num_k4
        k5_counts = [0] * num_k5
        
        # Fill counts
        for i in range(num_triples):
            c = colors[i]
            for k4_idx in t_to_k4[i]:
                if c == 0: # Red
                    k4_counts[k4_idx] += 1
            for k5_idx in t_to_k5[i]:
                if c == 1: # Blue
                    k5_counts[k5_idx] += 1
                    
        # Count violations
        violations = sum(1 for c in k4_counts if c == 4) + sum(1 for c in k5_counts if c == 10)
        
        best_violations = violations
        best_colors = colors[:]
        
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # Precompute random access lists for violated clauses?
        # It's hard to maintain dynamically in a fast way in pure Python.
        # We will pick random clauses and check if they are violated.
        
        end_time = time.time() + time_limit
        steps = 0
        
        while time.time() < end_time:
            if violations == 0:
                return colors[:]
            
            # Pick a move
            # 50% chance: pick from a violated clause
            # 50% chance: random triple
            
            if random.random() < 0.5:
                # Pick a random K4 or K5
                if random.random() < 0.5:
                    # Pick random K4
                    k4_idx = random.randint(0, num_k4 - 1)
                    if k4_counts[k4_idx] == 4:
                        # Violated
                        t_idx = random.choice(k4_data[k4_idx])
                    else:
                        # Not violated, fallback to random triple
                        t_idx = random.randint(0, num_triples - 1)
                else:
                    # Pick random K5
                    k5_idx = random.randint(0, num_k5 - 1)
                    if k5_counts[k5_idx] == 10:
                        # Violated
                        t_idx = random.choice(k5_data[k5_idx])
                    else:
                        # Not violated, fallback to random triple
                        t_idx = random.randint(0, num_triples - 1)
            else:
                t_idx = random.randint(0, num_triples - 1)
                
            # Flip t_idx
            old_val = colors[t_idx]
            new_val = 1 - old_val
            colors[t_idx] = new_val
            
            delta_red = 1 if new_val == 0 else -1
            delta_blue = 1 if new_val == 1 else -1
            
            new_violations = violations
            
            for k4_idx in t_to_k4[t_idx]:
                old_count = k4_counts[k4_idx]
                new_count = old_count + delta_red
                k4_counts[k4_idx] = new_count
                
                was_viol = 1 if old_count == 4 else 0
                is_viol = 1 if new_count == 4 else 0
                new_violations += (is_viol - was_viol)
                
            for k5_idx in t_to_k5[t_idx]:
                old_count = k5_counts[k5_idx]
                new_count = old_count + delta_blue
                k5_counts[k5_idx] = new_count
                
                was_viol = 1 if old_count == 10 else 0
                is_viol = 1 if new_count == 10 else 0
                new_violations += (is_viol - was_viol)
                
            # Acceptance
            delta_E = new_violations - violations
            if delta_E < 0:
                violations = new_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
            else:
                # Accept with prob exp(-delta_E / T)
                if random.random() < np.exp(-delta_E / T):
                    violations = new_violations
                else:
                    # Reject: revert
                    colors[t_idx] = old_val
                    for k4_idx in t_to_k4[t_idx]:
                        k4_counts[k4_idx] -= delta_red
                    for k5_idx in t_to_k5[t_idx]:
                        k5_counts[k5_idx] -= delta_blue
            
            # Cool down
            T = max(min_T, T * cooling_rate)
            
            steps += 1
            # Optional: restart if stuck too long? 
            # SA should handle it.
            
        return best_colors

    # Try n=35 first
    if time.time() < deadline - 60:
        res_35 = solve_n(35, 50)
        # Verify res_35
        # We need to check if it's actually valid.
        # The SA returns best_colors, which might not be 0 violations.
        # We need to check the violation count of best_colors.
        # But solve_n doesn't return the count.
        # Let's modify solve_n to return (colors, violations)
        
    # Let's refine solve_n to return the best violations found.
    
    def solve_n_with_check(n, time_limit):
        num_triples, t_to_k4, t_to_k5, k4_data, k5_data = build_data(n)
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        num_k4 = len(k4_data)
        num_k5 = len(k5_data)
        k4_counts = [0] * num_k4
        k5_counts = [0] * num_k5
        
        for i in range(num_triples):
            c = colors[i]
            for k4_idx in t_to_k4[i]:
                if c == 0: k4_counts[k4_idx] += 1
            for k5_idx in t_to_k5[i]:
                if c == 1: k5_counts[k5_idx] += 1
                
        violations = sum(1 for c in k4_counts if c == 4) + sum(1 for c in k5_counts if c == 10)
        best_violations = violations
        best_colors = colors[:]
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if violations == 0:
                return colors[:], 0
            
            if random.random() < 0.5:
                if random.random() < 0.5:
                    k4_idx = random.randint(0, num_k4 - 1)
                    if k4_counts[k4_idx] == 4:
                        t_idx = random.choice(k4_data[k4_idx])
                    else:
                        t_idx = random.randint(0, num_triples - 1)
                else:
                    k5_idx = random.randint(0, num_k5 - 1)
                    if k5_counts[k5_idx] == 10:
                        t_idx = random.choice(k5_data[k5_idx])
                    else:
                        t_idx = random.randint(0, num_triples - 1)
            else:
                t_idx = random.randint(0, num_triples - 1)
                
            old_val = colors[t_idx]
            new_val = 1 - old_val
            colors[t_idx] = new_val
            
            delta_red = 1 if new_val == 0 else -1
            delta_blue = 1 if new_val == 1 else -1
            
            new_violations = violations
            
            for k4_idx in t_to_k4[t_idx]:
                old_count = k4_counts[k4_idx]
                new_count = old_count + delta_red
                k4_counts[k4_idx] = new_count
                was_viol = 1 if old_count == 4 else 0
                is_viol = 1 if new_count == 4 else 0
                new_violations += (is_viol - was_viol)
                
            for k5_idx in t_to_k5[t_idx]:
                old_count = k5_counts[k5_idx]
                new_count = old_count + delta_blue
                k5_counts[k5_idx] = new_count
                was_viol = 1 if old_count == 10 else 0
                is_viol = 1 if new_count == 10 else 0
                new_violations += (is_viol - was_viol)
                
            delta_E = new_violations - violations
            if delta_E < 0:
                violations = new_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
            else:
                if random.random() < np.exp(-delta_E / T):
                    violations = new_violations
                else:
                    colors[t_idx] = old_val
                    for k4_idx in t_to_k4[t_idx]:
                        k4_counts[k4_idx] -= delta_red
                    for k5_idx in t_to_k5[t_idx]:
                        k5_counts[k5_idx] -= delta_blue
            
            T = max(min_T, T * cooling_rate)
            
        return best_colors, best_violations

    # Try n=35
    if time.time() < deadline - 50:
        colors_35, viol_35 = solve_n_with_check(35, 40)
        if viol_35 == 0:
            return {'n': 35, 'colors': colors_35}
            
    # Try n=34
    if time.time() < deadline - 10:
        colors_34, viol_34 = solve_n_with_check(34, 30)
        if viol_34 == 0:
            return {'n': 34, 'colors': colors_34}
            
    # Fallback: return a small valid n
    # n=9 is safe
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}

# EVOLVE-BLOCK-END
