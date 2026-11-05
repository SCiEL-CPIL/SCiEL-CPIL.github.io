"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # 90s limit, keep 5s margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // (i + 1) for i in range(30)]
    
    def compute_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * weights[i]
        return s
    
    def local_search(coeffs, current_sum):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_sum = compute_sum(coeffs)
                    if new_sum == 0:
                        coeffs[i] = old_val
                        continue
                    if abs(new_sum) < abs(current_sum):
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                if improved:
                    break
        return coeffs, current_sum
    
    def greedy_init():
        coeffs = [0] * 30
        current_sum = 0
        indices = sorted(range(30), key=lambda i: weights[i], reverse=True)
        for i in indices:
            best_val = 0
            best_abs = abs(current_sum) if current_sum != 0 else float('inf')
            for val in (-1, 0, 1):
                new_sum = current_sum + val * weights[i]
                if new_sum == 0:
                    continue
                new_abs = abs(new_sum)
                if new_abs < best_abs:
                    best_abs = new_abs
                    best_val = val
            coeffs[i] = best_val
            current_sum += best_val * weights[i]
        if current_sum == 0:
            for i in range(30):
                old = coeffs[i]
                for new in (-1, 0, 1):
                    if new == old:
                        continue
                    coeffs[i] = new
                    new_sum = current_sum - old * weights[i] + new * weights[i]
                    if new_sum != 0:
                        current_sum = new_sum
                        return coeffs, current_sum
                coeffs[i] = old
            coeffs[0] = 1
            current_sum = weights[0]
        return coeffs, current_sum
    
    best_sum = None
    best_coeffs = None
    
    def update_best(coeffs, int_sum):
        nonlocal best_sum, best_coeffs
        if int_sum == 0:
            return
        if best_sum is None or abs(int_sum) < abs(best_sum):
            best_sum = int_sum
            best_coeffs = list(coeffs)
    
    random.seed(42)
    
    coeffs, current_sum = greedy_init()
    update_best(coeffs, current_sum)
    coeffs, current_sum = local_search(coeffs, current_sum)
    update_best(coeffs, current_sum)
    
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        coeffs = [0] * 30
        num_nonzero = random.randint(5, 15)
        for _ in range(num_nonzero):
            idx = random.randint(0, 29)
            coeffs[idx] = random.choice([-1, 1])
        
        current_sum = compute_sum(coeffs)
        if current_sum == 0:
            idx = random.randint(0, 29)
            coeffs[idx] = random.choice([-1, 1])
            current_sum = compute_sum(coeffs)
        
        update_best(coeffs, current_sum)
        
        T_start = max(abs(current_sum), 10000)
        T_end = 1
        num_iterations = 3000
        cooling_rate = (T_end / T_start) ** (1.0 / num_iterations) if T_start > 0 else 1.0
        
        T = T_start
        for it in range(num_iterations):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            if move_type < 0.7:
                i = random.randint(0, 29)
                old_val = coeffs[i]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                coeffs[i] = new_val
                new_sum = compute_sum(coeffs)
                
                if new_sum == 0:
                    coeffs[i] = old_val
                    T *= cooling_rate
                    continue
                
                old_abs = abs(current_sum)
                new_abs = abs(new_sum)
                delta = new_abs - old_abs
                
                if delta < 0 or (T > 0 and random.random() < (2.718281828 ** (-delta / T))):
                    current_sum = new_sum
                    update_best(coeffs, current_sum)
                else:
                    coeffs[i] = old_val
            else:
                i, j = random.sample(range(30), 2)
                old_i, old_j = coeffs[i], coeffs[j]
                new_i = random.choice([-1, 0, 1])
                new_j = random.choice([-1, 0, 1])
                if new_i == old_i and new_j == old_j:
                    continue
                coeffs[i] = new_i
                coeffs[j] = new_j
                new_sum = current_sum - old_i * weights[i] - old_j * weights[j] + new_i * weights[i] + new_j * weights[j]
                
                if new_sum == 0:
                    coeffs[i] = old_i
                    coeffs[j] = old_j
                    T *= cooling_rate
                    continue
                
                old_abs = abs(current_sum)
                new_abs = abs(new_sum)
                delta = new_abs - old_abs
                
                if delta < 0 or (T > 0 and random.random() < (2.718281828 ** (-delta / T))):
                    current_sum = new_sum
                    update_best(coeffs, current_sum)
                else:
                    coeffs[i] = old_i
                    coeffs[j] = old_j
            
            T *= cooling_rate
        
        coeffs, current_sum = local_search(coeffs, current_sum)
        update_best(coeffs, current_sum)
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
