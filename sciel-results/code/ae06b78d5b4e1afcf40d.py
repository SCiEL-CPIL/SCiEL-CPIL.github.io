import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    weights = [L // i for i in range(1, n + 1)]
    
    def evaluate(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * weights[i]
        return s
    
    best_sum = None
    best_coeffs = None
    
    T_start = 1e12
    T_end = 1e3
    alpha = 0.99995
    
    def make_random_valid():
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        s = evaluate(coeffs)
        while s == 0:
            idx = random.randint(0, n-1)
            coeffs[idx] = random.choice([-1, 0, 1])
            s = evaluate(coeffs)
        return coeffs, s
    
    while time.time() < deadline:
        coeffs, current_sum = make_random_valid()
        
        if best_sum is None or abs(current_sum) < abs(best_sum):
            best_sum = current_sum
            best_coeffs = coeffs[:]
        
        T = T_start
        while T > T_end and time.time() < deadline:
            if random.random() < 0.85:
                idx = random.randint(0, n-1)
                old_val = coeffs[idx]
                new_val = random.choice([-1, 0, 1])
                while new_val == old_val:
                    new_val = random.choice([-1, 0, 1])
                delta = (new_val - old_val) * weights[idx]
                new_sum = current_sum + delta
                idx2 = None
            else:
                idx = random.randint(0, n-1)
                idx2 = random.randint(0, n-1)
                if idx2 == idx:
                    T *= alpha
                    continue
                old_val1 = coeffs[idx]
                old_val2 = coeffs[idx2]
                new_val1 = random.choice([-1, 0, 1])
                new_val2 = random.choice([-1, 0, 1])
                while new_val1 == old_val1:
                    new_val1 = random.choice([-1, 0, 1])
                while new_val2 == old_val2:
                    new_val2 = random.choice([-1, 0, 1])
                delta = (new_val1 - old_val1) * weights[idx] + (new_val2 - old_val2) * weights[idx2]
                new_sum = current_sum + delta
            
            if new_sum == 0:
                T *= alpha
                continue
            
            old_obj = abs(current_sum)
            new_obj = abs(new_sum)
            
            if new_obj < old_obj:
                if idx2 is None:
                    coeffs[idx] = new_val
                else:
                    coeffs[idx] = new_val1
                    coeffs[idx2] = new_val2
                current_sum = new_sum
                if best_sum is None or abs(current_sum) < abs(best_sum):
                    best_sum = current_sum
                    best_coeffs = coeffs[:]
            else:
                delta_obj = new_obj - old_obj
                if delta_obj < T:
                    if random.random() < math.exp(-delta_obj / T):
                        if idx2 is None:
                            coeffs[idx] = new_val
                        else:
                            coeffs[idx] = new_val1
                            coeffs[idx2] = new_val2
                        current_sum = new_sum
            
            T *= alpha
    
    if best_coeffs is not None:
        improved = True
        max_iters = 200
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < deadline:
            iter_count += 1
            improved = False
            for i in range(n):
                old_val = best_coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = best_sum + delta
                    if new_sum != 0 and abs(new_sum) < abs(best_sum):
                        best_coeffs[i] = new_val
                        best_sum = new_sum
                        improved = True
                        break
    
    if best_coeffs is None:
        fallback = [0] * 30
        fallback[1] = 1
        fallback[2] = -1
        return fallback
    
    return best_coeffs
