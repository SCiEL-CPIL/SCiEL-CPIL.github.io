import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
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

    def run_local_search(initial_coeffs):
        coeffs = initial_coeffs[:]
        current_sum = evaluate(coeffs)
        
        if current_sum == 0:
            best_neighbor_sum = None
            best_neighbor_coeffs = None
            for j in range(n):
                old_val = coeffs[j]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    new_sum = (new_val - old_val) * weights[j]
                    if new_sum != 0:
                        if best_neighbor_sum is None or abs(new_sum) < abs(best_neighbor_sum):
                            best_neighbor_sum = new_sum
                            best_neighbor_coeffs = coeffs[:]
                            best_neighbor_coeffs[j] = new_val
            if best_neighbor_coeffs:
                return best_neighbor_coeffs
            return coeffs

        improved = True
        while improved:
            improved = False
            for i in range(n):
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        best_neighbor_sum = None
                        best_neighbor_coeffs = None
                        for j in range(n):
                            old_v = coeffs[j]
                            for nv in [-1, 0, 1]:
                                if nv == old_v:
                                    continue
                                ns = (nv - old_v) * weights[j]
                                if ns != 0:
                                    if best_neighbor_sum is None or abs(ns) < abs(best_neighbor_sum):
                                        best_neighbor_sum = ns
                                        best_neighbor_coeffs = coeffs[:]
                                        best_neighbor_coeffs[j] = nv
                        if best_neighbor_coeffs:
                            return best_neighbor_coeffs
                        else:
                            return coeffs
                            
                    else:
                        if abs(new_sum) < abs(current_sum):
                            coeffs[i] = new_val
                            current_sum = new_sum
                            improved = True
                            
        return coeffs

    best_sum = None
    best_coeffs = None
    
    # Initial deterministic starts
    starts = []
    # All zeros
    starts.append([0]*n)
    # All ones
    starts.append([1]*n)
    # All -1
    starts.append([-1]*n)
    # Seed
    starts.append([0]*28 + [1, -1])
    
    for s in starts:
        if time.time() > deadline: break
        improved_coeffs = run_local_search(s)
        s_val = evaluate(improved_coeffs)
        if s_val != 0:
            if best_sum is None or abs(s_val) < abs(best_sum):
                best_sum = s_val
                best_coeffs = improved_coeffs

    # Random starts
    while time.time() < deadline:
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        improved_coeffs = run_local_search(coeffs)
        s_val = evaluate(improved_coeffs)
        if s_val != 0:
            if best_sum is None or abs(s_val) < abs(best_sum):
                best_sum = s_val
                best_coeffs = improved_coeffs

    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
