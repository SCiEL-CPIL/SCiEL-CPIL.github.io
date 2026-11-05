import time
import random
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_val = Fraction(1, 870)  # Seed value
    best_coeffs = [0]*28 + [1, -1]
    
    n = 30
    denoms = [Fraction(1, i) for i in range(1, n+1)]
    
    def calc_sum(coeffs):
        s = Fraction(0, 1)
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * denoms[i]
        return s
        
    def local_search(coeffs, max_iters=200):
        current_sum = calc_sum(coeffs)
        current_abs = abs(current_sum)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try single bit flips
            for i in range(n):
                for new_val in [-1, 0, 1]:
                    if new_val == coeffs[i]: continue
                    old_val = coeffs[i]
                    coeffs[i] = new_val
                    new_sum = current_sum + (new_val - old_val) * denoms[i]
                    new_abs = abs(new_sum)
                    
                    if new_abs < current_abs:
                        current_sum = new_sum
                        current_abs = new_abs
                        improved = True
                    else:
                        coeffs[i] = old_val
                        
        return coeffs, current_sum, current_abs

    def sa_search(initial_coeffs, time_limit=1.0):
        current_coeffs = initial_coeffs[:]
        current_sum = calc_sum(current_coeffs)
        current_abs = abs(current_sum)
        
        best_coeffs = current_coeffs[:]
        best_sum = current_sum
        best_abs = current_abs
        
        start_t = time.time()
        T = 0.05
        cooling = 0.9995
        
        while time.time() - start_t < time_limit and T > 1e-9:
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n-1)
            old_val = current_coeffs[idx]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
                
            current_coeffs[idx] = new_val
            current_sum = current_sum + (new_val - old_val) * denoms[idx]
            new_abs = abs(current_sum)
            
            delta = new_abs - current_abs
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_abs = new_abs
                if new_abs < best_abs:
                    best_abs = new_abs
                    best_sum = current_sum
                    best_coeffs = current_coeffs[:]
            else:
                current_coeffs[idx] = old_val
                current_sum = current_sum - (new_val - old_val) * denoms[idx]
            
            T *= cooling
            
        return best_coeffs, best_sum, best_abs

    try:
        restarts = 0
        while time.time() < deadline:
            restarts += 1
            
            if restarts == 1:
                coeffs = best_coeffs[:]
            elif restarts % 3 == 0:
                coeffs = [0]*n
                num_nz = random.randint(2, 5)
                indices = random.sample(range(n), num_nz)
                for i in indices:
                    coeffs[i] = random.choice([-1, 1])
            else:
                coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            
            coeffs, s, abs_s = local_search(coeffs)
            
            if abs_s < best_val and s != 0:
                best_val = abs_s
                best_coeffs = coeffs[:]
            
            if time.time() < deadline - 1.0:
                sa_coeffs, sa_s, sa_abs = sa_search(coeffs, time_limit=1.0)
                if sa_abs < best_val and sa_s != 0:
                    best_val = sa_abs
                    best_coeffs = sa_coeffs[:]
                
                if s == 0:
                    perturbed = coeffs[:]
                    idx = random.randint(0, n-1)
                    perturbed[idx] = random.choice([-1, 1]) if perturbed[idx] == 0 else 0
                    p_coeffs, p_s, p_abs = local_search(perturbed)
                    if p_abs < best_val and p_s != 0:
                        best_val = p_abs
                        best_coeffs = p_coeffs[:]

    except Exception:
        pass

    return best_coeffs
