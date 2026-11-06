import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    n = 30
    # denominators are 1, 2, ..., 30
    # sum = c[0]/1 + c[1]/2 + ... + c[29]/30
    
    best_val = None  # Fraction
    best_coeffs = None
    
    def eval_sum(coeffs):
        s = Fraction(0, 1)
        for i in range(n):
            if coeffs[i] != 0:
                s += Fraction(coeffs[i], i + 1)
        return s
    
    # Initial random solutions
    for _ in range(200):
        if time.time() > deadline:
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        val = eval_sum(coeffs)
        if val != 0:
            if best_val is None or abs(val) < abs(best_val):
                best_val = val
                best_coeffs = coeffs[:]
    
    # Local search: try flipping individual coefficients
    if best_coeffs is not None:
        current = best_coeffs[:]
        current_val = best_val
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                old = current[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    current[i] = new_val
                    new_sum = eval_sum(current)
                    if new_sum != 0:
                        if abs(new_sum) < abs(current_val):
                            current_val = new_sum
                            improved = True
                            break
                    else:
                        # zero sum is invalid, but we can use it as intermediate
                        pass
                    current[i] = old
                # restore
                current[i] = old
        
        if abs(current_val) < abs(best_val):
            best_val = current_val
            best_coeffs = current[:]
    
    # More extensive randomized restarts with local search
    for _ in range(5000):
        if time.time() > deadline:
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        val = eval_sum(coeffs)
        if val != 0:
            # Local search from this point
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                old = coeffs[i]
                new_val = random.choice([-1, 0, 1])
                if new_val == old:
                    continue
                coeffs[i] = new_val
                new_sum = eval_sum(coeffs)
                if new_sum != 0:
                    if abs(new_sum) < abs(val):
                        val = new_sum
                    else:
                        coeffs[i] = old
                else:
                    # zero is invalid; revert or keep? Revert to avoid getting stuck
                    coeffs[i] = old
            if abs(val) < abs(best_val):
                best_val = val
                best_coeffs = coeffs[:]
    
    if best_coeffs is None:
        # Fallback: should not happen, but just in case
        return [0] * 28 + [1, -1]
    
    return best_coeffs
