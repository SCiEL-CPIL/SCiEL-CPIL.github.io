import random
from fractions import Fraction
import time

def construct():
    n = 30
    start_time = time.time()
    timeout = 55  # seconds
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        return s
    
    # Multiple random restarts with local search
    best_coeffs = None
    best_abs = None
    
    for restart in range(200):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        while all(c == 0 for c in coeffs):
            coeffs[random.randrange(n)] = random.choice([-1, 1])
        
        current_sum = compute_sum(coeffs)
        if current_sum == 0:
            # Perturb to get nonzero
            idx = random.randrange(n)
            coeffs[idx] = random.choice([-1, 1])
            current_sum = compute_sum(coeffs)
        
        current_abs = abs(current_sum)
        
        if best_abs is None or current_abs < best_abs:
            best_abs = current_abs
            best_coeffs = coeffs[:]
        
        # Local search: try single flips
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            for idx in range(n):
                if time.time() - start_time > timeout:
                    break
                old_val = coeffs[idx]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = Fraction(new_val - old_val, idx + 1)
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < current_abs:
                        coeffs[idx] = new_val
                        current_sum = new_sum
                        current_abs = new_abs
                        improved = True
                        if best_abs is None or current_abs < best_abs:
                            best_abs = current_abs
                            best_coeffs = coeffs[:]
                        break  # Try next index
    
    if best_coeffs is None or all(c == 0 for c in best_coeffs):
        # Fallback: ensure nonzero
        return [0] * 28 + [1, -1]
    
    # Final verification
    final_sum = compute_sum(best_coeffs)
    if final_sum == 0:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
