import random
from fractions import Fraction
import time

def construct():
    deadline = time.time() + 85
    
    recips = [Fraction(1, i) for i in range(1, 31)]
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * recips[i]
        return s
    
    best_coeffs = [0] * 28 + [1, -1]
    best_sum = compute_sum(best_coeffs)
    best_abs = abs(best_sum)
    
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        
        # Single-coordinate moves
        for idx in range(30):
            if improved:
                break
            old_val = best_coeffs[idx]
            for new_val in (-1, 0, 1):
                if new_val == old_val:
                    continue
                delta = (new_val - old_val) * recips[idx]
                new_sum = best_sum + delta
                if new_sum != 0:
                    new_abs = abs(new_sum)
                    if new_abs < best_abs:
                        best_coeffs[idx] = new_val
                        best_sum = new_sum
                        best_abs = new_abs
                        improved = True
                        break
        
        # Two-coordinate moves
        if not improved:
            for idx1 in range(30):
                if improved:
                    break
                old1 = best_coeffs[idx1]
                for new1 in (-1, 0, 1):
                    if new1 == old1 or improved:
                        continue
                    delta1 = (new1 - old1) * recips[idx1]
                    for idx2 in range(idx1 + 1, 30):
                        if improved:
                            break
                        old2 = best_coeffs[idx2]
                        for new2 in (-1, 0, 1):
                            if new2 == old2 or improved:
                                continue
                            delta2 = (new2 - old2) * recips[idx2]
                            new_sum = best_sum + delta1 + delta2
                            if new_sum != 0:
                                new_abs = abs(new_sum)
                                if new_abs < best_abs:
                                    best_coeffs[idx1] = new1
                                    best_coeffs[idx2] = new2
                                    best_sum = new_sum
                                    best_abs = new_abs
                                    improved = True
                                    break
        
        if not improved:
            # Random restart
            best_coeffs = [random.choice((-1, 0, 1)) for _ in range(30)]
            best_sum = compute_sum(best_coeffs)
            if best_sum == 0:
                idx = random.randint(0, 29)
                best_coeffs[idx] = 1
                best_sum += recips[idx]
            if abs(best_sum) < best_abs:
                best_abs = abs(best_sum)
    
    return best_coeffs
