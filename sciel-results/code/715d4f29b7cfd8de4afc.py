import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    
    N = 30
    # Precompute reciprocals as Fractions
    recips = [Fraction(1, i) for i in range(1, N + 1)]
    
    best_val = Fraction(1, 1)  # Initial best absolute sum
    best_coeffs = [0] * N
    
    def local_search(c):
        # Compute initial sum
        s = Fraction(0, 1)
        for i in range(N):
            s += c[i] * recips[i]
            
        improved = True
        # Hill climbing
        while improved:
            improved = False
            for i in range(N):
                old_val = c[i]
                # Try changing c[i] to -1, 0, 1
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    # Calculate new sum
                    # delta = (new_val - old_val) / (i+1)
                    delta = Fraction(new_val - old_val, i + 1)
                    new_s = s + delta
                    
                    # Check if improvement (smaller absolute value)
                    # And ensure it's not zero? 
                    # If new_s is 0, it's invalid for the final answer, 
                    # but might be a good intermediate step? 
                    # Actually, if we hit 0, we can't use it. 
                    # But moving away from 0 is also an improvement in terms of "getting close to 0 but not 0"? 
                    # No, the objective is to minimize |S| subject to S != 0.
                    # If current S is 1e-5 and new S is 0, |0| < |1e-5|, but 0 is invalid.
                    # So we should NOT accept 0.
                    if new_s == 0:
                        continue
                    
                    if abs(new_s) < abs(s):
                        s = new_s
                        c[i] = new_val
                        improved = True
                        break
                if improved:
                    break
        return s, c

    # Main search loop
    while time.time() < deadline:
        # Random initialization
        c = [random.choice([-1, 0, 1]) for _ in range(N)]
        s, c = local_search(c)
        
        # Update best if valid and smaller
        if s != 0 and abs(s) < best_val:
            best_val = abs(s)
            best_coeffs = c[:]
            
    return best_coeffs
