import time
import random
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    n = 30
    # Coefficients c[i] correspond to term c[i] / (i+1)
    
    # Initialize with the known fallback from the seed program
    # This guarantees a valid non-zero sum: 1/29 - 1/30 = 1/870
    best_coeffs = [0] * n
    best_coeffs[28] = 1
    best_coeffs[29] = -1
    
    # Calculate initial sum
    best_sum = Fraction(0, 1)
    for i in range(n):
        if best_coeffs[i] != 0:
            best_sum += Fraction(best_coeffs[i], i + 1)
            
    best_abs_val = abs(best_sum)
    
    # Precompute denominators as Fractions for speed? 
    # Or just use Fraction(1, i+1)
    invs = [Fraction(1, i + 1) for i in range(n)]
    
    def get_sum(coeffs):
        s = Fraction(0, 1)
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * invs[i]
        return s

    # Local Search / Hill Climbing with Random Restarts
    # We want to minimize abs(sum) subject to sum != 0
    
    current_coeffs = best_coeffs[:]
    current_sum = best_sum
    
    while time.time() - start_time < deadline:
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        
        # Current value is current_coeffs[i]
        # New value can be any of {-1, 0, 1} except current_coeffs[i]
        old_val = current_coeffs[i]
        possible_vals = [v for v in (-1, 0, 1) if v != old_val]
        
        # Shuffle to ensure random choice among valid moves
        random.shuffle(possible_vals)
        
        moved = False
        for new_val in possible_vals:
            # Calculate new sum efficiently
            # new_sum = current_sum - old_val * invs[i] + new_val * invs[i]
            new_sum = current_sum + (new_val - old_val) * invs[i]
            
            # Constraint: new_sum must be non-zero
            if new_sum == 0:
                continue
                
            new_abs = abs(new_sum)
            
            # Accept if better
            if new_abs < best_abs_val:
                best_abs_val = new_abs
                best_coeffs = current_coeffs[:]
                best_sum = new_sum
                
            # Accept if better than current (Hill Climbing)
            if new_abs < abs(current_sum) or (abs(current_sum) == 0 and new_abs > 0):
                # Update current state
                current_coeffs[i] = new_val
                current_sum = new_sum
                moved = True
                break # Move to next iteration of outer loop
                
        if not moved:
            # Stuck in local minimum.
            # Perform a random restart or a random jump to escape.
            # Let's try a few random single flips even if they are worse, to escape.
            escape_success = False
            for _ in range(10):
                i = random.randint(0, n - 1)
                old_val = current_coeffs[i]
                possible_vals = [v for v in (-1, 0, 1) if v != old_val]
                new_val = random.choice(possible_vals)
                
                new_sum = current_sum + (new_val - old_val) * invs[i]
                
                if new_sum == 0:
                    continue
                    
                # Accept any move that doesn't make things too bad, or just accept a random bad move to escape
                # Simpler: Just accept a random valid move to escape local min
                current_coeffs[i] = new_val
                current_sum = new_sum
                escape_success = True
                break
                
            if not escape_success:
                # If we couldn't even make a random move (unlikely), restart from best
                current_coeffs = best_coeffs[:]
                current_sum = best_sum
                # And maybe do a random perturbation
                for _ in range(3):
                    i = random.randint(0, n - 1)
                    old_val = current_coeffs[i]
                    possible_vals = [v for v in (-1, 0, 1) if v != old_val]
                    new_val = random.choice(possible_vals)
                    new_sum = current_sum + (new_val - old_val) * invs[i]
                    if new_sum != 0:
                        current_coeffs[i] = new_val
                        current_sum = new_sum
                        break

    # Final check: ensure best_coeffs gives a non-zero sum
    # It should, as we only update best if new_sum != 0
    return best_coeffs
