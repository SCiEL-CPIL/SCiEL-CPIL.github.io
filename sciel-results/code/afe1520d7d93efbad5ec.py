import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe
    
    n = 30
    # Pre-compute weights as Fractions
    weights = [Fraction(1, i) for i in range(1, n + 1)]
    
    best_abs_val = None
    best_coeffs = None
    
    # Helper to check if a move is valid and better
    def try_improve(coeffs, current_sum):
        nonlocal best_abs_val, best_coeffs
        improved = False
        # Try to improve current solution
        # We will do a random walk with backtracking or just restart if stuck
        # Let's do a simple local search loop
        
        # To avoid getting stuck in a local minimum immediately, 
        # we can do a few random restarts.
        
        return coeffs, current_sum, improved

    # Main search loop
    # We will run multiple restarts.
    # Each restart:
    # 1. Random init
    # 2. Local search until no improvement for X steps or time limit
    
    while time.time() < deadline:
        # Random initialization
        # Bias towards balanced +/-? 
        # Just uniform random in {-1, 0, 1}
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Calculate initial sum
        current_sum = sum(c * w for c, w in zip(coeffs, weights))
        
        # If initial sum is 0, it's invalid. But we can start from there and move away?
        # The problem requires the final sum to be non-zero.
        # If we start at 0, we must move to a non-zero neighbor.
        # Let's just treat 0 as "infinitely bad" for the purpose of improvement, 
        # but we must ensure the final result is non-zero.
        
        if current_sum == 0:
            # Find a non-zero neighbor
            found = False
            for i in range(n):
                old_c = coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c: continue
                    delta = (new_c - old_c) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum != 0:
                        coeffs[i] = new_c
                        current_sum = new_sum
                        found = True
                        break
                if found: break
            if not found:
                # This shouldn't happen unless n=0
                continue
        
        # If this random start is already better than best, update
        abs_val = abs(current_sum)
        if best_abs_val is None or abs_val < best_abs_val:
            best_abs_val = abs_val
            best_coeffs = coeffs[:]
        
        # Local Search
        steps_since_improvement = 0
        max_stagnation = 5000 # If no improvement for 5000 steps, restart
        
        while time.time() < deadline:
            if steps_since_improvement > max_stagnation:
                break
            
            # Pick a random index
            i = random.randint(0, n - 1)
            old_c = coeffs[i]
            
            # Try the other two values
            # Order might matter? Randomize?
            candidates = [c for c in [-1, 0, 1] if c != old_c]
            random.shuffle(candidates)
            
            improved_in_this_step = False
            for new_c in candidates:
                delta = (new_c - old_c) * weights[i]
                new_sum = current_sum + delta
                
                if new_sum == 0:
                    continue # Invalid state
                
                new_abs_val = abs(new_sum)
                
                # Check if this is a better solution overall
                if best_abs_val is None or new_abs_val < best_abs_val:
                    best_abs_val = new_abs_val
                    best_coeffs = coeffs[:]
                    # Note: coeffs[i] is not updated yet in best_coeffs copy
                    best_coeffs[i] = new_c
                
                # Check if this is a better solution than current local state
                if new_abs_val < abs(current_sum):
                    coeffs[i] = new_c
                    current_sum = new_sum
                    improved_in_this_step = True
                    steps_since_improvement = 0
                    break # Accept this move and restart step selection
            
            if not improved_in_this_step:
                steps_since_improvement += 1
        
        # If time is up, break outer loop
    
    # Fallback if no valid solution found (should not happen)
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
