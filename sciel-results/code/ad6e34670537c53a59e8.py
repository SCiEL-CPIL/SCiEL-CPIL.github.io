"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 256
    
    # Optimized discrepancy calculation using NumPy
    def calc_disc(a):
        # a is a numpy array of int8 or int32
        max_abs = 0
        for d in range(1, n + 1):
            for r in range(d):
                # Extract the arithmetic progression
                # Slicing creates a view, cumsum creates a new array
                seq = a[r::d]
                if len(seq) == 0:
                    continue
                cs = np.cumsum(seq)
                m = np.max(np.abs(cs))
                if m > max_abs:
                    max_abs = m
                    # Early exit if we hit a high value? 
                    # Max possible is n. If max_abs > current_best_global, we might stop early?
                    # But we need the exact value for comparison. 
                    # However, if we are just checking if it's better than a threshold, we could.
                    # Here we return exact.
        return max_abs

    # Initialize with Thue-Morse
    signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    current_signs = signs.copy()
    current_disc = calc_disc(current_signs)
    
    best_signs = current_signs.copy()
    best_disc = current_disc
    
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 58.0
    
    # Parameters
    no_improve_count = 0
    max_no_improve = 500 # Restart if stuck for 500 steps
    
    while time.time() - start_time < time_limit:
        # Flip a random bit
        idx = rng.randint(0, n - 1)
        
        # Try flip
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        new_disc = calc_disc(current_signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            no_improve_count = 0
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = current_signs.copy()
        elif new_disc == current_disc:
            # Accept equal to explore plateau
            if rng.random() < 0.5:
                no_improve_count += 1
            else:
                # Revert
                current_signs[idx] = old_val
                no_improve_count += 1
        else:
            # Revert
            current_signs[idx] = old_val
            no_improve_count += 1
        
        # Restart logic
        if no_improve_count > max_no_improve:
            # Restart with a random mutation of the best solution
            # Or a completely new random start?
            # Let's try a "kick" from the best solution
            current_signs = best_signs.copy()
            current_disc = best_disc
            
            # Apply a few random flips to kick it out of the local minimum
            for _ in range(10):
                idx_kick = rng.randint(0, n - 1)
                current_signs[idx_kick] = -current_signs[idx_kick]
            
            current_disc = calc_disc(current_signs)
            no_improve_count = 0
            
            # If the kick made it worse, we might just stay on best and try again?
            # The loop will continue and try to improve from this kicked state.
            # If it can't improve, it will restart again.

    return best_signs.tolist()

# EVOLVE-BLOCK-END
