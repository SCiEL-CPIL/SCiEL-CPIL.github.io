"""Program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import math

def construct():
    n = 256
    # Initialize with random signs
    signs = [random.choice([-1, 1]) for _ in range(n)]
    
    def evaluate(s):
        """
        Computes the maximum absolute signed sum on ANY arithmetic progression.
        Returns the max discrepancy.
        """
        max_disc = 0
        # Iterate over all steps d
        for d in range(1, n + 1):
            # For each step d, there are d sequences based on start index mod d
            # Sequence j contains indices j, j+d, j+2d, ... (0-indexed)
            for j in range(d):
                # Extract the sequence
                # We can compute max abs subarray sum directly
                # Kadane's algorithm for max sum and min sum
                max_sum = 0
                curr_max = 0
                min_sum = 0
                curr_min = 0
                
                # Iterate through the sequence
                idx = j
                while idx < n:
                    val = s[idx]
                    # Update max
                    curr_max = max(0, curr_max + val)
                    if curr_max > max_sum:
                        max_sum = curr_max
                    # Update min
                    curr_min = min(0, curr_min + val)
                    if curr_min < min_sum:
                        min_sum = curr_min
                    idx += d
                
                local_max_abs = max(max_sum, -min_sum)
                if local_max_abs > max_disc:
                    max_disc = local_max_abs
                    
        return max_disc

    # Simulated Annealing
    current_signs = signs[:]
    current_D = evaluate(current_signs)
    
    best_signs = current_signs[:]
    best_D = current_D
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s buffer
    
    # Temperature schedule
    T = 5.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < timeout:
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        
        # Flip
        current_signs[i] *= -1
        
        # Evaluate new discrepancy
        new_D = evaluate(current_signs)
        
        # Acceptance criterion
        if new_D < current_D:
            current_D = new_D
            if new_D < best_D:
                best_D = new_D
                best_signs = current_signs[:]
        else:
            # Probabilistic acceptance
            delta = new_D - current_D
            if delta == 0:
                # Accept to allow wandering
                if random.random() < 0.5:
                    current_D = new_D
                else:
                    current_signs[i] *= -1 # Revert
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_D = new_D
                else:
                    current_signs[i] *= -1 # Revert
        
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_signs
# EVOLVE-BLOCK-END
