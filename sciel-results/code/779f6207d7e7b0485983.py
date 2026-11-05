import numpy as np
import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s kill
    
    def eval_discrepancy_fast(signs):
        """
        Compute max absolute sum over all APs.
        Uses vectorized cumsum for each step d.
        """
        max_disc = 0
        for d in range(1, n):
            # Max length of any AP with step d
            max_len = (n - 1) // d + 1
            
            # Allocate matrix for this step d
            # Shape: (d, max_len)
            # Note: For large d, max_len is small. For small d, max_len is large.
            # Total size is roughly n.
            M = np.zeros((d, max_len), dtype=np.int8)
            
            for r in range(d):
                L = (n - 1 - r) // d + 1
                if L > 0:
                    # Fill row r with the subsequence starting at r with step d
                    M[r, :L] = signs[r::d]
            
            # Compute prefix sums along the AP (axis 1)
            cum = np.cumsum(M, axis=1)
            
            # Find max absolute value
            local_max = int(np.max(np.abs(cum)))
            
            if local_max > max_disc:
                max_disc = local_max
                # Early exit if we hit the theoretical minimum
                if max_disc == 1:
                    return 1
                    
        return max_disc

    def thue_morse(n):
        """Generate Thue-Morse sequence."""
        s = np.zeros(n, dtype=np.int8)
        for i in range(n):
            s[i] = 1 if bin(i).count('1') % 2 == 0 else -1
        return s

    # --- Phase 1: Initialization ---
    candidates = []
    
    # Thue-Morse
    candidates.append(thue_morse(n))
    
    # Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    candidates.append(alt)
    
    # Random candidates
    rng_init = random.Random(42)
    for _ in range(20):
        candidates.append(np.array([rng_init.choice((-1, 1)) for _ in range(n)], dtype=np.int8))
        
    # Evaluate candidates and pick the best starting point
    best_start = candidates[0]
    best_start_disc = float('inf')
    for c in candidates:
        d = eval_discrepancy_fast(c)
        if d < best_start_disc:
            best_start_disc = d
            best_start = c
            
    signs = best_start.copy()
    current_disc = best_start_disc
    best_disc = current_disc
    best_signs = signs.copy()
    
    # --- Phase 2: Simulated Annealing ---
    rng = random.Random(12345)
    temp = 2.0
    cool = 0.999
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 10:
        # Pick a random index to flip
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        signs[idx] = -old_val
        
        new_disc = eval_discrepancy_fast(signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            no_improve = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            delta = new_disc - current_disc
            # Accept worse move with probability exp(-delta / temp)
            if delta > 0 and temp > 0.01:
                if rng.random() < math.exp(-delta / temp):
                    current_disc = new_disc
                    no_improve = 1 # Count as activity
                else:
                    signs[idx] = old_val
                    no_improve += 1
            else:
                signs[idx] = old_val
                no_improve += 1
                
        temp *= cool
        
        # Restart logic: If stuck for too long, reset to best and increase temp
        if no_improve > max_no_improve:
            signs = best_signs.copy()
            current_disc = best_disc
            temp = 2.0
            no_improve = 0

    # --- Phase 3: Hill Climbing (Polish) ---
    if time.time() < deadline - 2:
        signs = best_signs.copy()
        current_disc = eval_discrepancy_fast(signs)
        improved = True
        # Limit iterations to avoid infinite loop if eval is very slow (it shouldn't be)
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n):
                old_val = signs[idx]
                signs[idx] = -old_val
                new_disc = eval_discrepancy_fast(signs)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs.copy()
                else:
                    signs[idx] = old_val

    return list(best_signs)
