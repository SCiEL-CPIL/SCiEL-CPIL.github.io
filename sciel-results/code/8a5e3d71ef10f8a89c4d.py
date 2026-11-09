"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_progressions(n=256):
    """Precompute all arithmetic progressions as index arrays."""
    progressions = []
    for step in range(1, n + 1):
        for start in range(0, n, step):
            indices = list(range(start, n, step))
            if len(indices) >= 1:
                progressions.append(np.array(indices, dtype=np.int32))
    return progressions

def _max_discrepancy(signs, progressions):
    """Compute max |sum| over all arithmetic progressions."""
    max_disc = 0
    for idx in progressions:
        s = int(np.sum(signs[idx]))
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc

def _incremental_update(signs, progressions, max_disc, pos, old_sign, new_sign):
    """Update max discrepancy after flipping position pos from old_sign to new_sign."""
    delta = new_sign - old_sign
    if delta == 0:
        return max_disc
    new_max = max_disc
    for idx in progressions:
        if pos in idx:
            s = int(np.sum(signs[idx]))
            abs_s = abs(s)
            if abs_s > new_max:
                new_max = abs_s
    return new_max

def _simulated_annealing(signs, progressions, start_temp, end_temp, n_iterations, deadline, rng):
    """Simulated annealing to minimize max discrepancy."""
    signs = np.array(signs, dtype=np.int8)
    current_disc = _max_discrepancy(signs, progressions)
    best_signs = signs.copy()
    best_disc = current_disc
    
    for i in range(n_iterations):
        if time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (i / n_iterations)
        
        pos = rng.randint(0, 255)
        old_sign = signs[pos]
        new_sign = -old_sign
        signs[pos] = new_sign
        
        new_disc = _max_discrepancy(signs, progressions)
        
        if new_disc <= current_disc:
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        else:
            acceptance_prob = np.exp(-(new_disc - current_disc) / max(temp, 1e-10))
            if rng.random() < acceptance_prob:
                current_disc = new_disc
            else:
                signs[pos] = old_sign
    
    return best_signs.tolist(), best_disc

def _csp_backtracking(progressions, max_bound, deadline, rng):
    """CSP backtracking with forward checking to find solution within max_bound."""
    signs = np.zeros(256, dtype=np.int8)
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == 256:
            return True
        
        for sign in [1, -1]:
            signs[pos] = sign
            # Check if any progression containing pos exceeds bound
            valid = True
            for idx in progressions:
                if pos in idx:
                    s = int(np.sum(signs[idx]))
                    if abs(s) > max_bound:
                        valid = False
                        break
            if valid:
                if backtrack(pos + 1):
                    return True
            signs[pos] = 0
        
        return False
    
    if backtrack(0):
        return signs.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    rng = random.Random(42)
    progressions = _precompute_progressions(256)
    
    # Start with Thue-Morse sequence
    thue_morse = []
    for i in range(256):
        bit_count = bin(i).count('1')
        thue_morse.append(1 if bit_count % 2 == 0 else -1)
    
    best_signs = thue_morse
    best_disc = _max_discrepancy(np.array(best_signs), progressions)
    
    # Try CSP with decreasing bounds
    for bound in range(best_disc - 1, 2, -1):
        if time.time() > deadline - 20:
            break
        result = _csp_backtracking(progressions, bound, deadline - 20, rng)
        if result is not None:
            best_signs = result
            best_disc = bound
            break
    
    # Simulated annealing refinement
    if time.time() < deadline - 5:
        refined_signs, refined_disc = _simulated_annealing(
            best_signs, progressions, 10.0, 0.1, 5000, deadline - 5, rng
        )
        if refined_disc < best_disc:
            best_signs = refined_signs
            best_disc = refined_disc
    
    return best_signs
# EVOLVE-BLOCK-END
