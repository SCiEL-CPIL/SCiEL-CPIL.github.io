"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

N = 256

def _max_abs_aps(x):
    """Compute the maximum absolute signed sum over all arithmetic progressions.
    
    x is a list or array of length N with entries in {-1, +1}.
    """
    max_abs = 0
    # For each step d
    for d in range(1, N + 1):
        # For each residue class r in [0, d)
        for r in range(d):
            # The AP is x[r], x[r+d], x[r+2d], ...
            # Build the list of values in this AP
            chain = []
            pos = r
            while pos < N:
                chain.append(x[pos])
                pos += d
            # Now find the max absolute sum of any contiguous subsequence in chain
            # (since any AP contained in this chain starting at some index and of some length is a contiguous subsequence)
            if len(chain) == 0:
                continue
            # Kadane's algorithm for max subarray sum and min subarray sum
            max_so_far = 0
            min_so_far = 0
            current_max = 0
            current_min = 0
            for v in chain:
                current_max = max(0, current_max + v)
                current_min = min(0, current_min + v)
                if current_max > max_so_far:
                    max_so_far = current_max
                if abs(current_min) > abs(min_so_far):
                    min_so_far = current_min
            cand = max(max_so_far, abs(min_so_far))
            if cand > max_abs:
                max_abs = cand
    return max_abs

def _evaluate(x):
    return _max_abs_aps(x)

def _flip(x, i):
    x[i] = -x[i]

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s or 90s kill
    
    # Initial random seed
    rng = random.Random(42)
    
    best_x = [rng.choice((-1, 1)) for _ in range(N)]
    best_score = _evaluate(best_x)
    
    # Also try a few structured initializations
    # Alternating
    x_alt = [1 if i % 2 == 0 else -1 for i in range(N)]
    s_alt = _evaluate(x_alt)
    if s_alt < best_score:
        best_score = s_alt
        best_x = x_alt[:]
    
    # Random with different seeds
    for seed in [171, 123, 456, 789, 314, 271]:
        x_try = [random.Random(seed).choice((-1, 1)) for _ in range(N)]
        s_try = _evaluate(x_try)
        if s_try < best_score:
            best_score = s_try
            best_x = x_try[:]
    
    # Simulated Annealing with multiple flips and better schedule
    current_x = best_x[:]
    current_score = best_score
    
    T_start = 3.0
    T_end = 0.05
    total_iters = 200000  # nominal; time is the real guard
    
    iter_count = 0
    no_improve_count = 0
    stall_threshold = 5000
    
    while time.time() - start_time < deadline:
        # Adaptive temperature based on progress
        progress = iter_count / total_iters
        T = T_start * (T_end / T_start) ** progress
        T = max(T, T_end)
        
        # Decide number of flips (1-3)
        n_flips = 1 if rng.random() < 0.7 else (2 if rng.random() < 0.8 else 3)
        
        # Pick positions to flip
        indices = rng.sample(range(N), n_flips)
        for i in indices:
            _flip(current_x, i)
        
        new_score = _evaluate(current_x)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_x = current_x[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T) if T > 0 else 0.0
                if rng.random() < prob:
                    current_score = new_score
                    no_improve_count += 1
                else:
                    # Revert
                    for i in indices:
                        _flip(current_x, i)
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # If stuck for too long, do a restart from best
        if no_improve_count > stall_threshold:
            current_x = best_x[:]
            current_score = best_score
            no_improve_count = 0
            # Boost temperature temporarily
            T_start = max(T_start * 1.5, 1.0)
        
        iter_count += 1
    
    # Final polish: local search with single flips only, very low temperature
    polish_start = time.time()
    polish_deadline = min(deadline, start_time + 55.0)
    polish_x = best_x[:]
    polish_score = best_score
    
    while time.time() < polish_deadline:
        i = rng.randint(0, N - 1)
        _flip(polish_x, i)
        new_score = _evaluate(polish_x)
        
        if new_score < polish_score:
            polish_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = polish_x[:]
        else:
            _flip(polish_x, i)
            # If no improvement, try a different position
            if rng.random() < 0.1:
                # Occasionally try a random restart from best
                polish_x = best_x[:]
                polish_score = best_score
    
    return best_x
# EVOLVE-BLOCK-END
