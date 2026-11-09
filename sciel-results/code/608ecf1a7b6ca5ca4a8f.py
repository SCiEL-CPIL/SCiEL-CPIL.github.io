"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for 60s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A_set):
        """Calculate representation counts for sums in [N_LOW, N_HIGH] using unordered pairs."""
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = sorted(A_set)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            # For each ai, consider pairs (ai, aj) where j >= i
            # Sums are ai + A_list[i:]
            sums = ai + np.array(A_list[i:], dtype=np.int32)
            mask = (sums >= N_LOW) & (sums <= N_HIGH)
            if np.any(mask):
                valid_sums = sums[mask] - N_LOW
                # Use np.bincount for efficient counting
                counts = np.bincount(valid_sums, minlength=RANGE_SIZE)
                reps += counts
        return reps
    
    def is_valid(A_set):
        """Check if every integer in [N_LOW, N_HIGH] has at least one representation."""
        reps = calc_reps(A_set)
        return np.all(reps > 0)
    
    def get_peak(A_set):
        """Get the maximum representation multiplicity."""
        reps = calc_reps(A_set)
        return int(np.max(reps))
    
    # Start with a valid set: [64, 128]
    best_A = set(range(64, 129))
    best_peak = get_peak(best_A)
    
    # Try to improve by removing elements
    # Greedy removal: try removing each element, keep if still valid
    for _ in range(10):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        for elem in list(best_A):
            if time.time() > deadline:
                break
            candidate = best_A - {elem}
            if not candidate:
                continue
            if is_valid(candidate):
                candidate_peak = get_peak(candidate)
                if candidate_peak <= best_peak:
                    best_A = candidate
                    best_peak = candidate_peak
                    improved = True
    
    # Local search: try swapping elements
    all_elements = set(range(MAX_VAL + 1))
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Try random swaps: remove one element, add another
        if len(best_A) < MAX_VAL + 1:
            # Try adding a random element
            candidates_to_add = list(all_elements - best_A)
            if candidates_to_add:
                elem_to_add = random.choice(candidates_to_add)
                candidate = best_A | {elem_to_add}
                if is_valid(candidate):
                    candidate_peak = get_peak(candidate)
                    if candidate_peak < best_peak:
                        best_A = candidate
                        best_peak = candidate_peak
        
        # Try random removals
        if len(best_A) > 1:
            elem_to_remove = random.choice(list(best_A))
            candidate = best_A - {elem_to_remove}
            if is_valid(candidate):
                candidate_peak = get_peak(candidate)
                if candidate_peak <= best_peak:
                    best_A = candidate
                    best_peak = candidate_peak
        
        # Try random swaps (remove one, add another)
        if len(best_A) > 1:
            elem_to_remove = random.choice(list(best_A))
            candidates_to_add = list(all_elements - best_A)
            if candidates_to_add:
                elem_to_add = random.choice(candidates_to_add)
                candidate = best_A - {elem_to_remove} | {elem_to_add}
                if is_valid(candidate):
                    candidate_peak = get_peak(candidate)
                    if candidate_peak < best_peak:
                        best_A = candidate
                        best_peak = candidate_peak
    
    return list(best_A)
# EVOLVE-BLOCK-END
