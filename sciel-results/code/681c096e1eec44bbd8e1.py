import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def compute_reps(A_arr):
        """Compute rep counts for sums 128..256 using vectorized ops."""
        # A_arr is sorted numpy array
        n = len(A_arr)
        # Compute all pair sums efficiently
        # For each i, compute A_arr[i] + A_arr[i:]
        reps = np.zeros(129, dtype=np.int32)
        for i in range(n):
            sums = A_arr[i] + A_arr[i:]
            # Filter to [128, 256]
            mask = (sums >= 128) & (sums <= 256)
            if np.any(mask):
                valid_sums = sums[mask] - 128
                # Count occurrences
                hist, _ = np.histogram(valid_sums, bins=np.arange(130)-0.5, range=(0, 129))
                reps += hist
        return reps
    
    def is_valid(reps):
        return np.all(reps > 0)
    
    def peak_mult(reps):
        return int(np.max(reps))
    
    # Start with seed
    A = np.array(list(range(64, 129)), dtype=np.int32)
    reps = compute_reps(A)
    current_peak = peak_mult(reps)
    
    if not is_valid(reps):
        return list(A)
    
    random.seed(123)
    max_iters = 3000
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        A_set = set(A.tolist())
        A_list = A.tolist()
        
        # Try removing
        if len(A) > 25:
            elem = random.choice(A_list)
            A_test = np.array([x for x in A_list if x != elem], dtype=np.int32)
            reps_test = compute_reps(A_test)
            if is_valid(reps_test):
                p = peak_mult(reps_test)
                if p < current_peak:
                    A = A_test
                    current_peak = p
                    continue
        
        # Try adding
        if len(A) < 90:
            candidates = [x for x in range(0, 129) if x not in A_set]
            if candidates:
                elem = random.choice(candidates)
                A_test = np.array(A_list + [elem], dtype=np.int32)
                reps_test = compute_reps(A_test)
                if is_valid(reps_test):
                    p = peak_mult(reps_test)
                    if p < current_peak:
                        A = A_test
                        current_peak = p
                        continue
        
        # Try swap
        if len(A) > 25:
            elem_rm = random.choice(A_list)
            candidates_add = [x for x in range(0, 129) if x not in A_set]
            if candidates_add:
                elem_add = random.choice(candidates_add)
                A_test = np.array([x for x in A_list if x != elem_rm] + [elem_add], dtype=np.int32)
                reps_test = compute_reps(A_test)
                if is_valid(reps_test):
                    p = peak_mult(reps_test)
                    if p < current_peak:
                        A = A_test
                        current_peak = p
                        continue
    
    return sorted(A.tolist())
