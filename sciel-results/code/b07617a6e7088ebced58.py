import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    n = 128
    target_lo = 128
    target_hi = 256
    targets = list(range(target_lo, target_hi + 1))
    
    def compute_reps(A):
        """Compute representation counts for each target in [128,256]."""
        A_set = set(A)
        A_sorted = sorted(A)
        k = len(A_sorted)
        reps = np.zeros(target_hi - target_lo + 1, dtype=np.int32)
        for i in range(k):
            a = A_sorted[i]
            for j in range(i, k):
                b = A_sorted[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    reps[s - target_lo] += 1
        return reps
    
    def is_valid(A):
        reps = compute_reps(A)
        return np.all(reps > 0)
    
    def peak_mult(A):
        reps = compute_reps(A)
        if np.any(reps == 0):
            return float('inf')
        return int(np.max(reps))
    
    # Initial solution: use a structured set
    # Try starting with A = range(64, 129) which gives peak 33
    # But let's try smaller sets
    
    # Strategy: start with a random valid set, then local search
    best_A = list(range(64, 129))
    best_peak = peak_mult(best_A)
    
    # Try to find a better initial solution with smaller A
    # A set of size k can have at most k(k+1)/2 pairs
    # We need to cover 129 targets
    
    # Let's try various sizes
    for size in range(20, 40):
        if time.time() > deadline - 10:
            break
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            # Create a random set of given size in [0, 128]
            A = set(random.sample(range(0, 129), size))
            # Ensure it's valid by checking
            if is_valid(A):
                p = peak_mult(A)
                if p < best_peak:
                    best_peak = p
                    best_A = sorted(A)
    
    # Local search: try swapping elements
    if time.time() < deadline - 5:
        A = sorted(best_A)
        current_peak = best_peak
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline - 3 and no_improve_count < max_no_improve:
            improved = False
            k = len(A)
            
            # Try removing an element and adding a different one
            for i in range(k):
                if time.time() > deadline - 3:
                    break
                for _ in range(5):  # try a few random replacements
                    if time.time() > deadline - 3:
                        break
                    new_elem = random.randint(0, 128)
                    if new_elem in A:
                        continue
                    # Create new A
                    A_new = A[:i] + A[i+1:] + [new_elem]
                    A_new = sorted(A_new)
                    # Check validity and peak
                    A_set_new = set(A_new)
                    reps = np.zeros(target_hi - target_lo + 1, dtype=np.int32)
                    for ii in range(len(A_new)):
                        a = A_new[ii]
                        for jj in range(ii, len(A_new)):
                            b = A_new[jj]
                            s = a + b
                            if target_lo <= s <= target_hi:
                                reps[s - target_lo] += 1
                    if np.all(reps > 0):
                        p = int(np.max(reps))
                        if p < current_peak:
                            current_peak = p
                            best_peak = p
                            best_A = A_new
                            A = A_new
                            improved = True
                            no_improve_count = 0
                            break
                    elif p < current_peak:
                        pass  # not valid, skip
            
            if not improved:
                no_improve_count += 1
        
        # Also try adding elements to reduce peak
        if time.time() < deadline - 3:
            A = sorted(best_A)
            current_peak = best_peak
            for _ in range(50):
                if time.time() > deadline - 3:
                    break
                new_elem = random.randint(0, 128)
                if new_elem in A:
                    continue
                A_new = sorted(A + [new_elem])
                reps = np.zeros(target_hi - target_lo + 1, dtype=np.int32)
                for ii in range(len(A_new)):
                    a = A_new[ii]
                    for jj in range(ii, len(A_new)):
                        b = A_new[jj]
                        s = a + b
                        if target_lo <= s <= target_hi:
                            reps[s - target_lo] += 1
                if np.all(reps > 0):
                    p = int(np.max(reps))
                    if p < current_peak:
                        current_peak = p
                        best_peak = p
                        best_A = A_new
                        A = A_new
    
    return sorted(best_A)
