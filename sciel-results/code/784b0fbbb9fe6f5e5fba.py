"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    num_targets = max_sum - min_sum + 1  # 129
    
    def compute_reps(A):
        """Compute representation counts for all target sums."""
        A = sorted(A)
        reps = np.zeros(num_targets, dtype=np.int32)
        m = len(A)
        for i in range(m):
            for j in range(i, m):
                s = A[i] + A[j]
                if min_sum <= s <= max_sum:
                    reps[s - min_sum] += 1
        return reps
    
    def is_valid(A):
        """Check if A covers all target sums."""
        reps = compute_reps(A)
        return np.all(reps > 0)
    
    def get_peak(A):
        """Get peak multiplicity, or inf if not valid."""
        reps = compute_reps(A)
        if np.any(reps == 0):
            return float('inf')
        return int(np.max(reps))
    
    # Try different sizes for A
    best_A = None
    best_peak = float('inf')
    
    # Start with size 17 (minimum for theoretical peak 2)
    for m in range(17, 22):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initial A
            A = set(random.sample(range(0, n+1), m))
            
            # Local search
            current_peak = get_peak(A)
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = list(A)
            
            # Iterate local search
            for iteration in range(2000):
                if time.time() > deadline - 3:
                    break
                
                if current_peak == 2:
                    # Can't do better than 2 (since avg > 1)
                    break
                
                # Try swaps: remove one element, add another
                improved = False
                A_list = list(A)
                random.shuffle(A_list)
                
                for idx in range(len(A_list)):
                    if time.time() > deadline - 3:
                        break
                    elem_to_remove = A_list[idx]
                    A_temp = A - {elem_to_remove}
                    
                    # Try adding each possible element
                    candidates = [x for x in range(0, n+1) if x not in A_temp]
                    random.shuffle(candidates)
                    
                    for new_elem in candidates:
                        A_new = A_temp | {new_elem}
                        new_peak = get_peak(A_new)
                        if new_peak < current_peak:
                            A = A_new
                            current_peak = new_peak
                            if current_peak < best_peak:
                                best_peak = current_peak
                                best_A = list(A)
                            A_list = list(A)
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    # Try add operation (if size < max)
                    if len(A) < 21:
                        candidates = [x for x in range(0, n+1) if x not in A]
                        random.shuffle(candidates)
                        for new_elem in candidates:
                            A_new = A | {new_elem}
                            new_peak = get_peak(A_new)
                            if new_peak < current_peak:
                                A = A_new
                                current_peak = new_peak
                                if current_peak < best_peak:
                                    best_peak = current_peak
                                    best_A = list(A)
                                break
                    else:
                        # Try remove operation (if size > min)
                        if len(A) > 17:
                            A_list = list(A)
                            random.shuffle(A_list)
                            for elem_to_remove in A_list:
                                A_new = A - {elem_to_remove}
                                new_peak = get_peak(A_new)
                                if new_peak < current_peak:
                                    A = A_new
                                    current_peak = new_peak
                                    if current_peak < best_peak:
                                        best_peak = current_peak
                                        best_A = list(A)
                                    break
    
    if best_A is None:
        # Fallback: return seed solution
        return list(range(64, 129))
    
    return best_A
# EVOLVE-BLOCK-END
