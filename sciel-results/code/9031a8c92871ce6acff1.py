import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # seconds, leaving 5s safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    
    # Precompute all possible pairs (i, j) with i <= j where i+j is in target range
    # For fast evaluation: represent A as a boolean array, use convolution-like approach
    # But for speed, we'll use a direct pair enumeration with numpy
    
    def compute_mults_fast(A_bool):
        """Compute representation counts for all targets using A_bool (length 129)."""
        # A_bool[i] = 1 if i in A
        # For each target s, count pairs (i,j) with i<=j, i+j=s, A_bool[i] and A_bool[j]
        mults = np.zeros(target_max + 1, dtype=np.int32)
        
        # For each i in A
        for i in np.where(A_bool)[0]:
            # j must be >= i and i+j <= target_max
            j_min = i
            j_max = min(target_max - i, n)
            if j_min > j_max:
                continue
            # Count j in [j_min, j_max] where A_bool[j] is True
            # For j = i: contributes 1 if A_bool[i]
            # For j > i: contributes A_bool[j]
            if j_min <= i <= j_max:
                mults[i + i] += 1
            if i + 1 <= j_max:
                # Sum A_bool[i+1 : j_max+1]
                s_start = i + 1
                s_end = j_max + 1
                # These correspond to sums i+s_start to i+s_end
                # Add A_bool values to mults at positions i+s_start to i+s_end
                mults[i + s_start : i + s_end + 1] += A_bool[s_start : s_end + 1]
        
        return mults
    
    def evaluate(A_bool):
        """Return (is_valid, peak_mult) for A_bool."""
        mults = compute_mults_fast(A_bool)
        target_mults = mults[targets]
        if np.any(target_mults == 0):
            return False, np.max(target_mults) if np.max(target_mults) > 0 else 0
        return True, int(np.max(target_mults))
    
    # Start with a dense valid set
    A_bool = np.zeros(n + 1, dtype=np.bool_)
    A_bool[64:129] = True  # A = {64, 65, ..., 128}
    
    valid, peak = evaluate(A_bool)
    best_A = A_bool.copy()
    best_peak = peak if valid else float('inf')
    
    if not valid:
        # Ensure we have a valid starting point
        A_bool[:] = True
        valid, peak = evaluate(A_bool)
        best_A = A_bool.copy()
        best_peak = peak if valid else float('inf')
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    current_peak = best_peak
    current_A = best_A.copy()
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        step += 1
        
        # Check for timeout
        if time.time() > deadline:
            break
        
        # Choose a random element to flip
        elem = random.randint(0, n)
        
        # Try flipping
        new_A = current_A.copy()
        new_A[elem] = not new_A[elem]
        
        # Ensure non-empty
        if not np.any(new_A):
            new_A[elem] = True
        
        new_valid, new_peak = evaluate(new_A)
        
        # Compute energy (penalty for invalid + peak)
        if new_valid:
            new_energy = new_peak
        else:
            # Count uncovered targets
            mults = compute_mults_fast(new_A)
            uncovered = np.sum(mults[targets] == 0)
            new_energy = 1000 + uncovered * 10 + new_peak
        
        if current_valid:
            current_energy = current_peak
        else:
            mults_cur = compute_mults_fast(current_A)
            uncovered_cur = np.sum(mults_cur[targets] == 0)
            current_energy = 1000 + uncovered_cur * 10 + current_peak
        
        delta_energy = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
            current_A = new_A
            current_valid = new_valid
            current_peak = new_peak
            
            if new_valid and new_peak < best_peak:
                best_peak = new_peak
                best_A = new_A.copy()
                no_improve_steps = 0
            elif not new_valid:
                no_improve_steps += 1
            else:
                no_improve_steps += 1
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # If stuck for too long, restart temperature
        if no_improve_steps > max_no_improve:
            T = 5.0
            no_improve_steps = 0
        
        # Occasional random restart from a different structure
        if step % 5000 == 0 and time.time() < deadline - 10:
            # Try a few different initial structures
            for restart_seed in range(3):
                if time.time() > deadline - 5:
                    break
                test_A = np.zeros(n + 1, dtype=np.bool_)
                # Random subset with bias toward middle
                for i in range(40):
                    test_A[random.randint(0, n)] = True
                # Ensure some coverage
                test_A[0] = True
                test_A[128] = True
                
                rv, rp = evaluate(test_A)
                if rv and rp < best_peak:
                    best_peak = rp
                    best_A = test_A.copy()
                    current_A = test_A.copy()
                    current_valid = True
                    current_peak = rp
    
    return np.where(best_A)[0].tolist()
