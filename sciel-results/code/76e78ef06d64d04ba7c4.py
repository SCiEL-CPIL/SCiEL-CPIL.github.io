import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128

    def calc_reps_fast(A_set):
        """Use FFT convolution to count representations quickly."""
        if not A_set:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        A_arr = np.zeros(MAX_VAL + 1, dtype=np.float64)
        for a in A_set:
            A_arr[a] = 1.0
        # Convolution: c[s] = sum of A_arr[i]*A_arr[s-i] for all i
        # Use FFT for speed
        n_fft = 1
        while n_fft < 2 * (MAX_VAL + 1):
            n_fft *= 2
        fa = np.fft.rfft(A_arr, n=n_fft)
        fa2 = fa * fa
        conv = np.fft.irfft(fa2, n=n_fft)
        # conv[s] = number of ordered pairs (i,j) with i+j=s
        # For unordered with a<=b: reps = (conv[s] + diag[s]) / 2
        # where diag[s] = 1 if s/2 in A and s even, else 0
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx in range(RANGE_SIZE):
            s = N_LOW + idx
            ordered = int(round(conv[s]))
            # diagonal: number of a in A with 2a = s, i.e., a = s/2
            diag = 1 if (s % 2 == 0 and (s // 2) in A_set) else 0
            reps[idx] = (ordered + diag) // 2
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps)) if len(reps) > 0 else 0

    def get_sum_squares(reps):
        return int(np.sum(reps * reps))

    def get_objective(reps):
        """Combined objective: peak * 1000 + sum_squares (lexicographic-ish)"""
        return get_peak(reps) * 10000 + get_sum_squares(reps)

    # Phase 1: Structured initialization with symmetric sets
    # Try several different sizes and symmetric structures
    best_A = None
    best_reps = None
    best_obj = float('inf')

    def try_set(A_set):
        nonlocal best_A, best_reps, best_obj
        if not A_set:
            return
        reps = calc_reps_fast(A_set)
        if is_covered(reps):
            obj = get_objective(reps)
            if obj < best_obj:
                best_obj = obj
                best_A = set(A_set)
                best_reps = reps

    # Generate symmetric candidate sets
    # Symmetric about 64: if x in A, then 128-x in A
    # This means sums concentrate in [128, 256]
    
    # Try various sizes of symmetric sets
    for target_size in range(20, 80, 2):
        if time.time() > deadline - 5:
            break
        # Build symmetric set of approximately target_size
        # Pairs (x, 128-x) for x in [0, 64], plus possibly 64 itself
        # Number of independent choices: 65 (x=0..64), each contributes 1 or 2 elements
        # For size ~ target_size, we need ~target_size/2 pairs + maybe 64
        
        n_pairs = target_size // 2
        A_set = set()
        
        # Always include 0 and 128 (they form a pair summing to 128)
        A_set.add(0)
        A_set.add(128)
        
        # Add random symmetric pairs
        candidates = list(range(1, 64))  # x values where 128-x != x
        random.shuffle(candidates)
        
        for x in candidates:
            if len(A_set) >= target_size - 1:
                break
            A_set.add(x)
            A_set.add(128 - x)
        
        # Maybe add 64
        if len(A_set) < target_size and random.random() < 0.5:
            A_set.add(64)
        
        try_set(A_set)
        
        # Also try with some asymmetry
        if time.time() > deadline - 5:
            continue
        A_set2 = set(A_set)
        # Remove a few random elements and add different ones
        for _ in range(3):
            if A_set2:
                to_remove = random.choice(list(A_set2))
                A_set2.remove(to_remove)
                # Add a random element
                new_elem = random.randint(0, MAX_VAL)
                A_set2.add(new_elem)
        try_set(A_set2)

    # If no valid set found, use fallback
    if best_A is None:
        # Fallback: use [64, 128]
        A_fallback = list(range(64, 129))
        return A_fallback

    # Phase 2: Local search optimization
    # Try removals and swaps to improve
    
    current_A = set(best_A)
    current_reps = calc_reps_fast(current_A)
    current_obj = get_objective(current_reps)
    
    improved = True
    max_passes = 20
    passes = 0
    
    all_vals = list(range(MAX_VAL + 1))
    
    while improved and time.time() < deadline - 3 and passes < max_passes:
        improved = False
        passes += 1
        
        # Try removals
        for a in list(current_A):
            if time.time() > deadline - 3:
                break
            new_A = current_A - {a}
            if not new_A:
                continue
            new_reps = calc_reps_fast(new_A)
            if is_covered(new_reps):
                new_obj = get_objective(new_reps)
                if new_obj < current_obj:
                    current_A = new_A
                    current_reps = new_reps
                    current_obj = new_obj
                    improved = True
        
        if time.time() > deadline - 3:
            break
        
        # Try swaps
        A_list = list(current_A)
        non_A = [v for v in all_vals if v not in current_A]
        
        random.shuffle(A_list)
        random.shuffle(non_A)
        
        for a in A_list:
            if time.time() > deadline - 3:
                break
            for b in non_A:
                if time.time() > deadline - 3:
                    break
                new_A = current_A - {a}
                new_A.add(b)
                new_reps = calc_reps_fast(new_A)
                if is_covered(new_reps):
                    new_obj = get_objective(new_reps)
                    if new_obj < current_obj:
                        current_A = new_A
                        current_reps = new_reps
                        current_obj = new_obj
                        improved = True
                        # Restart with new sets
                        A_list = list(current_A)
                        non_A = [v for v in all_vals if v not in current_A]
                        break
            if improved:
                break
        
        # If no improvement, try random perturbation
        if not improved and time.time() < deadline - 5:
            for _ in range(random.randint(2, 4)):
                if not current_A:
                    break
                a = random.choice(list(current_A))
                b = random.choice(all_vals)
                if b in current_A:
                    continue
                new_A = current_A - {a}
                new_A.add(b)
                new_reps = calc_reps_fast(new_A)
                if is_covered(new_reps):
                    new_obj = get_objective(new_reps)
                    if new_obj <= current_obj:
                        current_A = new_A
                        current_reps = new_reps
                        current_obj = new_obj
                        improved = True
                        break

    # Final check
    final_reps = calc_reps_fast(current_A)
    if not is_covered(final_reps):
        # Fallback
        return list(range(64, 129))
    
    return list(current_A)

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
