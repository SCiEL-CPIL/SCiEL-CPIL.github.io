"""Solve additive base problem: find A ⊆ [0,128] covering [128,256] by pair sums, minimizing peak multiplicity."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    n = 128
    target_lo, target_hi = 128, 256  # inclusive
    target_range = target_hi - target_lo + 1  # 129 values

    def time_left():
        return DEADLINE - (time.time() - t0)

    def compute_multiplicities(A_set):
        """Compute multiplicity for each target using FFT convolution."""
        arr = np.zeros(2 * n + 1)
        for a in A_set:
            arr[a] = 1.0
        # Convolve with itself
        conv = np.fft.fft(arr, n=4 * n + 1)
        conv = conv * conv
        conv = np.fft.ifft(conv).real
        # Extract target range
        muls = conv[target_lo:target_hi + 1].round().astype(int)
        return muls

    def peak_mult(A_set):
        muls = compute_multiplicities(A_set)
        return int(muls.max()) if len(muls) > 0 else 0

    def is_valid(A_set):
        muls = compute_multiplicities(A_set)
        return np.all(muls >= 1)

    def variance_of_muls(A_set):
        muls = compute_multiplicities(A_set).astype(float)
        return float(np.var(muls))

    # --- Structured initialization ---
    def make_grid_set():
        """Create a set based on a coarse grid with refinements."""
        A = set()
        # Base grid
        for i in range(0, n + 1, 3):
            A.add(i)
        # Add some midpoints
        for i in range(1, n + 1, 3):
            if random.random() < 0.5:
                A.add(i)
        # Ensure validity
        for _ in range(100):
            if is_valid(A):
                break
            muls = compute_multiplicities(A)
            uncovered = [target_lo + i for i in range(len(muls)) if muls[i] == 0]
            if not uncovered:
                break
            for t in uncovered:
                for a in range(0, n + 1):
                    b = t - a
                    if 0 <= b <= n:
                        A.add(a)
                        A.add(b)
                        break
                break
        return A

    def make_initial_set():
        """Create a valid initial set using symmetric structure."""
        A = set()
        step = 2
        for i in range(0, n + 1, step):
            A.add(i)
        for _ in range(50):
            if is_valid(A):
                break
            muls = compute_multiplicities(A)
            uncovered = [target_lo + i for i in range(len(muls)) if muls[i] == 0]
            if not uncovered:
                break
            added = False
            for t in uncovered:
                for a in range(0, n + 1):
                    b = t - a
                    if 0 <= b <= n and a not in A and b not in A:
                        A.add(a)
                        A.add(b)
                        added = True
                        break
                if added:
                    break
            if not added:
                A.add(random.randint(0, n))
        return A

    # Generate initial set
    A = make_grid_set()
    if not is_valid(A):
        A = make_initial_set()

    # Reduce size while maintaining validity
    for _ in range(200):
        if time_left() < 10:
            break
        removed = False
        for a in list(A):
            A.remove(a)
            if is_valid(A):
                removed = True
            else:
                A.add(a)
            if removed:
                break
        if not removed:
            break

    best_peak = peak_mult(A)
    best_A = set(A)

    # --- Local search with smooth surrogate objective ---
    def local_search(A, max_iters=5000):
        """Local search to minimize peak multiplicity using smooth surrogate."""
        A = set(A)
        current_peak = peak_mult(A)
        best_peak = current_peak
        best_A = set(A)
        
        # Use a smooth surrogate: penalize peaks heavily via soft-max
        def smooth_objective(A_set):
            muls = compute_multiplicities(A_set).astype(float)
            # Soft-max with temperature to smooth the max
            temp = 5.0
            log_sum = np.log(np.sum(np.exp(muls / temp))) * temp
            # Also penalize variance
            var_penalty = np.var(muls) * 0.1
            # Penalize uncovered targets heavily
            uncovered = np.sum(muls < 1) * 100
            return log_sum + var_penalty + uncovered

        current_obj = smooth_objective(A)
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Try removing an element
                if len(A) <= 10:
                    continue
                a = random.choice(list(A))
                A.remove(a)
                if is_valid(A):
                    new_peak = peak_mult(A)
                    new_obj = smooth_objective(A)
                    if new_peak <= current_peak and new_obj < current_obj:
                        current_peak = new_peak
                        current_obj = new_obj
                        if new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(A)
                    else:
                        A.add(a)
                else:
                    A.add(a)
            
            elif move_type < 0.6:
                # Try swapping: remove one, add another
                a = random.choice(list(A))
                b = random.randint(0, n)
                if b == a or b in A:
                    continue
                A.remove(a)
                A.add(b)
                if is_valid(A):
                    new_peak = peak_mult(A)
                    new_obj = smooth_objective(A)
                    if new_peak <= current_peak and new_obj < current_obj:
                        current_peak = new_peak
                        current_obj = new_obj
                        if new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(A)
                    else:
                        A.add(a)
                        A.remove(b)
                else:
                    A.add(a)
                    A.remove(b)
            
            else:
                # Try adding an element (to help balance)
                b = random.randint(0, n)
                if b in A:
                    continue
                A.add(b)
                new_peak = peak_mult(A)
                new_obj = smooth_objective(A)
                old_obj = smooth_objective(best_A)
                if new_peak <= current_peak and new_obj < old_obj - 0.5:
                    current_peak = new_peak
                    current_obj = new_obj
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(A)
                else:
                    A.remove(b)
        
        return best_A, best_peak

    # Run local search
    A, best_peak = local_search(A, max_iters=3000)

    # Second pass with different initialization
    if time_left() > 15:
        A2 = make_grid_set()
        if is_valid(A2):
            for _ in range(100):
                if time_left() < 10:
                    break
                removed = False
                for a in list(A2):
                    A2.remove(a)
                    if is_valid(A2):
                        removed = True
                    else:
                        A2.add(a)
                    if removed:
                        break
                if not removed:
                    break
            A2, peak2 = local_search(A2, max_iters=2000)
            if peak2 < best_peak:
                best_peak = peak2
                A = A2

    # Final polish: targeted improvements on high-multiplicity targets
    if time_left() > 10:
        for _ in range(500):
            if time_left() < 5:
                break
            muls = compute_multiplicities(A)
            if muls.max() <= 4:
                break
            max_idx = int(np.argmax(muls))
            max_target = target_lo + max_idx
            max_count = muls[max_idx]
            
            pairs = []
            A_list = list(A)
            for i, a in enumerate(A_list):
                b = max_target - a
                if b in A and (a <= b or b not in A_list[i+1:]):
                    pairs.append((a, b))
            
            pair_count = Counter()
            for a, b in pairs:
                pair_count[a] += 1
                if a != b:
                    pair_count[b] += 1
            
            for elem, cnt in pair_count.most_common():
                if cnt >= 2:
                    A.remove(elem)
                    if is_valid(A):
                        new_peak = peak_mult(A)
                        if new_peak < best_peak:
                            best_peak = new_peak
                            A = set(A)
                    else:
                        A.add(elem)
                    break

    # Ensure final validity
    if not is_valid(A):
        muls = compute_multiplicities(A)
        for i in range(len(muls)):
            if muls[i] == 0:
                t = target_lo + i
                for a in range(0, n + 1):
                    b = t - a
                    if 0 <= b <= n:
                        A.add(a)
                        A.add(b)
                        break

    return sorted(A)
# EVOLVE-BLOCK-END
