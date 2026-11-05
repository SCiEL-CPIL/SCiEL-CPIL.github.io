import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    min_sum = 128
    max_sum = 256
    S = max_sum - min_sum + 1  # 129 sums

    def get_counts(mask):
        """Fast count of unordered pair sums (a<=b) in [128,256] using FFT."""
        m = len(mask)
        # Full linear convolution of mask with itself
        N = 1 << (2 * m - 1).bit_length()
        f = np.fft.rfft(mask.astype(np.float64), n=N)
        conv = np.fft.irfft(f * f, n=N)[:2 * m - 1]
        conv = np.rint(conv).astype(np.int64)
        
        # Extract sums in [128, 256]
        counts_ordered = conv[min_sum:max_sum + 1]
        
        # Calculate diagonal elements (a == b) for even sums
        # s is even if (s - min_sum) is even? No, s itself matters.
        # s ranges from 128 to 256.
        # diag[s] = 1 if s is even and s/2 in A.
        # We need to map s to index in conv.
        # Let's compute diag for the specific range.
        
        # Vectorized diagonal calculation:
        # s = min_sum + i
        # half = s / 2
        # We need mask[half] where half is integer.
        
        # Create array of sums
        sums = np.arange(min_sum, max_sum + 1)
        # Half sums
        half_sums = sums / 2.0
        # Check if integer
        is_even = (sums % 2 == 0)
        
        # Initialize diag as 0
        diag = np.zeros(S, dtype=np.int64)
        
        # For even sums, check if half is in mask
        # half_sums for even sums are integers
        # We need to be careful with indexing.
        # half_idx = half_sums[is_even].astype(int)
        # valid_half = (half_idx >= 0) & (half_idx <= n)
        # diag_vals = mask[valid_half] * is_even[valid_half] ? No, mask is 0/1.
        
        # Simpler:
        even_mask_idx = np.where(is_even)[0]
        if len(even_mask_idx) > 0:
            half_indices = half_sums[even_mask_idx].astype(int)
            valid_mask = (half_indices >= 0) & (half_indices <= n)
            # Only count if mask[half] is 1
            diag[even_mask_idx[valid_mask]] = mask[half_indices[valid_mask]]
            
        # Unordered count = (ordered + diag) // 2
        counts = (counts_ordered + diag) // 2
        return counts

    def soft_max(x, beta=40.0):
        """Smooth max via log-sum-exp."""
        x = np.asarray(x, dtype=np.float64)
        m = np.max(x)
        return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

    def get_cost(mask):
        counts = get_counts(mask)
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    def mask_to_list(mask):
        return [i for i in range(n + 1) if mask[i]]

    def list_to_mask(A_list):
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A_list:
            mask[x] = 1
        return mask

    best_score = float('inf')
    best_mask = None

    def update_best(mask):
        nonlocal best_score, best_mask
        if not np.any(mask):
            return
        cost = get_cost(mask)
        if cost < best_score:
            best_score = cost
            best_mask = mask.copy()

    # 1. Heuristic Initializations
    # Symmetric ranges
    for offset in range(0, 40):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        mask = np.zeros(n + 1, dtype=np.int64)
        mask[lo:hi + 1] = 1
        update_best(mask)

    # Modular
    for mod in [3, 4, 5, 6, 7, 8, 9, 10]:
        for keep in range(1, mod):
            mask = np.zeros(n + 1, dtype=np.int64)
            for x in range(n + 1):
                if x % mod != mod - keep:
                    mask[x] = 1
            update_best(mask)
            
    # Arithmetic Progressions
    for step in [2, 3, 4, 5, 6, 7, 8]:
        for start in range(step):
            mask = np.zeros(n + 1, dtype=np.int64)
            for x in range(start, n + 1, step):
                mask[x] = 1
            update_best(mask)

    # Random initializations
    random.seed(42)
    for _ in range(80):
        if time.time() > deadline - 40:
            break
        size = random.randint(25, 65)
        A = random.sample(range(n + 1), size)
        mask = list_to_mask(A)
        update_best(mask)

    # 2. Global Optimization with smooth surrogate
    if best_mask is not None and time.time() < deadline - 35:
        try:
            # Start from best mask, encode as continuous
            x0 = best_mask.astype(np.float64).copy()
            # Add small noise to break symmetry
            x0 = np.clip(x0 + random.gauss(0, 0.1, len(x0)), 0, 1)

            def smooth_obj(x):
                if time.time() > deadline - 10:
                    return 1e6
                # Threshold to binary for count computation
                mask = (x > 0.5).astype(np.int64)
                if not np.any(mask):
                    return 1e6
                counts = get_counts(mask).astype(np.float64)
                # Penalize zeros heavily
                zero_penalty = 100.0 * np.sum(counts == 0)
                # Soft max of counts
                sm = soft_max(counts, beta=30.0)
                return sm + zero_penalty

            # Bounded continuous optimization (L-BFGS-B)
            bounds = [(0.0, 1.0)] * (n + 1)
            res = minimize(
                smooth_obj, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'maxfun': 500}
            )
            best_de_mask = (res.x > 0.5).astype(np.int64)
            update_best(best_de_mask)

            # Differential Evolution as backup
            if time.time() < deadline - 25:
                def de_objective(x):
                    mask = (x > 0.5).astype(np.int64)
                    if not np.any(mask):
                        return 10000
                    counts = get_counts(mask)
                    if np.any(counts == 0):
                        return 10000 + int(np.sum(counts == 0))
                    return float(np.max(counts))

                de_res = differential_evolution(
                    de_objective,
                    bounds,
                    strategy='best1bin',
                    popsize=12,
                    maxiter=20,
                    tol=1e-4,
                    seed=42,
                    polish=False,
                    init='latinhypercube'
                )
                if de_res.fun < 10000:
                    best_de_mask = (de_res.x > 0.5).astype(np.int64)
                    update_best(best_de_mask)
        except Exception:
            pass

    # 3. Local Search Refinement
    if best_mask is None:
        best_mask = np.zeros(n + 1, dtype=np.int64)
        best_mask[64:129] = 1
        best_score = get_cost(best_mask)

    current_mask = best_mask.copy()
    current_cost = best_score

    def try_local_search(mask, cost, time_limit):
        nonlocal best_score, best_mask
        current_mask = mask.copy()
        current_cost = cost
        current_counts = get_counts(current_mask)

        while time.time() < time_limit:
            improved = False

            # 1. Try removing (order by least impact)
            ones = [i for i in range(n + 1) if current_mask[i] == 1]
            # Sort ones by their contribution to max counts? 
            # Simpler: random shuffle, but we can prioritize removing from high density areas if it doesn't break coverage.
            random.shuffle(ones)
            for i in ones:
                if time.time() > time_limit:
                    break
                trial_mask = current_mask.copy()
                trial_mask[i] = 0
                if not np.any(trial_mask):
                    continue
                tc = get_cost(trial_mask)
                if tc <= current_cost:
                    current_mask = trial_mask
                    current_cost = tc
                    current_counts = get_counts(current_mask)
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()
                    improved = True
                    break

            # 2. Try adding
            if not improved:
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(zeros)
                for i in zeros:
                    if time.time() > time_limit:
                        break
                    trial_mask = current_mask.copy()
                    trial_mask[i] = 1
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        current_counts = get_counts(current_mask)
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        break

            # 3. Try swapping (Smart Swap)
            if not improved:
                ones = [i for i in range(n + 1) if current_mask[i] == 1]
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                
                # Heuristic: Try swapping elements that are in "high count" regions to "low count" regions
                # We can approximate this by just trying random swaps but with a limit
                random.shuffle(ones)
                random.shuffle(zeros)
                
                # Limit to speed up
                max_swaps = 50
                swaps_done = 0
                for i in ones:
                    if time.time() > time_limit or swaps_done >= max_swaps:
                        break
                    for j in zeros:
                        if time.time() > time_limit or swaps_done >= max_swaps:
                            break
                        trial_mask = current_mask.copy()
                        trial_mask[i] = 0
                        trial_mask[j] = 1
                        tc = get_cost(trial_mask)
                        if tc < current_cost:
                            current_mask = trial_mask
                            current_cost = tc
                            current_counts = get_counts(current_mask)
                            if tc < best_score:
                                best_score = tc
                                best_mask = current_mask.copy()
                            improved = True
                            swaps_done += 1
                            break
                    if improved:
                        break

            # 4. Random perturbation to escape local minima
            if not improved and time.time() < time_limit - 3:
                flip_count = random.randint(2, 5)
                indices = random.sample(range(n + 1), flip_count)
                trial_mask = current_mask.copy()
                for idx in indices:
                    trial_mask[idx] = 1 - trial_mask[idx]
                if np.any(trial_mask):
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        current_counts = get_counts(current_mask)
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                    elif tc == current_cost and random.random() < 0.15:
                        current_mask = trial_mask
                        current_counts = get_counts(current_mask)
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True

            if not improved:
                break

    try_local_search(best_mask, best_score, deadline - 3)

    # Final check
    if best_mask is None or not np.any(best_mask):
        return list(range(64, 129))

    return mask_to_list(best_mask)
