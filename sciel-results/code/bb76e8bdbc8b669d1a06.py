import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 58.0  # safety margin before 60s wall-clock kill
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = np.arange(target_lo, target_hi + 1)
    num_targets = target_hi - target_lo + 1  # 129 targets
    
    def compute_counts(A_set):
        """Compute pair-sum counts for unordered pairs (i<=j) from A_set."""
        counts = np.zeros(513, dtype=np.int32)  # max sum is 256
        A_list = sorted(A_set)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            for j in range(i, m):
                aj = A_list[j]
                counts[ai + aj] += 1
        return counts
    
    def get_metrics(counts):
        """Return (peak, uncovered_count, sum_of_squares) for target range."""
        seg = counts[target_lo:target_hi + 1]
        peak = int(np.max(seg))
        uncovered = int(np.sum(seg == 0))
        ss = int(np.sum(seg ** 2))
        return peak, uncovered, ss
    
    # === Structured initializations ===
    # Strategy: use structured sets that are likely to have low peak multiplicities
    
    def make_even_set():
        # All even numbers in [0, 128]
        return set(range(0, 129, 2))
    
    def make_odd_set():
        # All odd numbers in [0, 127] plus some evens to cover even sums
        return set(range(1, 128, 2))
    
    def make_mid_set():
        # Numbers around the middle: [32, 96]
        return set(range(32, 97))
    
    def make_spaced_set(step=4):
        # Spaced set with some fill
        s = set(range(0, 129, step))
        return s
    
    def make_hybrid():
        # Combine even and odd strategically
        evens = set(range(0, 129, 2))
        odds = set(range(1, 128, 2))
        # Take all evens and half the odds
        odd_list = sorted(odds)
        half_odds = set(odd_list[::2])
        return evens | half_odds
    
    def make_banded():
        # Two bands: [0,32] and [96,128]
        return set(range(0, 33)) | set(range(96, 129))
    
    def make_quarter():
        # Four quarter-ranges
        return set(range(0, 33)) | set(range(32, 65)) | set(range(64, 97)) | set(range(96, 129))
    
    def make_golden():
        # Golden-ratio spaced
        phi = (1 + 5**0.5) / 2
        s = set()
        for k in range(64):
            val = int(k * 128 / 64 * phi) % 129
            s.add(val)
        return s
    
    initializers = [
        make_even_set,
        make_odd_set,
        make_mid_set,
        lambda: make_spaced_set(4),
        lambda: make_spaced_set(3),
        make_hybrid,
        make_banded,
        make_quarter,
        make_golden,
    ]
    
    best_A = None
    best_peak = float('inf')
    best_ss = float('inf')
    best_unc = float('inf')
    
    # Evaluate all initializations
    for init_fn in initializers:
        if time.time() - start_time > time_limit:
            break
        A = init_fn()
        if not A:
            continue
        counts = compute_counts(A)
        peak, unc, ss = get_metrics(counts)
        
        # If uncovered, try to fix by adding elements
        if unc > 0:
            # Greedy fix: add elements that cover most uncovered targets
            A_fixed = set(A)
            counts_fixed = counts.copy()
            for _ in range(50):  # max 50 additions
                if time.time() - start_time > time_limit:
                    break
                seg = counts_fixed[target_lo:target_hi + 1]
                if np.all(seg > 0):
                    break
                # Find uncovered targets
                unc_targets = [target_lo + i for i in range(num_targets) if seg[i] == 0]
                if not unc_targets:
                    break
                # Try each candidate element not in A_fixed
                best_gain = 0
                best_elem = None
                for cand in range(n + 1):
                    if cand in A_fixed:
                        continue
                    # Count how many uncovered targets this would cover
                    gain = 0
                    for t in unc_targets:
                        other = t - cand
                        if 0 <= other <= n and other in A_fixed:
                            gain += 1
                        # Also check if other == cand (i.e., 2*cand == t)
                        if other == cand and 2 * cand == t:
                            # This is already counted if cand in A_fixed, but it's not
                            # Actually if cand not in A_fixed, we're adding it, so (cand, cand) becomes available
                            pass
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = cand
                if best_elem is None or best_gain == 0:
                    break
                A_fixed.add(best_elem)
                counts_fixed = compute_counts(A_fixed)
            A = A_fixed
            counts = counts_fixed
            peak, unc, ss = get_metrics(counts)
        
        if unc == 0:  # Only consider valid solutions
            if (peak < best_peak or 
                (peak == best_peak and ss < best_ss) or
                (peak == best_peak and ss == best_ss and unc < best_unc)):
                best_peak = peak
                best_ss = ss
                best_unc = unc
                best_A = set(A)
    
    # If no valid solution found, fall back to a large set
    if best_A is None:
        best_A = set(range(64, 129))
        counts = compute_counts(best_A)
        best_peak, best_unc, best_ss = get_metrics(counts)
    
    # === Local search phase ===
    curr_A = set(best_A)
    curr_counts = compute_counts(curr_A)
    curr_peak, curr_unc, curr_ss = get_metrics(curr_counts)
    
    def try_remove(A_set, counts_arr, x):
        """Try removing x. Return (new_counts, valid) or (None, False)."""
        old_counts = counts_arr.copy()
        A_trial = set(A_set)
        A_trial.remove(x)
        
        # Subtract pairs involving x
        # Pairs (x, y) for y in A_trial where y != x, and (x, x)
        for y in A_trial:
            old_counts[x + y] -= 1
        old_counts[2 * x] -= 1
        
        # Check validity
        seg = old_counts[target_lo:target_hi + 1]
        if np.any(seg == 0):
            return None, False
        return old_counts, True
    
    def try_add(A_set, counts_arr, x):
        """Try adding x. Return new_counts."""
        new_counts = counts_arr.copy()
        for y in A_set:
            new_counts[x + y] += 1
        new_counts[2 * x] += 1
        return new_counts
    
    def try_swap(A_set, counts_arr, x_out, x_in):
        """Try swapping x_out for x_in. Return new_counts or None if invalid."""
        old_counts = counts_arr.copy()
        A_trial = set(A_set)
        A_trial.remove(x_out)
        A_trial.add(x_in)
        
        # Remove x_out contributions
        for y in A_trial:
            old_counts[x_out + y] -= 1
        old_counts[2 * x_out] -= 1
        
        # Add x_in contributions
        for y in A_trial:
            old_counts[x_in + y] += 1
        old_counts[2 * x_in] += 1
        
        seg = old_counts[target_lo:target_hi + 1]
        if np.any(seg == 0):
            return None
        return old_counts
    
    # Simulated annealing style local search
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try removals
        A_list = list(curr_A)
        random.shuffle(A_list)
        
        for x in A_list:
            if time.time() - start_time > time_limit:
                break
            if len(curr_A) <= 2:
                break
            
            new_counts, valid = try_remove(curr_A, curr_counts, x)
            if not valid:
                continue
            
            new_peak, new_unc, new_ss = get_metrics(new_counts)
            
            # Accept if better or with some probability (annealing)
            delta_peak = new_peak - curr_peak
            delta_ss = new_ss - curr_ss
            
            if delta_peak < 0 or (delta_peak == 0 and delta_ss < 0):
                # Definitely accept
                curr_A.remove(x)
                curr_counts = new_counts
                curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                improved = True
                if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                    best_peak = curr_peak
                    best_ss = curr_ss
                    best_A = set(curr_A)
                break
            else:
                # Accept with probability exp(-delta/temperature) if not too bad
                delta = delta_peak * 10 + delta_ss
                if delta > 0 and random.random() < np.exp(-delta / max(temperature, 0.01)):
                    curr_A.remove(x)
                    curr_counts = new_counts
                    curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                    improved = True
                    break
        
        if improved:
            continue
        
        # Phase 2: Try swaps
        if len(curr_A) > 1:
            x_out = random.choice(list(curr_A))
            candidates = [i for i in range(n + 1) if i not in curr_A]
            if candidates:
                x_in = random.choice(candidates)
                new_counts = try_swap(curr_A, curr_counts, x_out, x_in)
                if new_counts is not None:
                    new_peak, new_unc, new_ss = get_metrics(new_counts)
                    delta_peak = new_peak - curr_peak
                    delta_ss = new_ss - curr_ss
                    
                    if delta_peak < 0 or (delta_peak == 0 and delta_ss < 0):
                        curr_A.remove(x_out)
                        curr_A.add(x_in)
                        curr_counts = new_counts
                        curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                        improved = True
                        if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                            best_peak = curr_peak
                            best_ss = curr_ss
                            best_A = set(curr_A)
                    elif delta_peak == 0 and delta_ss >= 0:
                        delta = delta_ss
                        if random.random() < np.exp(-delta / max(temperature, 0.01)):
                            curr_A.remove(x_out)
                            curr_A.add(x_in)
                            curr_counts = new_counts
                            curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                            improved = True
        
        if improved:
            temperature *= cooling
            continue
        
        # Phase 3: 2-swap perturbation
        if len(curr_A) > 2:
            x1, x2 = random.sample(list(curr_A), 2)
            candidates = [i for i in range(n + 1) if i not in curr_A]
            if len(candidates) >= 2:
                y1, y2 = random.sample(candidates, 2)
                
                A_trial = set(curr_A)
                A_trial.remove(x1)
                A_trial.remove(x2)
                A_trial.add(y1)
                A_trial.add(y2)
                
                new_counts = compute_counts(A_trial)
                new_peak, new_unc, new_ss = get_metrics(new_counts)
                
                if new_unc == 0:
                    delta_peak = new_peak - curr_peak
                    delta_ss = new_ss - curr_ss
                    
                    if delta_peak < 0 or (delta_peak == 0 and delta_ss < 0):
                        curr_A = A_trial
                        curr_counts = new_counts
                        curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                        improved = True
                        if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                            best_peak = curr_peak
                            best_ss = curr_ss
                            best_A = set(curr_A)
                    elif random.random() < np.exp(-(delta_peak * 10 + delta_ss) / max(temperature, 0.01)):
                        curr_A = A_trial
                        curr_counts = new_counts
                        curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                        improved = True
        
        if improved:
            temperature *= cooling
            continue
        
        # Stagnation: cool down and try restart from best
        temperature *= 0.9
        if temperature < 0.05:
            # Restart from best solution
            curr_A = set(best_A)
            curr_counts = compute_counts(curr_A)
            curr_peak, curr_unc, curr_ss = get_metrics(curr_counts)
            temperature = 1.0
    
    return sorted(list(best_A))
