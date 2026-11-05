import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    n = 128
    min_sum = 128
    max_sum = 256
    target_len = max_sum - min_sum + 1  # 129

    # Precompute diagonal mask for unordered count conversion
    # diag[i] corresponds to sum s = min_sum + i
    # diag is 1 if s is even and s/2 is a valid index (which it always is for s in [128, 256] since 64 <= s/2 <= 128)
    # But we need to check if A[s/2] is 1. So we can't precompute the value, only the index.
    diag_indices = []
    for i in range(target_len):
        s = min_sum + i
        if s % 2 == 0:
            diag_indices.append((i, s // 2))
        else:
            diag_indices.append((i, -1)) # -1 indicates no diagonal contribution

    def get_peak_and_valid(A_bin):
        """
        A_bin: np.array of 0/1, length 129.
        Returns: (peak, is_valid)
        peak: max multiplicity in [128, 256]
        is_valid: True if all counts > 0
        """
        # Convolution gives ordered pair counts
        conv = np.convolve(A_bin, A_bin, mode='full')
        
        # Extract target range
        counts_ordered = conv[min_sum:max_sum + 1]
        
        # Convert to unordered
        # U[s] = (C[s] + D[s]) / 2
        # D[s] is 1 if s is even and A[s/2] == 1
        diag_vals = np.zeros(target_len, dtype=np.int8)
        for i, mid in diag_indices:
            if mid != -1:
                diag_vals[i] = A_bin[mid]
                
        counts_unordered = (counts_ordered.astype(np.float64) + diag_vals) / 2.0
        counts_unordered = np.round(counts_unordered).astype(np.int32)
        
        if np.any(counts_unordered == 0):
            return 0, False
        
        peak = np.max(counts_unordered)
        return int(peak), True

    def get_counts(A_bin):
        """Returns the unordered count vector for [128, 256]"""
        conv = np.convolve(A_bin, A_bin, mode='full')
        counts_ordered = conv[min_sum:max_sum + 1]
        diag_vals = np.zeros(target_len, dtype=np.int8)
        for i, mid in diag_indices:
            if mid != -1:
                diag_vals[i] = A_bin[mid]
        counts_unordered = (counts_ordered.astype(np.float64) + diag_vals) / 2.0
        return np.round(counts_unordered).astype(np.int32)

    best_peak = float('inf')
    best_A_bin = None

    # Multi-start Simulated Annealing
    num_restarts = 8
    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialization
        # Strategy: Start with a set that is likely to be valid.
        # A random set of size ~64-70 usually covers [128, 256] if distributed well.
        # Or start with a known valid set and perturb.
        # Let's try a mix:
        if restart % 2 == 0:
            # Random
            size = rng.integers(55, 75)
            indices = rng.choice(n + 1, size=size, replace=False)
            A_bin = np.zeros(n + 1, dtype=np.int8)
            A_bin[indices] = 1
        else:
            # Structured: Two intervals
            # e.g., [0, 40] U [80, 128]
            a = rng.integers(20, 50)
            b = rng.integers(70, 100)
            A_bin = np.zeros(n + 1, dtype=np.int8)
            A_bin[0:a+1] = 1
            A_bin[b:129] = 1

        # Ensure validity with greedy repair
        def repair_validity(A_bin):
            """Greedy addition to ensure coverage"""
            A_bin = A_bin.copy()
            for _ in range(100):
                counts = get_counts(A_bin)
                if np.all(counts > 0):
                    break
                missing = np.where(counts == 0)[0]
                # Find element that covers most missing sums
                best_elem = -1
                best_cover = 0
                for x in range(n + 1):
                    if A_bin[x] == 1:
                        continue
                    # How many missing sums does x help cover?
                    # Sum s is covered by x if s-x in A and x <= s-x (or just pair exists)
                    # Actually, if we add x, new pairs are (x, y) for y in A.
                    # Sums are x + y.
                    # We want to cover s in missing.
                    # So we need y = s - x in A.
                    cover = 0
                    for s_idx in missing:
                        s = min_sum + s_idx
                        y = s - x
                        if 0 <= y <= n and A_bin[y] == 1:
                            # Check if this pair is valid (unordered)
                            # If x <= y, it's a new pair (x,y).
                            # If x > y, the pair is (y,x), which was already counted if y was in A?
                            # No, if x was NOT in A, the pair (y,x) did not exist.
                            # So yes, adding x creates pair {x,y}.
                            cover += 1
                    if cover > best_cover:
                        best_cover = cover
                        best_elem = x
                if best_elem == -1 or best_cover == 0:
                    break
                A_bin[best_elem] = 1
            return A_bin

        A_bin = repair_validity(A_bin)
        
        peak, is_valid = get_peak_and_valid(A_bin)
        if is_valid and peak < best_peak:
            best_peak = peak
            best_A_bin = A_bin.copy()
            
        if not is_valid:
            # If still invalid, skip to next restart or force a full range
            # Full range is always valid
            A_bin = np.ones(n + 1, dtype=np.int8)
            peak, _ = get_peak_and_valid(A_bin)
            if peak < best_peak:
                best_peak = peak
                best_A_bin = A_bin.copy()
            continue

        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling = 0.95
        steps_per_temp = 500
        
        current_peak = peak
        current_A = A_bin.copy()
        
        # Pre-compute list of all indices for random selection
        all_indices = np.arange(n + 1)
        
        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 5:
                    break
                    
                # Pick random index to flip
                idx = rng.choice(all_indices)
                
                # Flip
                current_A[idx] = 1 - current_A[idx]
                
                new_peak, new_valid = get_peak_and_valid(current_A)
                
                if not new_valid:
                    # Penalty for invalidity
                    # Count missing sums
                    counts = get_counts(current_A)
                    missing_count = np.sum(counts == 0)
                    new_cost = 1000 + missing_count + new_peak
                else:
                    new_cost = new_peak
                    
                if current_A[idx] == 0: # We just removed it
                    old_A_temp = current_A.copy()
                    old_A_temp[idx] = 1
                    old_peak, old_valid = get_peak_and_valid(old_A_temp)
                    if not old_valid:
                        counts = get_counts(old_A_temp)
                        missing_count = np.sum(counts == 0)
                        old_cost = 1000 + missing_count + old_peak
                    else:
                        old_cost = old_peak
                else: # We just added it
                    old_A_temp = current_A.copy()
                    old_A_temp[idx] = 0
                    old_peak, old_valid = get_peak_and_valid(old_A_temp)
                    if not old_valid:
                        counts = get_counts(old_A_temp)
                        missing_count = np.sum(counts == 0)
                        old_cost = 1000 + missing_count + old_peak
                    else:
                        old_cost = old_peak

                delta = new_cost - old_cost
                
                if delta < 0:
                    # Accept
                    current_peak = new_peak if new_valid else current_peak # peak only tracked if valid
                    if new_valid and new_peak < best_peak:
                        best_peak = new_peak
                        best_A_bin = current_A.copy()
                else:
                    # Accept with probability exp(-delta/T)
                    if rng.random() < np.exp(-delta / T):
                        current_peak = new_peak if new_valid else current_peak
                        if new_valid and new_peak < best_peak:
                            best_peak = new_peak
                            best_A_bin = current_A.copy()
                    else:
                        # Reject: revert
                        current_A[idx] = 1 - current_A[idx]
                        
            T *= cooling
            
        # Final Hill Climbing on the best found in this restart
        # We will just run a simple hill climb on the current best global solution at the end
        
    # Final Polish: Deterministic Hill Climbing on best_A_bin
    if best_A_bin is not None:
        A_bin = best_A_bin.copy()
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_peak, _ = get_peak_and_valid(A_bin)
            
            # Try removing elements
            for i in range(n + 1):
                if A_bin[i] == 1:
                    A_bin[i] = 0
                    new_peak, new_valid = get_peak_and_valid(A_bin)
                    if new_valid and new_peak < current_peak:
                        current_peak = new_peak
                        improved = True
                        # Don't break, keep removing if possible? 
                        # Or break to re-evaluate all? 
                        # Let's break to restart the loop to be safe and simple
                        break
                    else:
                        A_bin[i] = 1 # Revert
            
            if improved:
                continue
                
            # Try adding elements
            for i in range(n + 1):
                if A_bin[i] == 0:
                    A_bin[i] = 1
                    new_peak, new_valid = get_peak_and_valid(A_bin)
                    if new_valid and new_peak < current_peak:
                        current_peak = new_peak
                        improved = True
                        break
                    else:
                        A_bin[i] = 0 # Revert
                        
            if improved:
                # Update global best
                best_peak = current_peak
                best_A_bin = A_bin.copy()
                
    # Convert back to list of integers
    final_A = np.where(best_A_bin == 1)[0].tolist()
    if not final_A:
        final_A = list(range(0, 129))
        
    return final_A
