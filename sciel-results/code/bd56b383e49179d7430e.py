import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    # Precompute diagonal mask for targets [128, 256]
    # diag_mask[i] corresponds to target t = 128 + i
    # diag is 1 if t is even and t//2 <= 128
    diag_mask = np.zeros(num_targets, dtype=np.int32)
    for i in range(num_targets):
        t = target_lo + i
        if t % 2 == 0:
            half_t = t // 2
            if 0 <= half_t <= n:
                diag_mask[i] = 1
    
    def compute_counts(freq):
        """
        Compute unordered representation counts for targets [128, 256]
        given a binary frequency array freq of length 129.
        """
        # Ordered pairs convolution
        conv = np.convolve(freq, freq, mode='full')
        # Extract relevant slice
        # conv[t] is the count of ordered pairs summing to t
        target_counts_ordered = conv[target_lo:target_hi+1]
        
        # Calculate diagonal contributions
        # D(t) = 1 if t is even and freq[t//2] == 1
        # We can vectorize this:
        # Indices in freq corresponding to t//2 for t in [128, 256]
        # t = 128 + i
        # t//2 = 64 + i//2
        # We need to be careful with the index mapping.
        
        # Let's just loop for clarity and speed is sufficient for 129 items
        diag_vals = np.zeros(num_targets, dtype=np.int32)
        for i in range(num_targets):
            t = target_lo + i
            if t % 2 == 0:
                idx = t // 2
                if idx <= n:
                    diag_vals[i] = freq[idx]
        
        # Unordered count = (Ordered + Diagonal) // 2
        # Note: Ordered is always even? No.
        # If a != b, ordered contributes 2. If a=b, ordered contributes 1.
        # So Ordered = 2 * (pairs with a!=b) + 1 * (pairs with a=b)
        # Diagonal = (pairs with a=b)
        # Ordered + Diagonal = 2 * (pairs with a!=b) + 2 * (pairs with a=b) = 2 * Total Unordered
        # So division by 2 is exact.
        
        counts = (target_counts_ordered + diag_vals) // 2
        return counts

    def is_complete(counts):
        return bool(np.all(counts > 0))
    
    def peak_of(counts):
        return int(np.max(counts))

    def local_search(A_init, max_iter=100000):
        """
        Local search using convolution for evaluation.
        """
        A = sorted(set(A_init))
        if not A:
            return A, float('inf'), np.zeros(num_targets, dtype=np.int32)
            
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            freq[x] = 1
            
        counts = compute_counts(freq)
        
        if not is_complete(counts):
            # If initial set is not complete, try to complete it greedily or return invalid
            # For now, return as is, caller will check completeness
            return A, peak_of(counts), counts
        
        cur_peak = peak_of(counts)
        A_set = set(A)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try Removals
            # Removing an element reduces the set size, potentially lowering peak.
            # We check all elements.
            for x in A:
                freq[x] = 0
                new_counts = compute_counts(freq)
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A_set.discard(x)
                        A = [a for a in A if a != x]
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
                    else:
                        freq[x] = 1
                else:
                    freq[x] = 1
            
            if improved:
                continue
            
            if time.time() > deadline - 5:
                break
            
            # 2. Try Additions
            # Adding an element might cover missing targets or balance peaks.
            # We check all non-elements.
            for x in range(n + 1):
                if x in A_set:
                    continue
                freq[x] = 1
                new_counts = compute_counts(freq)
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A_set.add(x)
                        A.append(x)
                        A.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
                    else:
                        freq[x] = 0
                else:
                    freq[x] = 0
            
            if improved:
                continue
            
            if time.time() > deadline - 5:
                break
            
            # 3. Try Swaps
            # Swap x in A with y not in A.
            # Sample a few swaps to avoid O(|A| * (N-|A|)) which is too slow.
            found_swap = False
            num_swaps_to_try = min(20, len(A))
            for _ in range(num_swaps_to_try):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, len(A) - 1)
                x_out = A[idx]
                x_in = random.randint(0, n)
                if x_in in A_set or x_in == x_out:
                    continue
                
                freq[x_out] = 0
                freq[x_in] = 1
                new_counts = compute_counts(freq)
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A_set.discard(x_out)
                        A_set.add(x_in)
                        A[idx] = x_in
                        A.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        found_swap = True
                        break
                    else:
                        freq[x_out] = 1
                        freq[x_in] = 0
                else:
                    freq[x_out] = 1
                    freq[x_in] = 0
            
            if not found_swap and not improved:
                # Local optimum reached for this configuration
                break
        
        return A, cur_peak, counts

    # Initializations
    initializations = []
    
    # 1. Dense middle: [64, 128]
    initializations.append(list(range(64, 129)))
    
    # 2. Symmetric: [0, 64] U [64, 128]
    initializations.append(list(range(0, 129, 2)))
    
    # 3. Random sets of various sizes
    random.seed(42)
    for k in [50, 60, 70, 80]:
        A_rand = sorted(random.sample(range(n + 1), k))
        initializations.append(A_rand)
        
    # 4. Structured: Every 3rd number
    initializations.append(list(range(0, 129, 3)))
    
    # 5. Structured: Low and High clusters
    A_low_high = list(range(0, 32)) + list(range(96, 129))
    initializations.append(A_low_high)
    
    best_A = None
    best_peak = float('inf')
    best_counts = None
    
    for init_A in initializations:
        if time.time() > deadline - 10:
            break
        
        A_ls, peak_ls, counts_ls = local_search(init_A)
        
        if is_complete(counts_ls) and peak_ls < best_peak:
            best_A = A_ls
            best_peak = peak_ls
            best_counts = counts_ls
    
    # If we haven't found a good solution, try more random restarts with smaller sets
    # Smaller sets tend to have lower peak multiplicity if they cover the range efficiently
    if best_peak > 4:
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            k = random.randint(40, 60)
            A_rand = sorted(random.sample(range(n + 1), k))
            A_ls, peak_ls, counts_ls = local_search(A_rand, max_iter=50000)
            if is_complete(counts_ls) and peak_ls < best_peak:
                best_A = A_ls
                best_peak = peak_ls
                best_counts = counts_ls

    # Final validation
    if best_A is None or not is_complete(best_counts):
        # Fallback to a known valid solution (dense middle)
        fallback_A = list(range(64, 129))
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in fallback_A:
            freq[x] = 1
        counts = compute_counts(freq)
        if is_complete(counts):
            return fallback_A
        # If even fallback fails (should not happen), return empty or single element? 
        # Task requires non-empty subset covering [128, 256].
        # [64, 128] definitely covers [128, 256] with max sum 256 and min sum 128.
        return fallback_A
    
    return list(best_A)
