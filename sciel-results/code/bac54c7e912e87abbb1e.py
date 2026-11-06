import numpy as np
import time
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # 1. Precompute all arithmetic progressions
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # 2. Build incidence matrix M (n, Np)
    M = np.zeros((n, Np), dtype=np.int8)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1
    
    # 3. Continuous Optimization (Subgradient Descent on Surrogate)
    # Surrogate: f(x) = sum_p (sum_i in p x_i)^2 = x^T (M M^T) x
    # We want to find a continuous vector x close to ±1 that minimizes this.
    # Then project to signs.
    
    rng = np.random.RandomState(42)
    best_signs = None
    best_score = 1000000
    
    def compute_sums(signs):
        return signs @ M
    
    def max_abs_sum(sums):
        return int(np.max(np.abs(sums)))
    
    # Initialize continuous vector
    # Start with random signs as a good initial basin for the continuous space
    x = rng.choice([-1.0, 1.0], size=n)
    
    # Precompute M M^T? No, it's n x n (256x256). That's small.
    # But M is (256, ~100k). M M^T is sum of outer products.
    # Actually, gradient of sum_p (M_p^T x)^2 is 2 * M (M^T x) = 2 * M sums.
    # So we don't need to form M M^T explicitly, just compute sums = M^T x, then grad = M sums.
    
    # Subgradient descent
    lr = 0.05
    for step in range(2000):
        if time.time() > deadline - 30:
            break
            
        sums = M.T @ x
        # Gradient of sum(sums**2) is 2 * M @ sums
        grad = M @ sums
        
        # Update x
        x -= lr * grad
        
        # Periodically project and check
        if step % 50 == 0:
            signs = np.sign(x)
            # Handle zeros (though unlikely with random init and updates)
            signs[signs == 0] = 1.0
            
            s_sums = compute_sums(signs)
            score = max_abs_sum(s_sums)
            
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
                
    # Ensure we have a valid best_signs
    if best_signs is None:
        signs = np.sign(x)
        signs[signs == 0] = 1.0
        best_signs = signs
        best_score = max_abs_sum(compute_sums(best_signs))

    # 4. Local Search with Histogram Optimization
    # Current best is a good starting point for local search.
    # The key insight from "inspiration" and "knowledge" is that histogram-based
    # max tracking allows O(1) (or very fast) updates.
    
    signs = best_signs.astype(np.int32)
    sums = compute_sums(signs).astype(np.int32)
    
    # Histogram of sums
    # Max possible sum is n (256). Min is -256.
    # Range: [-256, 256]. Size: 513.
    # We need to track counts of sums in this range.
    # However, sums can be large? No, max sum of 256 items is 256.
    # So range [-256, 256] is sufficient.
    
    OFFSET = 256
    HIST_SIZE = 513
    
    def build_histogram(sums):
        hist = np.zeros(HIST_SIZE, dtype=np.int32)
        # Clip sums to valid range just in case, though they should be in range
        idxs = np.clip(sums + OFFSET, 0, HIST_SIZE - 1)
        np.add.at(hist, idxs, 1)
        return hist
    
    hist = build_histogram(sums)
    
    # Helper to get current max abs sum from histogram
    def get_max_abs_from_hist(hist):
        # We want the max |s| such that hist[s+OFFSET] > 0
        # We can maintain this incrementally, but for simplicity in the first pass
        # or if we just recalculate occasionally, we can do:
        # But we want O(1) or O(log N) updates.
        # Let's maintain current_max_abs.
        # If we remove the only entry at current_max_abs, we need to find the next one.
        # This can be slow if we scan.
        # Alternative: maintain a set of active magnitudes?
        # Or simply: since Np is ~100k, and we do many steps, 
        # maybe we should just use a different trick.
        
        # Actually, let's look at the "Histogram-based max tracking" claim.
        # It usually implies maintaining a pointer to the current max.
        # If the count at the current max drops to 0, we scan down.
        # Since sums are bounded by [-256, 256], the scan is at most 256 steps.
        # This is effectively O(1) compared to O(Np) scan of all sums.
        
        pass # We'll implement the incremental logic in the loop
        
    current_max_abs = int(np.max(np.abs(sums)))
    
    # We need to know the count at the current max_abs and -current_max_abs
    # Actually, we just need to know if there are any sums with absolute value == current_max_abs.
    # If we remove one and it was the last one, we need to find the new max.
    
    def update_max_abs(hist, old_val, new_val, current_max):
        # This is tricky because we are changing one sum.
        # Let's just recalculate max if current_max becomes 0?
        # No, let's maintain a structure.
        
        # Simpler approach for local search:
        # Since the range is small, we can maintain the histogram.
        # To get the new max, if the count at current_max (and -current_max) becomes 0,
        # we scan down.
        
        pass
        
    # Let's implement a robust local search step:
    # 1. Pick random i
    # 2. Determine old_sum_p for each p containing i? No, that's O(Np) to find which p contain i.
    #    Wait, M[i] gives the mask.
    #    The sums that change are those where M[i, p] == 1.
    #    Number of such p is the number of APs containing i.
    #    On average, how many APs contain a given index?
    #    Total APs ~ sum_{a,s} (n-a)/s ~ n * sum 1/s ~ n log n ~ 256 * 6 ~ 1500?
    #    Actually, let's count.
    #    For a fixed i, how many (a, s) such that i is in {a, a+s, ...}?
    #    i = a + k*s.
    #    This is still a significant number.
    
    # Let's estimate the cost of updating the histogram.
    # For a flip at i:
    # For each p where M[i,p]==1:
    #   old_s = sums[p]
    #   new_s = old_s + 2 * signs[i]  (since sign flips, delta is -2*old_sign, but we are computing new sum)
    #   Actually: new_sum = old_sum - 2 * old_sign_i * M[i,p]
    #   Since M[i,p] is 1, new_sum = old_sum - 2 * old_sign_i
    #   Update hist: hist[old_s]--, hist[new_s]++
    #   Update sums[p] = new_s
    
    # Cost is proportional to degree(i) in the hypergraph.
    
    # Let's precompute the list of progression indices for each position i to speed up iteration.
    # pos_to_progs[i] = list of p such that M[i,p] == 1
    pos_to_progs = [[] for _ in range(n)]
    for i in range(n):
        # Find non-zeros in M[i]
        nz = np.nonzero(M[i])[0]
        pos_to_progs[i] = nz
    
    # Local Search Loop
    # We use the histogram to track max_abs efficiently.
    # We maintain `current_max_abs`.
    # When we update sums, we update hist.
    # After updates, we check if the count at `current_max_abs` (and `-current_max_abs`) is 0.
    # If so, we scan down to find the new max.
    
    T = 2.0
    cooling = 0.999
    
    # Initialize hist
    hist = build_histogram(sums)
    current_max_abs = int(np.max(np.abs(sums)))
    
    # Helper to adjust max_abs
    def adjust_max_abs(hist, current_max):
        # Check if current_max is still valid
        # We need to check both +current_max and -current_max
        if hist[current_max + OFFSET] > 0 or (current_max > 0 and hist[-current_max + OFFSET] > 0):
            return current_max
        # Otherwise, scan down
        # Max possible is 256
        for m in range(current_max - 1, -1, -1):
            if hist[m + OFFSET] > 0 or (m > 0 and hist[-m + OFFSET] > 0):
                return m
        return 0

    for step in range(50000): # Limit steps to ensure time
        if time.time() > deadline - 2:
            break
            
        i = rng.randint(n)
        old_sign = signs[i]
        new_sign = -old_sign
        
        progs_idx = pos_to_progs[i]
        
        # Calculate delta for max_abs
        # We need to know the new max_abs to decide acceptance.
        # We can simulate the changes on the histogram?
        # Or just apply, check, and revert if rejected?
        # Applying is O(degree(i)). Reverting is O(degree(i)).
        # Checking max is O(1) amortized.
        
        old_sums_backup = []
        for p in progs_idx:
            old_s = sums[p]
            new_s = old_s - 2 * old_sign # Since M[i,p]=1
            old_sums_backup.append((p, old_s, new_s))
            
        # Apply changes to hist and sums
        for p, old_s, new_s in old_sums_backup:
            hist[old_s + OFFSET] -= 1
            hist[new_s + OFFSET] += 1
            sums[p] = new_s
            
        # Get new max
        new_max_abs = adjust_max_abs(hist, current_max_abs)
        
        # Acceptance criterion
        # Standard SA: accept if new_max < current_max, or with prob exp(-(new_max - current_max)/T)
        # Note: new_max >= current_max is not necessarily true, it can be smaller.
        # If new_max < current_max, accept.
        # If new_max > current_max, accept with prob.
        # If new_max == current_max, accept with prob 1 (or 0.5 to encourage exploration? No, 1 is fine for SA).
        
        delta = new_max_abs - current_max_abs
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = True # Flat move, helps escape local minima
        else:
            if delta < 50 * T:
                p_accept = math.exp(-delta / T)
                if rng.random_sample() < p_accept:
                    accept = True
            
        if accept:
            signs[i] = new_sign
            current_max_abs = new_max_abs
            if current_max_abs < best_score:
                best_score = current_max_abs
                best_signs = signs.copy()
        else:
            # Revert
            for p, old_s, new_s in old_sums_backup:
                hist[new_s + OFFSET] -= 1
                hist[old_s + OFFSET] += 1
                sums[p] = old_s
            # current_max_abs remains unchanged
            
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    # Final Polish: Hill Climbing
    signs = best_signs.astype(np.int32)
    sums = compute_sums(signs).astype(np.int32)
    hist = build_histogram(sums)
    current_max_abs = int(np.max(np.abs(sums)))
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for i in range(n):
            if time.time() > deadline - 1:
                break
                
            old_sign = signs[i]
            new_sign = -old_sign
            progs_idx = pos_to_progs[i]
            
            old_sums_backup = []
            for p in progs_idx:
                old_s = sums[p]
                new_s = old_s - 2 * old_sign
                old_sums_backup.append((p, old_s, new_s))
                
            for p, old_s, new_s in old_sums_backup:
                hist[old_s + OFFSET] -= 1
                hist[new_s + OFFSET] += 1
                sums[p] = new_s
                
            new_max_abs = adjust_max_abs(hist, current_max_abs)
            
            if new_max_abs < current_max_abs:
                signs[i] = new_sign
                current_max_abs = new_max_abs
                improved = True
                if current_max_abs < best_score:
                    best_score = current_max_abs
                    best_signs = signs.copy()
            else:
                # Revert
                for p, old_s, new_s in old_sums_backup:
                    hist[new_s + OFFSET] -= 1
                    hist[old_s + OFFSET] += 1
                    sums[p] = old_s
                    
    return [int(x) for x in best_signs]

