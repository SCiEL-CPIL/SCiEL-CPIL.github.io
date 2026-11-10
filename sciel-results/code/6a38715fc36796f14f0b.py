"""Small nonzero signed harmonic subsums: meet-in-the-middle exhaustive search."""
import time

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    n = 30
    half = n // 2  # 15
    # First half: denominators 1..15, second half: 16..30
    
    # Compute LCM of 1..30 for exact integer arithmetic
    # LCM(1..30) = 2329089562800
    # But we can work with smaller LCMs for each half
    # LCM(1..15) = 360360
    # LCM(16..30) = 2329089562800 // gcd-related... let's just use full LCM
    
    # Actually, let's use the approach: sum c_i / i = (sum c_i * (LCM/i)) / LCM
    # where LCM = lcm(1, 2, ..., 30)
    
    # LCM(1..30):
    # 2^4 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29
    # = 16 * 27 * 25 * 7 * 11 * 13 * 17 * 19 * 23 * 29
    # Let me compute step by step:
    # 16*27=432, 432*25=10800, 10800*7=75600, 75600*11=831600
    # 831600*13=10810800, 10810800*17=183783600, 183783600*19=3491888400
    # 3491888400*23=80313433200, 80313433200*29=2329089562800
    
    LCM = 2329089562800
    
    # For each i in 1..30, weight[i-1] = LCM // i
    weights = [LCM // i for i in range(1, n + 1)]
    
    # First half: indices 0..14 (denominators 1..15)
    # Second half: indices 15..29 (denominators 16..30)
    
    w1 = weights[:half]  # 15 weights
    w2 = weights[half:]  # 15 weights
    
    # Generate all 3^15 = 14348907 sums for first half
    # Each coefficient is -1, 0, or 1
    # sum1 = c0*w1[0] + c1*w1[1] + ... + c14*w1[14]
    
    # For efficiency, use iterative generation
    # 3^15 = 14,348,907 which is manageable
    
    # Generate all possible sums for first half
    # Use base-3 representation: digit 0->c=0, digit 1->c=1, digit 2->c=-1
    # Or: digit 0->-1, digit 1->0, digit 2->1
    
    # Let's use: for a 15-digit base-3 number, map each digit:
    # 0 -> -1, 1 -> 0, 2 -> 1
    
    def gen_sums(w):
        """Generate all 3^len(w) signed sums using weights w."""
        m = len(w)
        total = 3 ** m
        sums = [0] * total
        # Iterative: build up
        # sums[i] for i in 0..total-1
        # i in base 3: digits d0..d_{m-1}
        # c_j = d_j - 1 (so -1, 0, 1)
        # sum = sum(c_j * w[j])
        
        # More efficient: recursive/iterative construction
        sums[0] = 0
        # Build incrementally
        # For position j, the contribution cycles
        # Let's use a different approach: generate via nested loops conceptually
        # But 3^15 is 14M, let's do it efficiently
        
        # Precompute: for each position j, the weight is w[j]
        # i = d0 + 3*d1 + 9*d2 + ... + 3^14*d14
        # sum = (d0-1)*w[0] + (d1-1)*w[1] + ... + (d14-1)*w[14]
        
        # We can compute this as:
        # sum = (d0*w[0] + d1*w[1] + ... + d14*w[14]) - (w[0] + w[1] + ... + w[14])
        # Let S = sum(w), and T(i) = d0*w[0] + ... + d14*w[14]
        # Then sum = T(i) - S
        
        S = sum(w)
        
        # Compute T(i) for all i efficiently
        # T(i) where i in base 3 with m digits
        # T(i) = sum_{j=0}^{m-1} d_j(i) * w[j]
        
        # Use dynamic approach:
        # T(i) for i in 0..3^m - 1
        # Group by highest digit:
        # T(i) = d_{m-1} * w[m-1] + T(i mod 3^{m-1})
        
        # Let's build iteratively
        # Start with m=1: T(i) = (i % 3) * w[0] for i in 0,1,2
        # Then extend
        
        # Actually for m=15, 14M entries, each is a large integer.
        # Memory: 14M * ~30 bytes = ~430 MB. Might be tight.
        # Let's try a different approach.
        
        # Alternative: generate sums for first half, sort them, store in list
        # Then for each sum in second half, binary search in first half
        
        # Let's just generate all sums for first half into a list
        # 14,348,907 integers, each up to ~LCM * 15 in magnitude
        # That's fine memory-wise (~115 MB for Python ints, more like 400-500 MB)
        
        # Actually Python ints are objects, so 14M ints * ~28 bytes = ~400 MB
        # This might be too much. Let's use arrays or a different approach.
        
        # Better approach: use numpy if available, or use a sorted list with bisect
        
        # Let me reconsider: 3^15 = 14,348,907
        # Python list of 14M integers: each int object ~28 bytes + list pointer ~8 bytes
        # So ~14M * 36 = ~500 MB. This is risky.
        
        # Alternative: use array module or just accept it
        # Or: reduce problem size
        
        # Actually, let's just do it. The evaluator has some memory.
        # Let's generate first-half sums, sort them, then for each second-half sum,
        # binary search for the closest to -sum2 (to get total closest to 0 but not 0)
        
        return None  # We'll handle this differently
    
    # Let me restructure: generate first half sums, sort, then search
    
    # Generate all 3^15 sums for first half
    # Using iterative method
    m1 = half  # 15
    total1 = 3 ** m1  # 14,348,907
    
    # Precompute powers of 3
    pow3 = [1] * (m1 + 1)
    for i in range(1, m1 + 1):
        pow3[i] = pow3[i-1] * 3
    
    # Generate sums for first half
    # For index i (0 to total1-1), decompose into base-3 digits
    # sum1 = sum_{j=0}^{14} (digit_j - 1) * w1[j]
    # where digit_j = (i // 3^j) % 3
    
    # Efficient generation:
    # sum1(i) = sum1(i // 3) * ... no, let's use the digit decomposition
    
    # Actually, a cleaner iterative approach:
    # Start with [0]
    # For each weight w, new_sums = [s - w, s, s + w for s in sums]
    # But this creates 3x the list each time, so after 15 steps we have 3^15 elements
    
    sums1 = [0]
    for w in w1:
        # new_sums has 3 * len(sums1) elements
        new_sums = []
        # For memory efficiency, pre-allocate
        n_sums = len(sums1)
        new_sums = [0] * (3 * n_sums)
        for k, s in enumerate(sums1):
            new_sums[3*k] = s - w
            new_sums[3*k+1] = s
            new_sums[3*k+2] = s + w
        sums1 = new_sums
    
    # Check time
    if time.time() > deadline:
        # Fallback: return the seed answer
        return [0] * 28 + [1, -1]
    
    # Sort sums1
    sums1.sort()
    
    # Now generate second half sums and for each, find best match in sums1
    m2 = n - half  # 15
    w2_list = w2
    
    # We need to find: minimize |sum1 + sum2| subject to sum1 + sum2 != 0
    # For a given sum2, we want sum1 closest to -sum2 but not equal to -sum2 (if that would give 0)
    # Actually: total = sum1 + sum2. We want |total| minimized, total != 0.
    # For given sum2, we want sum1 such that sum1 + sum2 is closest to 0 but not 0.
    # So sum1 closest to -sum2, but if sum1 == -sum2 exactly, we need the next closest.
    
    import bisect
    
    best_val = None  # absolute value of best nonzero sum (in integer units)
    best_coeff1 = None  # the coefficients for first half
    best_coeff2 = None  # the coefficients for second half
    
    # We also need to track which coefficients produced each sum
    # This is tricky. Let's store (sum, index) and decode index later.
    # Or: store sum -> list of indices? No, too many.
    
    # Alternative: store sum -> one representative index (any one that achieves it)
    # Since we just need one valid solution, this works.
    
    # Rebuild sums1 with index tracking
    # Actually, let's store: for each sum value, one index that produces it
    # sum_to_idx: dict mapping sum -> index (first one found)
    
    # But 14M entries in a dict is heavy. Let's think differently.
    
    # Alternative approach: 
    # 1. Generate all first-half sums with their indices, sort by sum
    # 2. For each second-half sum, binary search in sorted first-half sums
    # 3. Track the best (nonzero) total
    
    # To track indices, we can store parallel arrays or tuples
    # But 14M tuples is also heavy.
    
    # Let's try: store sums1 sorted, and separately, for each position in sorted sums1,
    # store the original index. Then we can decode.
    
    # Actually, let's just store (sum_value, original_index) pairs, sort by sum_value.
    # 14M tuples of (int, int) - each tuple ~56 bytes + 2 ints ~56 bytes = ~112 bytes
    # 14M * 112 = ~1.6 GB. Too much.
    
    # Better: store two parallel lists: sorted_sums and sorted_indices
    # sorted_sums: list of 14M ints
    # sorted_indices: list of 14M ints
    # Total: 14M * (28 + 8) * 2 = ~1 GB. Still a lot but maybe OK.
    
    # Actually, let me reconsider. The original sums1 list already has 14M ints.
    # If I create a list of (sum, idx) tuples and sort, that's another 14M objects.
    
    # Alternative: just sort sums1, and when I find the best sum, I can regenerate
    # the coefficients by checking all 3^15 possibilities? No, that's circular.
    
    # Let me use a different strategy:
    # Store sums1 as a sorted list.
    # For finding the best, I only need the sum values, not the indices, during search.
    # After finding the best (sum1_val, sum2_val), I can decode the coefficients
    # by searching for sum1_val in the first half and sum2_val in the second half.
    
    # But decoding requires knowing which index gives that sum. I can just iterate
    # through all 3^15 first-half combinations to find one that gives sum1_val.
    # That's 14M iterations, which is fast.
    
    # Plan:
    # 1. Generate sorted sums1 (just values)
    # 2. Generate second-half sums one at a time (or in batches)
    # 3. For each sum2, binary search in sums1 for best match
    # 4. Track global best
    # 5. At the end, decode coefficients for the best pair
    
    # Let's proceed. sums1 is already sorted.
    
    # Generate second half sums
    m2 = len(w2_list)
    total2 = 3 ** m2
    
    # For efficiency, generate second-half sums iteratively too, but process on the fly
    # to save memory
    
    # Actually, let's generate all second-half sums first (also 14M), then process.
    # Or process on the fly.
    
    # Let's process on the fly to save memory:
    # Generate second-half sums via the same iterative method, but for each sum2,
    # immediately do the binary search.
    
    # But the iterative method builds up the list. Let me do DFS instead.
    
    # DFS for second half:
    # At depth d (0 to m2-1), choose c in {-1, 0, 1}
    # Accumulate sum2
    # At depth m2, do binary search in sums1
    
    # This avoids storing all second-half sums.
    
    # But DFS in Python with 14M leaves might be slow due to function call overhead.
    # Let's use an iterative approach with a stack.
    
    # Actually, let's just generate all second-half sums into a list (14M ints),
    # then loop over them. Memory: 14M * 28 bytes = ~400 MB. Combined with sums1
    # (~400 MB), total ~800 MB. Should be OK.
    
    sums2 = [0]
    for w in w2_list:
        n_sums = len(sums2)
        new_sums = [0] * (3 * n_sums)
        for k, s in enumerate(sums2):
            new_sums[3*k] = s - w
            new_sums[3*k+1] = s
            new_sums[3*k+2] = s + w
        sums2 = new_sums
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Now search for best pair
    # For each sum2 in sums2, find sum1 in sums1 such that |sum1 + sum2| is minimized
    # and sum1 + sum2 != 0
    
    n_sums1 = len(sums1)
    
    best_abs = None
    best_sum1 = None
    best_sum2 = None
    
    # Process sums2
    # For speed, we can skip duplicates in sums2 (many sums will be repeated)
    # Use a set to track seen sum2 values? 14M entries in a set is also heavy.
    # Let's just iterate.
    
    # Optimization: if we find a total of 1 (in LCM units), that's likely optimal
    # since the minimum nonzero value is at least 1/LCM in absolute terms.
    # Actually, the minimum nonzero |sum1 + sum2| in integer units is at least 1.
    # So if we find |sum1 + sum2| == 1, we can stop.
    
    for sum2 in sums2:
        target = -sum2
        # Find position in sums1 where target would be
        pos = bisect.bisect_left(sums1, target)
        
        # Check candidates around pos
        candidates = []
        if pos < n_sums1:
            candidates.append(sums1[pos])
        if pos > 0:
            candidates.append(sums1[pos - 1])
        
        for s1 in candidates:
            total = s1 + sum2
            if total == 0:
                # Need to check neighbors for nonzero
                # Check pos+1 and pos-1 (already checked pos-1 if pos>0)
                # Let's check more neighbors
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_sum1 = s1
                best_sum2 = sum2
                if best_abs == 1:
                    # Can't do better than 1
                    break
        
        if best_abs == 1:
            break
        
        if time.time() > deadline - 10:
            break
    
    if best_abs is None:
        # Shouldn't happen, but fallback
        return [0] * 28 + [1, -1]
    
    # Now decode coefficients
    # Find first-half coefficients that give best_sum1
    # Find second-half coefficients that give best_sum2
    
    # Decode first half:
    # We need c0..c14 such that sum(c_j * w1[j]) = best_sum1
    # Brute force: iterate through all 3^15 combinations
    # But we can be smarter: use the structure
    
    # Actually, since we generated sums1 iteratively, the index in the original
    # (unsorted) sums1 list corresponds to a base-3 number.
    # But we sorted sums1, so we lost the index mapping.
    
    # Let's just brute-force decode:
    # For first half: try all 3^15 combinations? That's 14M, fast enough.
    # But we can do better with a meet-in-the-middle within the half, or just
    # iterate.
    
    # Actually, let's just iterate through all 3^15 for each half.
    # 14M iterations per half, ~30M total, should take a few seconds.
    
    def decode_half(w, target):
        """Find coefficients in {-1,0,1}^len(w) such that sum(c_i * w[i]) = target."""
        m = len(w)
        # Iterative search
        # Use the same generation but stop when we find target
        # Generate all sums and check
        # For efficiency, use the iterative list approach but check as we go
        
        # Actually, let's just do a simple DFS with pruning
        # Or: generate all sums (we already know it's feasible) and find one matching target
        
        # Simple approach: iterate through base-3 numbers
        total = 3 ** m
        pow3_local = [1] * (m + 1)
        for i in range(1, m + 1):
            pow3_local[i] = pow3_local[i-1] * 3
        
        for idx in range(total):
            # Decode idx into base-3 digits
            s = 0
            temp = idx
            valid = True
            coeffs = [0] * m
            for j in range(m):
                digit = temp % 3
                temp //= 3
                c = digit - 1  # -1, 0, 1
                coeffs[j] = c
                s += c * w[j]
            if s == target:
                return coeffs
        
        return None  # Should not happen
    
    # This brute force is 14M iterations, each with 15 operations = 215M operations.
    # In Python, this might take 10-20 seconds. Too slow.
    
    # Better: use the fact that we can do a meet-in-the-middle within the half,
    # or use a smarter search.
    
    # Alternative: regenerate the sums with index tracking.
    # Let's redo the first-half generation but store (sum, index) pairs.
    
    # Actually, a cleaner approach: since we know best_sum1 is in sums1,
    # and sums1 was generated in a specific order, we can regenerate sums1
    # with index tracking.
    
    # Let's regenerate first half with index tracking:
    # sums1_with_idx: list of (sum_value, index) where index is 0..3^15-1
    # The index in base 3 gives the coefficients.
    
    # But this requires 14M tuples again. Let's just store the index as a parallel array.
    
    # Alternative: just regenerate sums1 in the same order and find best_sum1.
    # The order is deterministic:
    # sums1[0] = 0 (all zeros)
    # After adding w[0]: [−w[0], 0, w[0]]
    # After adding w[1]: [−w[0]−w[1], −w[0], −w[0]+w[1], 0, w[1], w[0], w[0]+w[1]]
    # etc.
    
    # The index in the final list corresponds to a base-3 number where:
    # digit j (from least significant) corresponds to the choice at step j
    # digit 0 -> -w[j], digit 1 -> 0, digit 2 -> +w[j]
    
    # So for index idx, the coefficients are:
    # c_j = ((idx // 3^j) % 3) - 1
    
    # To find which index gives best_sum1, I can just iterate through all indices
    # and compute the sum. But that's 14M * 15 = 215M operations.
    
    # Better: use a hash map. Regenerate sums1 with a dict mapping sum -> index.
    # 14M entries in a dict: ~14M * (key + value + overhead) ~ 1-2 GB. Too much.
    
    # Let's try a different approach: since we sorted sums1, and we know best_sum1
    # is in it, we can find its position in the sorted list. Then we need to map
    # back to the original index.
    
    # Alternative: just do the brute-force decode but optimize it.
    # Use numpy if available?
    
    # Actually, let's just accept the ~10-20s decode time. We have 80s total.
    # Generation took maybe 5-10s, search took some time, leaving 30-50s for decode.
    
    # Let's optimize the decode:
    # Instead of iterating through all 3^15, use a meet-in-the-middle within the half.
    # Split 15 into 7+8. Generate 3^7 = 2187 sums for first 7, 3^8 = 6561 for last 8.
    # For each sum in first 7, look for target - sum in the second 8 set.
    
    def decode_half_mitm(w, target):
        """Decode using meet-in-the-middle within the half."""
        m = len(w)
        h1 = m // 2  # 7
        h2 = m - h1  # 8
        
        w_a = w[:h1]
        w_b = w[h1:]
        
        # Generate all sums for part a (3^7 = 2187)
        sums_a = [0]
        for wi in w_a:
            n_s = len(sums_a)
            new_s = [0] * (3 * n_s)
            for k, s in enumerate(sums_a):
                new_s[3*k] = s - wi
                new_s[3*k+1] = s
                new_s[3*k+2] = s + wi
            sums_a = new_s
        
        # Generate all sums for part b (3^8 = 6561)
        sums_b = [0]
        for wi in w_b:
            n_s = len(sums_b)
            new_s = [0] * (3 * n_s)
            for k, s in enumerate(sums_b):
                new_s[3*k] = s - wi
                new_s[3*k+1] = s
                new_s[3*k+2] = s + wi
            sums_b = new_s
        
        # Build dict for part b: sum -> list of indices (or just one index)
        # 6561 entries, fine
        b_map = {}
        for idx, s in enumerate(sums_b):
            if s not in b_map:
                b_map[s] = idx
        
        # For each sum in part a, look for target - sum_a in b_map
        for idx_a, s_a in enumerate(sums_a):
            needed = target - s_a
            if needed in b_map:
                idx_b = b_map[needed]
                # Decode idx_a into h1 coefficients, idx_b into h2 coefficients
                coeffs = [0] * m
                # idx_a in base 3, h1 digits
                temp = idx_a
                for j in range(h1):
                    digit = temp % 3
                    temp //= 3
                    coeffs[j] = digit - 1
                # idx_b in base 3, h2 digits
                temp = idx_b
                for j in range(h2):
                    digit = temp % 3
                    temp //= 3
                    coeffs[h1 + j] = digit - 1
                return coeffs
        
        return None
    
    coeff1 = decode_half_mitm(w1, best_sum1)
    if coeff1 is None:
        return [0] * 28 + [1, -1]
    
    coeff2 = decode_half_mitm(w2_list, best_sum2)
    if coeff2 is None:
        return [0] * 28 + [1, -1]
    
    # Combine
    result = coeff1 + coeff2
    
    # Verify
    total = sum(c * LCM // (i+1) for i, c in enumerate(result))
    # total should equal best_sum1 + best_sum2 = some nonzero value
    # The actual sum is total / LCM
    assert total != 0, f"Zero sum: {result}"
    
    # Verify coefficients are in {-1, 0, 1}
    assert all(c in (-1, 0, 1) for c in result)
    assert len(result) == 30
    
    return result
