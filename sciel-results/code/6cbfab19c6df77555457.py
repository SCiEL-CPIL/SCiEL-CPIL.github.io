def construct():
    import math
    
    MAX_N = 10_000_000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma for all n
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        # Factorize n using spf
        orig_n = n
        s = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            s *= power_sum
        sigma[orig_n] = s

    # 3. Group numbers by sigma
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
        
    # 4. Find best pair
    best_error = float('inf')
    best_a, best_b = None, None
    best_num = 0
    best_den = 1
    
    for sig_val, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        
        # Sort nums to allow efficient search? 
        # Or just double loop if len is small.
        # Max len is likely small. Let's check if we need sorting.
        # If we sort, we can binary search for 1.5*b.
        # But double loop is simpler and likely fast enough if k is small.
        # Let's assume k is small (< 100). k^2 = 10000 per group.
        # Number of groups is up to 10^7. But most groups have size 1.
        # Only groups with size >= 2 matter.
        # How many such groups? Probably much less than 10^7.
        
        for i in range(len(nums)):
            b = nums[i]
            # Ideal a is 1.5 * b
            ideal_a = 1.5 * b
            
            # Check all other a in the group?
            # This is O(k^2).
            # We can optimize by only checking a close to ideal_a.
            # But without sorting, we don't know which are close.
            # Let's sort the list once per group.
            # Sorting cost O(k log k).
            # Then for each b, binary search for ideal_a.
            
            # Actually, let's just sort the list once.
            # But we are inside the loop over groups.
            # Let's move sorting outside the inner loop.
            pass

    # Let's restructure:
    # Sort each group's list.
    for sig_val, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        nums.sort()
        k = len(nums)
        
        # For each b in nums, find a in nums closest to 1.5*b
        for i in range(k):
            b = nums[i]
            target = 1.5 * b
            
            # Binary search for target in nums
            # We want a such that |a - target| is minimized.
            # Since nums is sorted, we can use bisect.
            import bisect
            pos = bisect.bisect_left(nums, target)
            
            candidates = []
            if pos < k:
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Check error |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                
                # Compare num/den with best_num/best_den
                # num/den < best_num/best_den  => num * best_den < best_num * den
                if best_a is None or num * best_den < best_num * den:
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den
                    
    if best_a is None:
        return [11, 6]
        
    return [best_a, best_b]
