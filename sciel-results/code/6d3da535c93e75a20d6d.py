import math

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    count = 0
    while temp % 2 == 0:
        count += 1
        temp //= 2
    if count > 0:
        result *= (2**(count+1) - 1)
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            power = 1
            while temp % p == 0:
                count += 1
                temp //= p
                power *= p
            result *= (power * (p + 1)) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    N = 10000000
    
    # Generate all n <= N where n is a square or twice a square
    candidates = set()
    
    # Perfect squares: k^2 <= N => k <= sqrt(N)
    max_k_sq = int(math.isqrt(N))
    for k in range(1, max_k_sq + 1):
        candidates.add(k * k)
    
    # Twice squares: 2*k^2 <= N => k <= sqrt(N/2)
    max_k_twice = int(math.isqrt(N // 2))
    for k in range(1, max_k_twice + 1):
        candidates.add(2 * k * k)
    
    # Compute sigma for each candidate and group by sigma value
    sigma_groups = {}
    for n in candidates:
        s = _sigma(n)
        if s not in sigma_groups:
            sigma_groups[s] = []
        sigma_groups[s].append(n)
    
    # Find best pair
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    for s, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        nums_sorted = sorted(nums)
        # For each b in nums, find a in nums closest to 1.5*b
        # a/b ≈ 3/2 => a ≈ 1.5*b
        for i in range(len(nums_sorted)):
            b = nums_sorted[i]
            target_a = 1.5 * b
            # Binary search for target_a in nums_sorted
            # We want a such that |a - 1.5*b| is minimized, but a must be in nums_sorted
            # Use bisect
            lo, hi = 0, len(nums_sorted)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums_sorted[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for idx in [lo - 1, lo]:
                if 0 <= idx < len(nums_sorted):
                    a = nums_sorted[idx]
                    if a == b:
                        continue
                    # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                    err_num = abs(2 * a - 3 * b)
                    if err_num == 0:
                        continue  # Invalid per contract
                    err_den = 2 * b
                    err = err_num / err_den
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
    
    return [best_a, best_b]
