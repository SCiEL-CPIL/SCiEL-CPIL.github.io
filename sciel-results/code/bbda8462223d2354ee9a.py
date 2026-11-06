"""Improved program for powerful-sum representation task."""
import time
import sys
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number (every prime exponent >= 2)."""
    if n < 2:
        return True  # 1 is powerful
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt == 1:
                return False
        d += 1
    if n > 1:
        return False  # remaining factor has exponent 1
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    pw = set()
    pw.add(1)
    # Powerful numbers: product of p^e where e >= 2
    # Equivalent to a^2 * b^3 where b is square-free
    # Iterate over square-free b (cube part) and a (square part)
    # b^3 <= limit => b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    is_sq_free[0] = False
    # Mark non-square-free: for each prime p, multiples of p^2 are not square-free
    for p in range(2, int(max_b ** 0.5) + 1):
        if is_sq_free[p]:  # p is prime
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                is_sq_free[k] = False
    square_free = [b for b in range(1, max_b + 1) if is_sq_free[b]]
    for b in square_free:
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            pw.add(val)
    return sorted(pw)

def construct():
    start_time = time.time()
    deadline = 135  # 150s total, leave 15s margin

    # Generate powerful numbers up to a reasonable limit
    # The count of powerful numbers up to x is ~ C * x^(2/3)
    # For x = 10^12, that's ~ 10^8 * C which is too many
    # For x = 10^9, that's ~ 10^6 * C ~ manageable
    # Let's try x = 10^9 first, and if time permits, go higher
    # Actually, let's be smarter: generate up to 10^8 first (fast),
    # then try 10^9, then 10^10 if time allows
    
    best_N = None
    best_pairs = []
    best_count = 0

    # Try different limits
    limits_to_try = [10**7, 10**8, 10**9, 10**10]
    
    for limit in limits_to_try:
        if time.time() - start_time > deadline - 10:
            break
        
        pw = _generate_powerful(limit)
        pw_set = set(pw)
        n_pw = len(pw)
        if n_pw < 2:
            continue
        
        # For each powerful number a <= b, a+b <= limit, count reps for N
        # Use a dict to count
        # Only consider N up to limit (since a,b <= limit means a+b <= 2*limit,
        # but we need a+b <= limit for both to be in our generated set)
        # Actually a+b can be up to 2*limit, but we need a+b to be the N
        # and we need both a and b to be in pw_set
        # Let's count for N values that are sums of two powerful numbers both <= limit
        
        # Efficient approach: for each a in pw, for each b in pw with b >= a and a+b <= limit:
        # That's O(n^2) which is too slow for n ~ 10^6
        # Better: use convolution-like or just iterate
        # For n ~ 10^5 (limit ~ 10^7), n^2 ~ 10^10, too slow
        # For n ~ 10^4 (limit ~ 10^5), n^2 ~ 10^8, borderline
        # 
        # Alternative: for each possible N, count pairs. But N ranges up to limit.
        # 
        # Let's use the approach: for each a in pw, for each b in pw with b >= a:
        #   N = a + b
        #   if N <= limit: count[N] += 1
        # This is O(n^2) but with early termination if b > limit - a
        # 
        # For limit = 10^7, n_pw ~ a few thousand, n^2 ~ 10^7, feasible
        # For limit = 10^8, n_pw ~ ~50k, n^2 ~ 2.5*10^9, too slow
        # 
        # So let's stick to limit ~ 10^7 or use a smarter method
        # 
        # Smarter: use the fact that for a fixed a, b ranges from a to limit-a
        # and we need b in pw_set. We can iterate a and for each, find b's in pw_set
        # in range [a, limit-a]. Still O(n^2) worst case.
        
        # Let's just do the direct approach for small enough n_pw
        if n_pw <= 20000:
            counts = {}
            for i, a in enumerate(pw):
                if a > limit // 2:
                    break
                # b >= a, b <= limit - a
                max_b = limit - a
                for j in range(i, n_pw):
                    b = pw[j]
                    if b > max_b:
                        break
                    N = a + b
                    counts[N] = counts.get(N, 0) + 1
            
            if counts:
                max_N = max(counts, key=counts.get)
                max_count = counts[max_N]
                if max_count > best_count:
                    best_count = max_count
                    best_N = max_N
                    # Collect pairs
                    best_pairs = []
                    for a in pw:
                        if 2 * a > max_N:
                            break
                        b = max_N - a
                        if b >= a and b in pw_set:
                            best_pairs.append([a, b])
        
        # Check time
        if time.time() - start_time > deadline - 5:
            break

    # If we didn't find anything, fall back to small limit
    if best_N is None:
        limit = 20000
        pw = _generate_powerful(limit)
        pw_set = set(pw)
        best = max(range(2, limit + 1), key=lambda n: sum(1 for a in pw if 2 * a <= n and n - a in pw_set))
        best_N = best
        best_pairs = [[a, best - a] for a in pw if 2 * a <= best and best - a in pw_set]

    return {'N': best_N, 'pairs': best_pairs}
