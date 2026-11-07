import math
import time

def construct():
    deadline = time.time() + 55.0
    n = 210
    max_val = 10210
    
    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    # 210 = 2 * 3 * 5 * 7
    # We want a subset S of {211..10210} such that 210 * prod(S) is a square.
    # This is equivalent to finding S such that the parity vector of prime exponents of prod(S) 
    # is equal to the parity vector of 210 (which is (1,1,1,1) for primes 2,3,5,7).
    
    # Strategy: Direct construction for small |S|.
    # If |S|=1: x must be 210 * k^2. Smallest x > 210 is 210 * 4 = 840.
    # If |S|=2: x * y = 210 * k^2. We want to minimize max(x,y).
    #   This implies x and y should be close to sqrt(210) * k.
    #   Also x, y must be distinct integers > 210.
    #   Let x = 210 * a, y = b / a where b = 210 * k^2? No, x*y = 210 k^2.
    #   Actually, let's just iterate k for |S|=2 as in the seed, but we can be smarter.
    #   However, the seed already does this and gets 21.0 (meaning max=231).
    #   Let's check if we can do better than 231.
    #   If max=230, then x,y <= 230. Product <= 230*229 = 52670.
    #   210 * k^2 <= 52670 => k^2 <= 250 => k <= 15.
    #   Let's check if any k in 1..15 yields a factorization into two distinct ints > 210.
    
    best_max = max_val + 1
    best_result = None

    # Check |S|=1
    # x = 210 * k^2. k=1 -> 210 (not > 210). k=2 -> 840.
    # 840 > 210, 840 <= 10210.
    # 210 * 840 = 176400 = 420^2.
    # max = 840. raw = 630.
    # This is worse than 21. So we ignore |S|=1 unless |S|=2 fails.

    # Check |S|=2
    # x * y = 210 * k^2
    # We want min max(x,y).
    # Iterate k from 1 up.
    # For each k, M = 210 * k^2.
    # Find factor pair (x, y) of M such that x < y, x > 210, y > 210, y <= 10210.
    # To minimize y, we want x as large as possible (close to sqrt(M)).
    
    for k in range(1, 100):
        if time.time() > deadline:
            break
        M = 210 * k * k
        sqrt_M = math.isqrt(M)
        
        # We want x to be as close to sqrt_M as possible, but x < sqrt_M (since x<y)
        # And x > 210.
        # And y = M/x must be integer.
        # And y <= 10210.
        
        # Iterate x downwards from min(sqrt_M-1, 10210) down to 211.
        # But we can limit the range.
        # y = M/x <= 10210 => x >= M/10210.
        # So x is in [max(211, ceil(M/10210)), min(sqrt_M-1, 10210)].
        
        lo = max(211, (M + 10209) // 10210)
        hi = min(sqrt_M - 1, 10210)
        
        if lo > hi:
            continue
            
        # We want the largest x in this range that divides M.
        # This minimizes y = M/x.
        found = False
        # To speed up, we can check divisors of M.
        # But M is small enough (210 * 100^2 = 2,100,000) that simple loop is fine.
        # However, iterating x downwards from hi to lo might be fast if a solution exists near sqrt.
        
        for x in range(hi, lo - 1, -1):
            if M % x == 0:
                y = M // x
                if y > x and y > 210 and y <= 10210:
                    # Valid pair
                    if y < best_max:
                        best_max = y
                        best_result = [x, y]
                    found = True
                    break # Since we iterate x downwards, the first valid x gives the smallest y for this k.
                    # But we should check if a smaller k could yield an even smaller y?
                    # No, as k increases, M increases, so potential y increases.
                    # Wait, is it monotonic?
                    # For k=1, M=210. No pair.
                    # For k=2, M=840. x=21, y=40 (too small). x=... 
                    # Let's just iterate k and keep global best.
        
        # If we found a very small y, we might stop early?
        # The theoretical minimum for y is 211.
        # If best_max - 210 <= 1, we can't do better.
        if best_max - 210 <= 1:
            break

    # Check |S|=3?
    # x*y*z = 210 * k^2
    # This is more complex. But usually |S|=2 is optimal for minimizing max element
    # because splitting into more factors tends to increase the max element unless the product is very small.
    # Given 210 is small, |S|=2 is likely optimal.
    
    # The seed program found a result with raw value 21.0, meaning max=231.
    # Let's verify if my loop finds it.
    # If not, I will return the known best from the seed logic or a safe fallback.
    
    if best_result is None:
        # Fallback to the specific case found by seed or manual check
        # Seed found max=231.
        # Let's check k values manually for a moment to ensure we don't miss anything.
        # If the loop above didn't find anything, it means no pair exists for k<100?
        # That seems unlikely given the seed found one.
        # Ah, the seed iterates t (which is k) and finds a pair.
        # My loop is essentially the same but optimized to check x downwards.
        
        # Let's re-verify the logic.
        # If best_result is still None, it means no pair was found in k=1..100.
        # But the seed found one. Why?
        # Maybe the seed's `a` corresponds to my `x`.
        # Seed: `for a in range(lo, hi + 1)`. It iterates upwards.
        # It updates `best_max` if `candidate_max < best_max`.
        # My loop iterates `x` downwards from `hi`.
        # This should find the same pairs, just in a different order.
        # If I break on the first found `x` (largest x), I get the smallest `y` for that `k`.
        # This is correct.
        
        # Why might it fail?
        # Maybe `hi` calculation?
        # `hi = min(sqrt_M - 1, 10210)`.
        # If `sqrt_M` is an integer, `x` must be < `sqrt_M` to ensure `x < y`.
        # If `M` is a square, `x = sqrt_M` gives `y = x`, which is not distinct.
        # So `x < sqrt_M` is correct.
        
        # Let's assume the loop works. If it doesn't, I'll add a fallback.
        return [840] # Fallback, raw=630. Worse than 21, but valid.

    # Verify the result
    x, y = best_result
    prod = x * y * 210
    if is_square(prod) and x != y and x > 210 and y > 210 and x <= 10210 and y <= 10210:
        return [int(x), int(y)]
    else:
        return [840]
