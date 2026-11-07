import time
import math
import sys
from collections import defaultdict
from itertools import combinations

def construct():
    deadline = time.time() + 55.0
    n = 210
    max_val = 10210

    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    def square_free_kernel(x):
        """Compute the square-free kernel of x."""
        result = 1
        d = 2
        while d * d <= x:
            if x % d == 0:
                count = 0
                while x % d == 0:
                    x //= d
                    count += 1
                if count % 2 == 1:
                    result *= d
            d += 1 if d == 2 else 2
        if x > 1:
            result *= x
        return result

    # Precompute square-free kernels for all numbers in (210, 10210]
    sf = {}
    for i in range(211, max_val + 1):
        sf[i] = square_free_kernel(i)

    # Build a dictionary: square_free_value -> sorted list of numbers with that kernel
    kernel_to_numbers = defaultdict(list)
    for i in range(211, max_val + 1):
        kernel_to_numbers[sf[i]].append(i)

    for k in kernel_to_numbers:
        kernel_to_numbers[k].sort()

    best_max = max_val + 1
    best_result = None

    # Strategy 1: Pairs (Current logic, kept as baseline)
    # We want a*b*210 to be a square.
    # sf(a*b*210) = 1 => sf(a)*sf(b)*sf(210) = square.
    # Let K = sf(210). sf(a)*sf(b) must have the same square-free part as K.
    # Actually, it's easier to just check: is_square(a*b*210)
    # But we can use the kernel property: sf(a)*sf(b)*K must be a square.
    # This implies sf(a)*sf(b) = K * (square).
    # Since sf(a) and sf(b) are square-free, sf(a)*sf(b) is square-free times a square?
    # No, sf(a)*sf(b) is not necessarily square-free if they share primes.
    # The condition is: for every prime p, exp_p(a) + exp_p(b) + exp_p(210) is even.
    
    # Let's stick to the previous pair logic which was efficient.
    for a in range(211, max_val + 1):
        if time.time() >= deadline:
            break
        A = sf[a]
        # We need b such that sf(a*b*210) == 1.
        # This is equivalent to: sf(A * sf(210) * sf(b)) == 1? No.
        # Let's just use the property: sf(a*b*210) = sf(a) * sf(b) * sf(210) / (squares).
        # Actually, sf(x*y) = sf(sf(x)*sf(y)).
        # So sf(a*b*210) = sf(sf(a) * sf(b) * sf(210)).
        # We want this to be 1.
        # Let K210 = sf(210).
        # We want sf(A * sf(b) * K210) = 1.
        # This implies A * sf(b) * K210 is a square.
        # Let T = A * K210. We want sf(T * sf(b)) = 1.
        # This implies sf(b) = sf(T).
        # So we look for b such that sf(b) == sf(A * K210).
        
        T = A * sf[210] # Wait, sf(210) is just a number. 210 = 2*3*5*7. sf(210)=210.
        # Actually, let's just compute the required kernel for b directly.
        # Condition: exp_p(a) + exp_p(b) + exp_p(210) ≡ 0 (mod 2)
        # exp_p(b) ≡ exp_p(a) + exp_p(210) (mod 2)
        # So sf(b) must have prime factors p where exp_p(a)+exp_p(210) is odd.
        # This is exactly sf(a * 210).
        
        B_required = square_free_kernel(A * 210)
        
        if B_required in kernel_to_numbers:
            candidates = kernel_to_numbers[B_required]
            # Find smallest b > a
            lo, hi = 0, len(candidates)
            while lo < hi:
                mid = (lo + hi) // 2
                if candidates[mid] <= a:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(candidates):
                b = candidates[lo]
                if b <= max_val:
                    candidate_max = b
                    if candidate_max < best_max:
                        best_max = candidate_max
                        best_result = (a, b)
                    if best_max - 210 <= 1:
                        return [int(a), int(b)]

    # Strategy 2: Triples
    # We want a*b*c*210 to be a square.
    # This is equivalent to sf(a)*sf(b)*sf(c)*sf(210) being a square (in terms of parity).
    # Let K210 = sf(210) = 210.
    # We need sf(a)*sf(b)*sf(c) * 210 to be a square.
    # Let X = sf(a), Y = sf(b), Z = sf(c).
    # We need X*Y*Z*210 to be a square.
    # This means X*Y*Z must have the same square-free kernel as 210.
    # i.e., sf(X*Y*Z) == 210.
    
    # To minimize max(a,b,c), we should look at small a, b, c.
    # We can iterate over small a, then small b, and find the smallest c.
    
    # Precompute a list of (value, kernel) for small numbers to speed up search?
    # Or just iterate.
    
    # We want to find a, b, c in [211, M] such that max(a,b,c) is minimized.
    # We can iterate M from 211 upwards.
    # For a given M, check if there exist a,b,c in [211, M] satisfying the condition.
    # This is a bit expensive if M is large, but we expect M to be small.
    
    # Alternative: Iterate a from 211.
    # For each a, iterate b from 211 to a (to avoid duplicates, assume a<=b<=c).
    # Calculate required kernel for c:
    # sf(a)*sf(b)*sf(c)*210 = square
    # => sf(c) = sf(sf(a)*sf(b)*210)
    # Let R = sf(sf(a)*sf(b)*210).
    # We need to find c in [b, max_val] such that sf(c) == R.
    # To minimize max(a,b,c) = c (since c>=b>=a), we want the smallest such c >= b.
    
    # Optimization:
    # We can break early if a > best_max.
    
    for a in range(211, max_val + 1):
        if time.time() >= deadline:
            break
        if a > best_max:
            break
            
        A = sf[a]
        
        for b in range(211, a + 1): # b <= a to maintain order a>=b? No, let's say a <= b <= c.
            # If we enforce a <= b <= c, then max is c.
            # We want to minimize c.
            # So we iterate a from 211.
            # For each a, iterate b from a to ...?
            # If we iterate b from a, and find smallest c >= b, then max is c.
            # We want to find global min c.
            
            # Let's swap loops:
            # Iterate c from 211 to best_max? No, we don't know best_max yet.
            
            # Let's stick to:
            # Iterate a from 211.
            # Iterate b from a to max_val? No, that's too slow.
            
            # Better:
            # Iterate a from 211.
            # Iterate b from a to a + (best_max - a) ?
            
            # Let's just iterate b from a to max_val, but break if b > best_max.
            # And for each (a,b), find min c >= b.
            
            if b > best_max:
                break
                
            B = sf[b]
            
            # Calculate required kernel for c
            # sf(c) = sf(A * B * 210)
            R = square_free_kernel(A * B * 210)
            
            if R in kernel_to_numbers:
                candidates = kernel_to_numbers[R]
                # Find smallest c >= b
                lo, hi = 0, len(candidates)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if candidates[mid] < b:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < len(candidates):
                    c = candidates[lo]
                    if c <= max_val:
                        candidate_max = c
                        if candidate_max < best_max:
                            best_max = candidate_max
                            best_result = (a, b, c)
                            if best_max - 210 <= 1:
                                return [int(a), int(b), int(c)]

    # Verify and return
    if best_result is None:
        return [840]

    # Check if result is pair or triple
    if len(best_result) == 2:
        a, b = best_result
        prod = a * b * 210
        if is_square(prod):
            return [int(a), int(b)]
        else:
            return [840]
    else:
        a, b, c = best_result
        prod = a * b * c * 210
        if is_square(prod):
            return [int(a), int(b), int(c)]
        else:
            return [840]

