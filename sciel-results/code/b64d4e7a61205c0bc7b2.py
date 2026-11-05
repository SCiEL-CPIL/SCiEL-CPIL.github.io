import time
from fractions import Fraction
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    max_terms_allowed = 256
    max_denom_limit = 10000000
    
    best_max = float('inf')
    best_solution = None

    def update_best(denoms, current_max):
        nonlocal best_max, best_solution
        if current_max < best_max:
            best_max = current_max
            best_solution = list(denoms)

    # Meet-in-the-middle: Precompute all 2-term sums 1/a + 1/b with a < b, a,b <= bound
    # Key: the Fraction value, Value: list of (a, b) pairs
    def build_two_term_map(bound):
        """Build map from Fraction -> list of (a,b) pairs where 1/a+1/b=value, a<b, a,b<=bound"""
        two_term_map = defaultdict(list)
        for a in range(1, bound + 1):
            for b in range(a + 1, bound + 1):
                val = Fraction(1, a) + Fraction(1, b)
                two_term_map[val].append((a, b))
        return two_term_map

    # Strategy: Try increasing bounds for the two-term precomputation
    # For 3 terms: 1/c + (1/a + 1/b) = 5/31, so 1/a+1/b = 5/31 - 1/c
    # For 4 terms: 1/c + 1/d + (1/a + 1/b) = 5/31, so 1/a+1/b = 5/31 - 1/c - 1/d

    # First, try to find good 3-term solutions with a reasonable bound
    # The largest denominator in a 3-term solution for 5/31 is likely not too large
    # Let's try bounds: 100, 200, 500, 1000, 2000, 5000, 10000
    
    bounds_to_try = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    
    for bound in bounds_to_try:
        if time.time() > deadline:
            break
        
        # Build two-term map up to this bound
        two_term_map = build_two_term_map(bound)
        
        # Search for 3-term solutions: 1/c + 1/a + 1/b = 5/31
        # => 1/a + 1/b = 5/31 - 1/c
        # c must be > b > a, so c >= 3
        # Also 1/c < 5/31 => c > 31/5 = 6.2 => c >= 7
        # And 3/c > 5/31 => c < 93/5 = 18.6 => c <= 18 for the largest term bound
        # But we want to minimize the LARGEST denominator, which is c
        # So try c from 7 upward
        
        found_any = False
        for c in range(7, bound + 1):
            if time.time() > deadline:
                break
            rem = target - Fraction(1, c)
            if rem <= 0:
                continue
            # Look up rem in two_term_map
            if rem in two_term_map:
                for a, b in two_term_map[rem]:
                    # Check distinctness: a, b, c all distinct and a < b < c
                    if a != b and b != c and a != c and a < b < c:
                        if c <= max_denom_limit:
                            update_best([a, b, c], c)
                            found_any = True
            # Early exit if we found a solution with small c
            if found_any and c >= best_max:
                break
        
        # If we found a solution with max denom <= bound, we can stop increasing bound
        if best_max <= bound:
            break
    
    # Now search for 4-term solutions using the same two-term map
    # 1/a + 1/b + 1/c + 1/d = 5/31 with a < b < c < d
    # => 1/a + 1/b = 5/31 - 1/c - 1/d
    # We need c < d, and the two-term sum 1/a+1/b must have a < b < c
    
    # Use the largest two_term_map we built
    # Rebuild with a good bound if needed
    final_bound = min(bounds_to_try[-1], max(best_max, 1000))
    if time.time() < deadline - 5:
        two_term_map = build_two_term_map(final_bound)
        
        # For 4 terms: iterate over c, d with c < d, c >= 4
        # 1/c + 1/d < 5/31 (since 1/a + 1/b > 0)
        # Also 1/c + 1/d > 5/31 - 2/final_bound (rough lower bound for 1/a+1/b)
        
        # c < d, and we want d to be as small as possible
        # Try c from 4 upward
        for c in range(4, final_bound + 1):
            if time.time() > deadline:
                break
            if c >= best_max:
                break  # d > c, so d >= c+1 > best_max
            
            # d > c, and 1/c + 1/d < 5/31
            # 1/d < 5/31 - 1/c
            # d > 1 / (5/31 - 1/c)
            
            rem_cd_max = target - Fraction(1, c)
            if rem_cd_max <= 0:
                continue
            
            # d must satisfy: 1/d < rem_cd_max => d > 1/rem_cd_max
            # Also 1/d > 0 always
            # And we need 5/31 - 1/c - 1/d > 0 => 1/d < rem_cd_max => d > 1/rem_cd_max
            # And 5/31 - 1/c - 1/d must be representable as 1/a + 1/b with a < b < c
            
            d_min = c + 1
            # d_max from 1/d < rem_cd_max => d > 1/rem_cd_max
            # So d_max is not directly bounded by this, but we need rem = target - 1/c - 1/d to be positive
            # and representable as sum of two unit fractions with denominators < c
            
            # Upper bound on d: rem = target - 1/c - 1/d must be > 0
            # => 1/d < target - 1/c
            # => d > 1/(target - 1/c)
            # Also rem must be >= 1/(c-1) + 1/c (minimum two-term sum with b < c, a < b)
            # Actually minimum is 1/(c-1) + 1/(c-2) if c-2 >= 2
            # Let's just iterate d from c+1 upward with a reasonable cap
            
            # Cap d at best_max - 1 since we need d < best_max to improve
            d_max = min(final_bound, best_max - 1) if best_max < float('inf') else final_bound
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    break
                rem = target - Fraction(1, c) - Fraction(1, d)
                if rem <= 0:
                    continue
                # Look up rem in two_term_map
                if rem in two_term_map:
                    for a, b in two_term_map[rem]:
                        # Check: a < b < c < d, all distinct
                        if a < b < c < d and a != b and b != c and c != d and a != c and a != d and b != d:
                            if d <= max_denom_limit:
                                update_best([a, b, c, d], d)
        
    # Fallback
    if best_solution is None:
        best_solution = [8, 28, 1736]
        best_max = 1736
    else:
        # Validate
        if 0 < len(best_solution) <= 256:
            if all(1 <= d <= 10000000 for d in best_solution):
                if len(set(best_solution)) == len(best_solution):
                    s = sum(Fraction(1, d) for d in best_solution)
                    if s != target:
                        best_solution = [8, 28, 1736]
                        best_max = 1736
                else:
                    best_solution = [8, 28, 1736]
                    best_max = 1736
            else:
                best_solution = [8, 28, 1736]
                best_max = 1736
        else:
            best_solution = [8, 28, 1736]
            best_max = 1736
            
    return best_solution
