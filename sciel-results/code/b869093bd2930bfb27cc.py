import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe within 60s limit

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def search(num, den, k, min_d, path):
        """
        DFS to find k distinct denominators > min_d-1 such that sum(1/d) = num/den.
        Returns list of denominators or None.
        """
        # Check time
        if time.time() > deadline:
            return None
        
        if k == 0:
            return path if num == 0 and den == 1 else None
            
        if k == 1:
            # We need num/den = 1/d
            # So d = den/num
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000:
                    return path + [d]
            return None

        # Lower bound for d:
        # 1/d <= num/den => d >= den/num
        # Also d >= min_d
        d_min = max(min_d, math.ceil(den / num))
        
        # Upper bound for d:
        # We need to fit k terms. The largest possible sum for k terms starting from d is
        # sum_{i=0}^{k-1} 1/(d+i). If this sum is less than num/den, then d is too large?
        # No, if d is large, 1/d is small, so we need more terms or larger terms?
        # Actually, if d is too large, the sum of k terms 1/d + ... + 1/(d+k-1) will be small.
        # If max_sum < num/den, then we can't reach num/den with denominators >= d.
        # So d must be small enough such that max_sum >= num/den.
        # This gives an upper bound on d.
        
        # Let's approximate max_sum.
        # For small k, we can compute exactly.
        # max_sum(d) = sum(1/(d+i) for i in range(k))
        # We need max_sum(d) >= num/den.
        
        # Also, since denominators are distinct and > d_min, the smallest possible sum for k terms
        # starting from d is just 1/d + ... which is the max sum.
        # Wait, if we pick d, the next denominators are > d.
        # So the max sum of the remaining k-1 terms is sum_{i=1}^{k-1} 1/(d+i).
        # Total max sum if we pick d is 1/d + sum_{i=1}^{k-1} 1/(d+i).
        # This must be >= num/den.
        
        # Let's find d_max.
        # Since d_min is the lower bound, we iterate d from d_min.
        # We can break early if max_possible_sum(d) < num/den.
        
        d_max = 10000000
        # Optimization: if d is very large, 1/d is very small.
        # num/den <= k/d => d <= k * den / num
        d_max = min(d_max, int((k * den) / num))
        
        if d_min > d_max:
            return None

        # Iterate d
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
                
            # Calculate remaining fraction: num/den - 1/d
            # = (num*d - den) / (den*d)
            n2 = num * d - den
            d2 = den * d
            
            # n2 must be positive (1/d < num/den)
            if n2 <= 0:
                continue
            
            # Check if it's possible to complete with k-1 terms starting from d+1
            # Max sum of k-1 terms with denominators d+1, d+2, ...
            # S_max = sum(1/(d+i) for i in range(1, k))
            # If n2/d2 > S_max, then d is too small?
            # No, if n2/d2 is too large, we need smaller denominators (smaller d).
            # But we are iterating d upwards.
            # So if n2/d2 > S_max, it means even with the smallest possible remaining denominators,
            # we can't sum up to n2/d2. This implies d is too small?
            # Wait.
            # R = num/den.
            # R = 1/d + R'.
            # R' = R - 1/d.
            # As d increases, 1/d decreases, so R' increases.
            # The capacity of the remaining k-1 terms (denoms > d) is S_max(d).
            # S_max(d) decreases as d increases.
            # We need R' <= S_max(d).
            # R - 1/d <= S_max(d).
            # If this fails, it means R' is too large for the capacity.
            # This happens when d is too small?
            # Let's check.
            # If d is small, 1/d is large, R' is small. Capacity S_max is large.
            # So R' <= S_max is likely true.
            # If d is large, 1/d is small, R' is large. Capacity S_max is small.
            # So R' <= S_max might fail.
            # So if R' > S_max, we need to increase d? No, increasing d makes R' larger and S_max smaller.
            # So if it fails at d, it will fail for all larger d.
            # So we can break?
            # Wait, if d is small, R' is small. It fits.
            # If d is large, R' is large. It might not fit.
            # So if we find a d where R' > S_max, then for any d' > d, R' will be even larger and S_max smaller.
            # So we can break.
            
            # Let's verify this logic.
            # f(d) = R - 1/d - S_max(d).
            # We need f(d) <= 0.
            # R is constant.
            # -1/d increases with d (becomes less negative).
            # -S_max(d) increases with d (S_max decreases, so -S_max increases).
            # So f(d) is increasing with d.
            # If f(d) > 0, then for all d' > d, f(d') > 0.
            # So if R' > S_max, we can break.
            
            # Calculate S_max
            s_max = 0.0
            for i in range(1, k):
                s_max += 1.0 / (d + i)
            
            # Compare n2/d2 with s_max
            # Use floating point for speed, but be careful.
            # n2/d2 > s_max ?
            if n2 / d2 > s_max:
                break
            
            # Reduce fraction
            g = gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            result = search(n2_reduced, d2_reduced, k - 1, d + 1, path + [d])
            if result is not None:
                return result
        return None

    # Try k=8 first, as k<8 is impossible
    for k in range(8, 12):
        if time.time() > deadline:
            break
        solution = search(1, 1, k, 5, [])
        if solution is not None:
            return solution
            
    # Fallback: Greedy for k=256
    # This is a safety net, though unlikely to be needed if k=8 works
    def greedy_egyptian(target_frac, min_denom, max_terms):
        remaining = target_frac
        chosen = []
        used_denoms = set()
        
        for _ in range(max_terms):
            if remaining <= 0:
                break
            if time.time() > deadline:
                return None
                
            d_min_from_frac = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d_min = max(min_denom, d_min_from_frac)
            
            found = False
            for d in range(d_min, min(d_min + 100000, 10000001)):
                if d in used_denoms:
                    continue
                if d > 10000000:
                    break
                unit = Fraction(1, d)
                if unit <= remaining:
                    terms_left = max_terms - len(chosen) - 1
                    if terms_left > 0:
                        if remaining - unit > terms_left * Fraction(1, d + 1):
                            continue
                    
                    chosen.append(d)
                    used_denoms.add(d)
                    remaining -= unit
                    min_denom = d + 1
                    found = True
                    break
            
            if not found:
                return None
        
        if remaining == 0:
            return chosen
        return None

    return greedy_egyptian(Fraction(1, 1), 5, 256)
# EVOLVE-BLOCK-END
