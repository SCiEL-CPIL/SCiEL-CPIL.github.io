"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from math import ceil, gcd

def construct():
    """Find minimum number of distinct denominators in [5,10000000] summing reciprocals to 1."""
    
    def solve(remaining, min_denom, max_terms, path):
        """Find representation of 'remaining' as sum of exactly max_terms unit fractions
        with denominators >= min_denom, strictly increasing."""
        if max_terms == 0:
            return None if remaining != 0 else list(path)
        if remaining == 0:
            return list(path) if max_terms == 0 else None
        
        num = remaining.numerator
        den = remaining.denominator
        
        # Minimum denominator: 1/d <= remaining => d >= ceil(den/num)
        min_d = max(min_denom, ceil(den / num))
        
        # Maximum denominator: we need to be able to form remaining - 1/d with max_terms-1 terms
        # The minimum sum of max_terms-1 unit fractions with denominators > d is
        # 1/(d+1) + 1/(d+2) + ... + 1/(d+max_terms-1)
        # So we need remaining - 1/d <= sum_{i=1}^{max_terms-1} 1/(d+i)
        # And remaining - 1/d >= 0, so 1/d <= remaining => d >= ceil(den/num) (already have)
        
        # Upper bound: 1/d >= remaining / max_terms => d <= max_terms * den / num
        max_d = min(10000000, max(min_d, int(max_terms * den / num) + 1))
        
        # For the last term (max_terms == 1), remaining must be a unit fraction
        if max_terms == 1:
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000:
                    return list(path) + [d]
            return None
        
        # For max_terms == 2, solve 1/b + 1/c = remaining with b < c, b >= min_d
        # (b+c)/(bc) = num/den => den(b+c) = num*bc => num*bc - den*b - den*c = 0
        # (num*b - den)(num*c - den) = den^2
        if max_terms == 2:
            # Factor den^2 and find pairs
            target = den * den
            # Find factor pairs (f1, f2) where f1 * f2 = target, f1 <= f2
            # b = (f1 + den) / num, c = (f2 + den) / num
            # Need b, c integers, b >= min_d, b < c, c <= 10000000
            # f1 = num*b - den, so f1 > 0 means b > den/num
            for f1 in range(1, int(target**0.5) + 1):
                if target % f1 == 0:
                    f2 = target // f1
                    # b = (f1 + den) / num, c = (f2 + den) / num
                    if (f1 + den) % num == 0 and (f2 + den) % num == 0:
                        b = (f1 + den) // num
                        c = (f2 + den) // num
                        if b >= min_d and b < c and c <= 10000000:
                            return list(path) + [b, c]
                    # Also try swapped
                    if (f2 + den) % num == 0 and (f1 + den) % num == 0:
                        b = (f2 + den) // num
                        c = (f1 + den) // num
                        if b >= min_d and b < c and c <= 10000000:
                            return list(path) + [b, c]
            return None
        
        # General case: try denominators from min_d upward
        # Pruning: the maximum possible sum with max_terms terms starting from min_d is
        # sum_{i=0}^{max_terms-1} 1/(min_d + i)
        # If remaining > that, no solution
        max_possible = sum(1.0 / (min_d + i) for i in range(max_terms))
        if remaining > max_possible + 1e-15:
            return None
        
        # Try each valid denominator
        for d in range(min_d, max_d + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining <= 0:
                if new_remaining == 0 and max_terms == 1:
                    return list(path) + [d]
                break
            # Pruning: check if new_remaining can be represented with max_terms-1 terms
            # Minimum denominator for next is d+1
            # Maximum possible sum: sum_{i=1}^{max_terms-1} 1/(d+i)
            if max_terms - 1 > 0:
                max_next = sum(1.0 / (d + i) for i in range(1, max_terms))
                if new_remaining > max_next + 1e-15:
                    continue
                # Minimum possible sum with max_terms-1 terms each >= 1/(d+max_terms-1)
                # Actually the minimum is when denominators are as large as possible
                # But we have an upper bound on denominators of 10000000
                # Skip this check for now
            result = solve(new_remaining, d + 1, max_terms - 1, path + [d])
            if result is not None:
                return result
        
        return None
    
    target = Fraction(1, 1)
    
    # Try increasing number of terms from minimum possible
    # We know 3 terms is impossible (proved in analysis)
    for num_terms in range(4, 30):
        result = solve(target, 5, num_terms, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
