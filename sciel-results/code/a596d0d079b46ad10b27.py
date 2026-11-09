"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # We need to find distinct integer denominators in [5, 10000000]
    # whose reciprocal sum is exactly 1, minimizing the count.
    
    # Let's verify that range(5, 257) sums to 1
    denom_list = list(range(5, 257))
    total = sum(Fraction(1, d) for d in denom_list)
    
    if total == 1:
        return denom_list
    
    # If not, use a greedy Egyptian fraction approach
    def greedy_egyptian(target, min_denom, max_terms):
        """Find Egyptian fraction for target using greedy algorithm."""
        terms = []
        remaining = target
        used = set()
        
        for _ in range(max_terms):
            if remaining == 0:
                break
            if remaining < 0:
                return None
            
            # Find the largest unit fraction <= remaining with denom >= min_denom
            # 1/d <= remaining means d >= 1/remaining = remaining.denominator/remaining.numerator
            # So d >= ceil(remaining.denominator / remaining.numerator)
            
            if remaining.numerator == 0:
                break
            
            d_min = max(min_denom, -(-remaining.denominator // remaining.numerator))
            
            # Try denominators starting from d_min
            found = False
            for d in range(d_min, d_min + 100000):
                if d in used:
                    continue
                val = Fraction(1, d)
                if val <= remaining:
                    terms.append(d)
                    used.add(d)
                    remaining -= val
                    found = True
                    break
            
            if not found:
                return None
        
        if remaining == 0:
            return terms
        return None
    
    # Try greedy with increasing max_terms
    for max_terms in range(1, 257):
        if time.time() > deadline - 5:
            break
        
        result = greedy_egyptian(Fraction(1, 1), 5, max_terms)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(result) <= 256:
                return result
    
    # Fallback: return range(5, 257) if it works
    total = sum(Fraction(1, d) for d in range(5, 257))
    if total == 1:
        return list(range(5, 257))
    
    raise RuntimeError("no valid representation found")
# EVOLVE-BLOCK-END

