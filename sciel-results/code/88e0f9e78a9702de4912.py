"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try to find minimum number of distinct denominators >= 5 summing to 1
    # Start with small counts and increase
    
    def dfs(remaining, min_denom, current, depth, max_depth):
        """
        remaining: numerator/denominator of remaining sum (as fraction num/den)
        min_denom: minimum allowed denominator
        current: list of chosen denominators
        depth: current depth
        max_depth: max allowed depth
        Returns list of denominators if found, else None
        """
        if time.time() > deadline:
            return None
        
        if remaining[0] == 0:
            return current
        
        if len(current) >= max_depth:
            return None
        
        # If remaining is a unit fraction, we can add it directly
        if remaining[1] == 1:
            d = remaining[0]
            if d >= min_denom and d >= 5 and d not in current:
                return current + [d]
        
        # Try adding the next denominator
        # 1/d <= remaining, so d >= 1/remaining = den/num
        min_d = max(min_denom, 5, remaining[1] // remaining[0] + 1)
        if remaining[0] * (min_d - 1) >= remaining[1]:
            min_d = remaining[1] // remaining[0] + 1
        min_d = max(min_d, min_denom, 5)
        
        # Upper bound: 1/d >= remaining/(max_depth - len(current))
        # So d <= (max_depth - len(current)) / remaining
        max_terms_left = max_depth - len(current)
        max_d = (max_terms_left * remaining[1]) // remaining[0]
        
        for d in range(min_d, min(max_d + 1, 10000001)):
            if d in current:
                continue
            # new_remaining = remaining - 1/d
            # = (num*d - den) / (den*d)
            new_num = remaining[0] * d - remaining[1]
            if new_num < 0:
                break
            new_den = remaining[1] * d
            # Reduce fraction
            from math import gcd
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            result = dfs((new_num, new_den), d + 1, current + [d], depth + 1, max_depth)
            if result is not None:
                return result
        
        return None
    
    # Try increasing number of terms
    for max_terms in range(2, 10):
        result = dfs((1, 1), 5, [], 0, max_terms)
        if result is not None:
            # Verify
            from fractions import Fraction
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(result) == len(set(result)) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: use the known solution if we can find one quickly
    # 1/5 + 1/6 + 1/7 + 1/42 = 1? Let's check: 42/210 + 35/210 + 30/210 + 5/210 = 112/210 = 8/15 ≠ 1
    # Let me try: 1/6 + 1/7 + 1/43 + 1/1806? This is getting complex.
    
    # Use a simpler known decomposition
    # 1 = 1/2 + 1/3 + 1/6 but 2,3 < 5
    # 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 but not distinct
    
    # Let's just return a valid solution we can construct
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + ... this is getting complicated
    
    # Fallback: use the dynamic programming approach from the seed
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
