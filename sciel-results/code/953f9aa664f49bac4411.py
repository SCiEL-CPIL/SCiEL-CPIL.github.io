from fractions import Fraction
import math

def construct():
    target = Fraction(1, 1)
    min_denom = 5
    max_denom = 10000000
    
    def dfs(remaining, min_d, used, depth, max_depth):
        if remaining == 0:
            return []
        if depth >= max_depth:
            return None
        if remaining < 0:
            return None
        
        # d must satisfy: 1/d <= remaining, so d >= ceil(1/remaining)
        d_min = max(min_d, math.ceil(1 / float(remaining)))
        
        # d must also satisfy: the remaining sum after 1/d can be achieved
        # with (max_depth - depth - 1) terms, each >= 1/(d+1), 1/(d+2), ...
        # Max possible remaining sum with (max_depth-depth-1) terms starting at d+1:
        # S_max = sum_{i=0}^{max_depth-depth-2} 1/(d+1+i)
        # We need remaining - 1/d <= S_max
        # Also, min possible: S_min = (max_depth-depth-1) / max_denom
        # We need remaining - 1/d >= S_min
        
        terms_left = max_depth - depth - 1
        if terms_left < 0:
            return None
        
        # Upper bound on d: 1/d >= remaining / (1 + terms_left) approximately
        # More precisely, the max sum of terms_left unit fractions each <= 1/(d+1) is terms_left/(d+1)
        # So we need remaining - 1/d <= terms_left / (d+1)
        # This gives an upper bound on d
        
        d_max = max_denom
        # From remaining - 1/d <= terms_left/(d+1):
        # remaining <= 1/d + terms_left/(d+1)
        # For large d, RHS ~ (1+terms_left)/d, so d <= (1+terms_left)/remaining
        d_max = min(d_max, int((1 + terms_left) / float(remaining)) + 2)
        
        if d_min > d_max:
            return None
        
        for d in range(d_min, d_max + 1):
            if d in used:
                continue
            if d < 5 or d > max_denom:
                continue
            
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                continue
            # Pruning: check if new_rem can be achieved with terms_left terms
            if terms_left > 0:
                # Max sum with terms_left terms starting from d+1
                # Approximate: terms_left / (d+1)
                if new_rem > Fraction(terms_left, d + 1):
                    continue
                # Min sum: terms_left / max_denom
                if new_rem < Fraction(terms_left, max_denom):
                    # Can't reach new_rem with such small terms
                    # But as d increases, new_rem increases, so maybe continue
                    pass
            
            used.add(d)
            result = dfs(new_rem, d + 1, used, depth + 1, max_depth)
            used.remove(d)
            
            if result is not None:
                return [d] + result
        
        return None
    
    # Try increasing number of terms
    for count in range(4, 257):
        used = set()
        result = dfs(target, min_denom, used, 0, count)
        if result is not None:
            return result
    
    raise RuntimeError("no solution found")

