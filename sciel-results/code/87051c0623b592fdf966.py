import time
from fractions import Fraction

def construct():
    """
    Finds 8 distinct integer denominators in [5, 10000000] whose reciprocal sum is exactly 1.
    Uses Depth-First Search with pruning.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    target = Fraction(1, 1)
    
    def dfs(rem_terms, min_den, current_sum, current_list):
        if time.time() > deadline:
            return None
        
        if rem_terms == 0:
            if current_sum == 0:
                return current_list
            return None
        
        if rem_terms == 1:
            if current_sum > 0:
                d_val = 1 / current_sum
                if d_val.denominator == 1:
                    d = d_val.numerator
                    if d > min_den and d <= 10000000:
                        return current_list + [d]
            return None
        
        # Upper bound for the next denominator d
        # We need 1/d + ... <= current_sum
        # So 1/d <= current_sum => d >= 1/current_sum
        # Also 1/d + ... + 1/d_last <= rem_terms / d
        # So rem_terms / d >= current_sum => d <= rem_terms / current_sum
        max_den = int(rem_terms / current_sum) if current_sum > 0 else 10000000
        if max_den > 10000000:
            max_den = 10000000
        
        # Lower bound is min_den
        # We also need to ensure the remaining sum is achievable
        # Min sum of rem_terms fractions starting from min_den is roughly rem_terms / 10000000
        # Max sum of rem_terms fractions starting from min_den is rem_terms / min_den
        
        for d in range(max(min_den, 5), max_den + 1):
            new_sum = current_sum - Fraction(1, d)
            if new_sum < 0:
                break
            
            # Pruning 1: Max possible sum of remaining (rem_terms - 1) fractions
            # The largest possible sum is if we pick the smallest possible denominators: d+1, d+2, ...
            # Sum approx (rem_terms - 1) / (d + 1)
            if new_sum > Fraction(rem_terms - 1, d + 1):
                continue
            
            # Pruning 2: Min possible sum of remaining (rem_terms - 1) fractions
            # The smallest possible sum is if we pick the largest possible denominators: 10000000
            # Sum approx (rem_terms - 1) / 10000000
            if new_sum < Fraction(rem_terms - 1, 10000000):
                break
            
            res = dfs(rem_terms - 1, d + 1, new_sum, current_list + [d])
            if res:
                return res
        return None

    # Search for 8 terms
    res = dfs(8, 5, target, [])
    if res:
        return res

    # Fallback: Search for 9 terms if 8 fails (should not happen)
    res = dfs(9, 5, target, [])
    if res:
        return res

    # Fallback: Return a known valid set with more terms if all else fails
    # Example: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + ...
    # But we must ensure sum is 1.
    # If DFS fails, it's a bug in pruning or logic.
    raise RuntimeError("No solution found within deadline")
