import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for evaluator overhead
    
    def find_solution(k):
        # Find k distinct denominators > 4 summing to 1
        # We use backtracking
        # To speed up, we only search denominators up to a certain limit initially
        # If not found, expand limit
        
        # For k=5, 6, 7, the denominators are likely small.
        # Let's try to find a solution with denominators < 1000 first.
        
        def backtrack(remaining, min_den, current, k_left):
            if k_left == 0:
                if remaining == 0:
                    return current
                return None
            if k_left == 1:
                # remaining = 1/d
                # d = 1/remaining
                # Check if d is integer, > min_den, and not in current
                num, den = remaining.numerator, remaining.denominator
                if num == 1:
                    d = den
                    if d > min_den and d not in current:
                        return current + [d]
                return None
            
            # Pruning: max possible sum with k_left terms starting from min_den
            # If 1/min_den * k_left < remaining, impossible
            # Actually, sum of 1/i for i from min_den to min_den+k_left-1 is an upper bound? No.
            # Simple bound: remaining < k_left / min_den
            if remaining < k_left / min_den:
                return None
            
            # Next denominator d
            # d must be > min_den
            # 1/d < remaining / k_left  => d > k_left / remaining
            # Also 1/d > remaining / 1 (if k_left=1) or remaining - sum of rest
            # Max d: 1/d > remaining - (k_left-1)/max_possible_den?
            # Simple lower bound for d: d > k_left / remaining
            # Simple upper bound for d: 1/d > remaining - sum of largest (k_left-1) fractions?
            # Actually, 1/d must be less than remaining.
            # And 1/d > remaining / k_left.
            # So d < remaining.numerator / remaining.denominator * k_left? No.
            # 1/d < remaining => d > 1/remaining.
            # 1/d > remaining / k_left => d < k_left / remaining.
            
            d_min = max(min_den, k_left // remaining.numerator * remaining.denominator + 1)
            # Wait, k_left / remaining = k_left * den / num.
            # d must be strictly less than k_left / remaining?
            # If 1/d + ... < remaining, then 1/d < remaining.
            # If 1/d is too small, we can't reach remaining with k_left terms.
            # Max sum with k_left terms starting from d is roughly k_left/d.
            # So we need k_left/d >= remaining => d <= k_left/remaining.
            d_max = (k_left * remaining.denominator) // remaining.numerator
            # d must be > min_den
            
            # Iterate d
            # To keep it fast, we limit d_max to some value if not found quickly
            # But for small k, d is small.
            
            for d in range(d_min, d_max + 1):
                if d in current:
                    continue
                new_remaining = remaining - 1/d
                if new_remaining < 0:
                    break
                res = backtrack(new_remaining, d + 1, current + [d], k_left - 1)
                if res:
                    return res
                if time.time() > deadline:
                    return None
            return None

        # Try k=5
        from fractions import Fraction
        target = Fraction(1, 1)
        res = backtrack(target, 5, [], 5)
        if res:
            return res
        
        # Try k=6
        res = backtrack(target, 5, [], 6)
        if res:
            return res
        
        # Try k=7
        res = backtrack(target, 5, [], 7)
        if res:
            return res
            
        # Fallback to the known 8-term solution if search fails or times out
        # 5, 6, 7, 8, 9, 10, 12, 14 is NOT a solution (sums to 2521/2520)
        # Let's find a valid 8-term solution or use the DP one.
        # The DP one in the prompt returned [5, 6, 7, 8, 9, 10, 12, 14]?
        # Let's check the prompt's code.
        # target=2520.
        # It finds a subset.
        # I will just return the result of the DP search if the fast search fails.
        
        # Re-run DP search as fallback
        target_val = 2520
        reached = {0: []}
        for d in range(5, target_val + 1):
            if target_val % d:
                continue
            w = target_val // d
            for total, values in list(reached.items()):
                if total + w <= target_val and total + w not in reached:
                    reached[total + w] = values + [d]
            if target_val in reached:
                return reached[target_val]
        
        raise RuntimeError("no solution found")

    return find_solution(0)
