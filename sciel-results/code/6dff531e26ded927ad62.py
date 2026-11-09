"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    def dfs(rem, min_d, k_rem, path):
        if time.time() > deadline:
            return None

        if k_rem == 0:
            if rem == 0:
                return path
            return None

        if k_rem == 1:
            if rem > 0:
                d = 1 / rem
                # Check if d is integer
                if d.denominator == 1:
                    d_int = d.numerator
                    if d_int >= min_d and d_int <= 10000000:
                        return path + [d_int]
            return None

        # Pruning: Max possible sum with k_rem terms starting from min_d
        # If this sum is less than rem, it's impossible.
        max_possible = Fraction(0, 1)
        for i in range(k_rem):
            max_possible += Fraction(1, min_d + i)
        
        if max_possible < rem:
            return None

        # Determine range for d
        # d must be >= min_d
        # 1/d <= rem => d >= 1/rem
        d_min = max(min_d, (1 / rem).numerator + (1 / rem).denominator - 1) // (1 / rem).denominator
        # Simpler: d_min = max(min_d, int(1/rem) + 1) is not quite right if 1/rem is integer.
        # If rem = 1/5, d=5. 1/rem = 5. d >= 5.
        # If rem = 1/4, d=4.
        # d >= ceil(1/rem).
        inv_rem = 1 / rem
        if inv_rem.denominator == 1:
            d_min = max(min_d, inv_rem.numerator)
        else:
            d_min = max(min_d, (inv_rem.numerator // inv_rem.denominator) + 1)

        # Upper bound for d:
        # rem - 1/d must be <= max sum of next (k_rem - 1) terms starting from d+1
        # Let max_next(d) = sum(1/(d+1+i) for i in range(k_rem-1))
        # We need rem - 1/d <= max_next(d)
        # 1/d >= rem - max_next(d)
        # Since max_next(d) depends on d, we can iterate d and break if condition fails.
        # Or estimate d_max.
        # For small k_rem, iteration is fine.
        
        # We also know d must be such that 1/d < rem (strictly, unless k_rem=1 which is handled)
        # Actually 1/d can be equal to rem if k_rem=1.
        
        # Let's iterate d from d_min upwards.
        # To avoid infinite loops or huge ranges, we can set a heuristic d_max.
        # If d is very large, 1/d is small.
        # The condition rem - 1/d <= max_next(d) will eventually fail if d is too large?
        # No, if d is large, 1/d is small, rem - 1/d is close to rem.
        # max_next(d) is approx (k_rem-1)/d.
        # So we need rem <= (k_rem-1)/d => d <= (k_rem-1)/rem.
        d_max = int((k_rem - 1) / rem)
        if d_max < d_min:
            return None

        # Cap d_max to reasonable range or 10^7
        d_max = min(d_max, 10000000)

        for d in range(d_min, d_max + 1):
            # Check if rem - 1/d is representable
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            
            # Quick check: max sum of remaining k_rem-1 terms starting from d+1
            # If new_rem > max_next, skip
            # max_next = sum(1/(d+1+i) for i in range(k_rem-1))
            # This is expensive to compute every time. 
            # But k_rem is small.
            
            result = dfs(new_rem, d + 1, k_rem - 1, path + [d])
            if result is not None:
                return result
        
        return None

    # Try increasing k from 7
    # It is known that k=7 is the minimum possible for distinct denominators >= 5
    # (k=6 max sum < 1)
    for k in range(7, 15):
        target = Fraction(1, 1)
        result = dfs(target, 5, k, [])
        if result is not None:
            # Validate
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: return a known valid solution if search fails
    # One known solution with k=7 is {5, 6, 7, 8, 9, 10, 420}? No, checked above.
    # Let's try to find one or use a larger k solution.
    # A known solution for k=10:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + ...
    # Actually, let's just raise an error if not found, but the prompt says "invalid result scores 0".
    # So we must return something valid.
    # The DP approach in the original code always finds a solution (suboptimal k).
    # Let's add the DP fallback to ensure validity.
    
    target_dp, reached = 2520, {0: []}
    for d in range(5, target_dp + 1):
        if target_dp % d:
            continue
        w = target_dp // d
        for total, values in list(reached.items()):
            if total + w <= target_dp and total + w not in reached:
                reached[total + w] = values + [d]
        if target_dp in reached:
            return reached[target_dp]
            
    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
