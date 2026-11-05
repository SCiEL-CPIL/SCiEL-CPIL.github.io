"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill

    def search_k(k):
        """
        Finds k distinct denominators d_i in [5, 10^7] such that sum(1/d_i) = 1.
        Returns the list of denominators or None if not found within time limit.
        """
        found = [None]
        
        def dfs(current_sum, last_den, depth):
            if time.time() > deadline:
                return
            
            if depth == k:
                if current_sum == 1:
                    found[0] = list(current_path)
                return

            if found[0] is not None:
                return

            remaining = 1 - current_sum
            if remaining <= 0:
                return

            terms_left = k - depth
            
            # Lower bound for next denominator:
            # 1/d <= remaining => d >= ceil(1/remaining)
            # Also d > last_den
            min_d = max(last_den + 1, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            
            # Upper bound for next denominator:
            # The maximum sum of 'terms_left' distinct denominators starting from d+1 is 
            # approx terms_left / (d+1). 
            # We need remaining <= terms_left / (d+1)
            # => d+1 <= terms_left / remaining
            # => d <= terms_left / remaining - 1
            # Using exact fractions to be safe:
            # remaining <= sum_{i=1}^{terms_left} 1/(d+i)
            # A tighter bound is harder, but terms_left/(d+1) is a good upper bound on the sum.
            # So if terms_left/(d+1) < remaining, it's impossible.
            # d+1 > terms_left / remaining
            # d > terms_left / remaining - 1
            # Wait, if max possible sum < remaining, we prune.
            # Max possible sum is with smallest denominators d+1, d+2, ...
            # Sum = H_{d+terms_left} - H_d.
            # For simplicity and speed, use the bound:
            # If terms_left / (d+1) < remaining, then even the largest possible terms (smallest denominators) 
            # starting from d+1 cannot sum to 'remaining'.
            # So we require terms_left / (d+1) >= remaining
            # d+1 <= terms_left / remaining
            # d <= terms_left / remaining - 1
            
            # Calculate max_d
            # terms_left / remaining is a Fraction
            ratio = Fraction(terms_left, 1) / remaining
            # d <= ratio - 1
            max_d = ratio.numerator // ratio.denominator - 1
            
            # Also d must be <= 10^7
            max_d = min(max_d, 10000000)
            
            if min_d > max_d:
                return

            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return
                if found[0] is not None:
                    return
                
                new_sum = current_sum + Fraction(1, d)
                if new_sum > 1:
                    # Since d is increasing, 1/d is decreasing, so new_sum will only get smaller?
                    # No, we are adding 1/d. If 1/d makes it > 1, then for larger d, 1/d is smaller, 
                    # so it might not be > 1.
                    # However, we established min_d such that 1/min_d <= remaining.
                    # So new_sum <= 1 should always hold for d >= min_d.
                    # Let's double check.
                    # min_d >= ceil(1/remaining) => 1/min_d <= remaining.
                    # So current_sum + 1/min_d <= current_sum + remaining = 1.
                    # So new_sum <= 1 is guaranteed.
                    pass
                
                current_path.append(d)
                dfs(new_sum, d, depth + 1)
                current_path.pop()
                if found[0] is not None:
                    return

        current_path = []
        dfs(Fraction(0, 1), 4, 0)
        return found[0]

    # Theoretical minimum k is 8, as sum(1/5..1/12) > 1 and sum(1/5..1/11) < 1.
    # We try k from 8 upwards.
    for k in range(8, 257):
        if time.time() > deadline - 2:
            break
        result = search_k(k)
        if result is not None:
            # Verification
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result
            else:
                # Should not happen if search is correct, but handle gracefully
                continue

    # Fallback: Use a known valid solution if DFS fails (e.g. due to timeout or bug)
    # A solution with k=12 or similar from the DP approach in inspirations
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35 + 1/42 + 1/70 + 1/105 + 1/140 + 1/210
    # Let's verify this sum.
    # LCM is 420? No, 210*2=420.
    # 420/5=84, 420/6=70, 420/7=60, 420/10=42, 420/14=30, 420/21=20, 420/35=12, 420/42=10, 420/70=6, 420/105=4, 420/140=3, 420/210=2.
    # Sum: 84+70+60+42+30+20+12+10+6+4+3+2 = 343. Not 420.
    
    # Let's use the DP fallback from the original program to be safe.
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
