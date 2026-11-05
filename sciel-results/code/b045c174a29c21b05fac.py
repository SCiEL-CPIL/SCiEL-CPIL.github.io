import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    MAX_DEN = 10000000
    
    def max_harmonic_sum(start, count):
        """
        Calculate sum_{i=0}^{count-1} 1/(start+i) using NumPy for speed.
        This is used for pruning: if this max possible sum < remaining, we can stop.
        """
        if count <= 0:
            return 0.0
        denoms = np.arange(start, start + count, dtype=np.float64)
        return float(np.sum(1.0 / denoms))

    def dfs(remaining, k_left, min_den, current):
        # Time check
        if time.time() > deadline:
            return "TIMEOUT"
        
        if k_left == 0:
            if remaining == 0:
                return current[:]
            return None
            
        if k_left == 1:
            # 1/x = remaining => x = 1/remaining
            if remaining > 0:
                num, den = remaining.numerator, remaining.denominator
                if den % num == 0:
                    x = den // num
                    if x > min_den and x <= MAX_DEN:
                        return current + [x]
            return None

        # Bounds for next denominator d
        # 1. d must be > 1/remaining => d >= floor(1/remaining) + 1
        # 2. d must be >= min_den
        lower_d = max(min_den, int(1 / float(remaining)) + 1)
        
        # Upper bound:
        # We need sum_{i=0}^{k_left-1} 1/(d+i) >= remaining
        # As d increases, this sum decreases.
        # We can estimate an upper bound.
        # A loose upper bound: k_left / d >= remaining => d <= k_left / remaining
        # But this is very loose. Let's just iterate and break when the sum is too small.
        
        # To avoid infinite loops or huge ranges, we can set a hard upper limit based on k_left/remaining
        # but rely on the max_harmonic_sum check to break early.
        
        # Optimization: If even the max sum starting from lower_d is < remaining, no solution.
        if max_harmonic_sum(lower_d, k_left) < float(remaining) - 1e-15:
            return None

        # Iterate d
        # We don't have a tight analytic upper bound for d without binary search on the harmonic sum,
        # but we can just iterate. The range is usually small due to the tight lower bound.
        
        # To be safe against huge ranges if bounds are loose, we can cap the iteration count or use binary search.
        # However, for k=9, the search tree is small.
        
        # Let's try iterating.
        # We need to be careful with the upper limit of the for loop.
        # If remaining is very small, 1/remaining is large.
        
        # Let's use a binary search or linear scan with early termination.
        # Linear scan is fine if the range is small.
        
        # Estimate upper_d:
        # We know 1/d < remaining, so d > 1/remaining.
        # Also, the sum of the remaining k_left-1 terms must be <= remaining - 1/d.
        # The max sum of k_left-1 terms starting from d+1 is approx (k_left-1)/(d+1).
        # So remaining - 1/d <= (k_left-1)/(d+1) is a necessary condition for existence? No, that's for the max.
        # We need remaining - 1/d >= min possible sum of k_left-1 terms.
        
        # Let's just iterate d from lower_d upwards.
        # We break if max_harmonic_sum(d, k_left) < remaining.
        
        # To prevent a huge loop if the lower bound is bad, we can check the condition at the start.
        
        d = lower_d
        # We need an upper bound for the loop to avoid hanging if logic fails.
        # A safe upper bound is MAX_DEN, but that's too big.
        # Let's use the loose bound: d <= k_left / remaining.
        upper_d = min(MAX_DEN, int(k_left / float(remaining)) + 1)
        
        # If lower_d > upper_d, no solution.
        if lower_d > upper_d:
            return None
        
        rem_float = float(remaining)
        
        while d <= upper_d:
            if time.time() > deadline:
                return "TIMEOUT"
            
            # Prune: if the maximum possible sum from d onwards is less than remaining, break.
            # This sum decreases as d increases.
            max_sum = max_harmonic_sum(d, k_left)
            if max_sum < rem_float - 1e-15:
                break
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining <= 0:
                d += 1
                continue
            
            # Additional prune: check if remaining k_left-1 terms can sum to new_remaining
            if new_remaining > 0:
                max_rem_sum = max_harmonic_sum(d + 1, k_left - 1)
                if max_rem_sum < float(new_remaining) - 1e-15:
                    d += 1
                    continue
            
            result = dfs(new_remaining, k_left - 1, d + 1, current + [d])
            if result == "TIMEOUT":
                return "TIMEOUT"
            if result is not None:
                return result
            
            d += 1
        
        return None

    # Search for minimal k
    # We know k >= 9
    for k in range(9, 20):
        if time.time() > deadline - 5:
            break
            
        # Try to find a solution with exactly k terms
        res = dfs(Fraction(1), k, 5, [])
        
        if res == "TIMEOUT":
            # If we timed out, we might have a partial solution or nothing.
            # Since we are searching for minimum k, and we didn't find one for this k,
            # we should probably stop and return a known valid solution if we had one, 
            # or try a larger k if we have time. But timeout means we hit the wall.
            # The task requires returning a valid solution.
            # If we time out during search, we should fallback to a greedy solution 
            # which is guaranteed to terminate quickly and produce a valid (though possibly non-minimal) result.
            
            # Let's generate a greedy solution immediately as a fallback.
            def greedy_solution():
                remaining = Fraction(1)
                denoms = []
                last = 4
                while remaining > 0:
                    # Find smallest d > last such that 1/d <= remaining
                    # 1/d <= remaining => d >= 1/remaining
                    d = max(last + 1, int(1 / float(remaining)) + 1)
                    # Ensure 1/d <= remaining
                    while Fraction(1, d) > remaining:
                        d += 1
                    denoms.append(d)
                    remaining -= Fraction(1, d)
                    last = d
                    if len(denoms) > 256:
                        break
                return denoms

            sol = greedy_solution()
            # Verify
            s = sum(Fraction(1, x) for x in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= MAX_DEN for x in sol):
                return sol
            else:
                raise RuntimeError("Greedy fallback failed")

        if res is not None:
            # Verify
            s = sum(Fraction(1, x) for x in res)
            if s == 1 and len(set(res)) == len(res) and all(5 <= x <= MAX_DEN for x in res):
                return res
        
        # If res is None, no solution found for this k within time (or search space exhausted).
        # Continue to next k.

    # Fallback if DFS didn't find a solution for small k (unlikely if bounds are good, 
    # but possible if k=9 has no solution and k=10 search is slow)
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = 4
        while remaining > 0:
            d = max(last + 1, int(1 / float(remaining)) + 1)
            while Fraction(1, d) > remaining:
                d += 1
            denoms.append(d)
            remaining -= Fraction(1, d)
            last = d
            if len(denoms) > 256:
                break
        return denoms

    sol = greedy_solution()
    s = sum(Fraction(1, x) for x in sol)
    if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= MAX_DEN for x in sol):
        return sol

    raise RuntimeError("No solution found")
