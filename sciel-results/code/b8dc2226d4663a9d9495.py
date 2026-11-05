import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    MAX_DEN = 10000000
    MIN_DEN = 5

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

        # Lower bound for d
        # 1/d <= remaining => d >= 1/remaining
        lower_d = max(min_den, int(1 / float(remaining)) + 1)
        
        # If lower_d is too large, no solution
        if lower_d > MAX_DEN:
            return None

        d = lower_d
        
        # Iterate d
        while d <= MAX_DEN:
            if time.time() > deadline:
                return "TIMEOUT"
            
            # Prune: 
            # We need remaining - 1/d to be representable by k_left-1 terms > d.
            # Max sum of k_left-1 terms > d is sum(1/(d+i) for i in 1..k_left-1)
            # If remaining - 1/d > max_sum, then impossible.
            # As d increases, 1/d decreases, so remaining - 1/d increases.
            # Max sum decreases.
            # So if condition fails, break.
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining <= 0:
                d += 1
                continue
            
            # Calculate max possible sum for remaining k_left-1 terms
            # Sum_{i=1}^{k_left-1} 1/(d+i)
            max_sum = 0
            for i in range(1, k_left):
                max_sum += Fraction(1, d + i)
                
            if new_remaining > max_sum:
                break
            
            result = dfs(new_remaining, k_left - 1, d + 1, current + [d])
            if result == "TIMEOUT":
                return "TIMEOUT"
            if result is not None:
                return result
            
            d += 1
        
        return None

    # Search for minimal k
    # We know k >= 8
    for k in range(8, 20):
        if time.time() > deadline - 5:
            break
            
        res = dfs(Fraction(1), k, MIN_DEN, [])
        
        if res == "TIMEOUT":
            # Fallback to greedy if timeout (unlikely for small k)
            def greedy_solution():
                remaining = Fraction(1)
                denoms = []
                last = MIN_DEN - 1
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
            if s == 1 and len(set(sol)) == len(sol) and all(MIN_DEN <= x <= MAX_DEN for x in sol):
                return sol
            else:
                raise RuntimeError("Greedy fallback failed")

        if res is not None:
            # Verify
            s = sum(Fraction(1, x) for x in res)
            if s == 1 and len(set(res)) == len(res) and all(MIN_DEN <= x <= MAX_DEN for x in res):
                return res
        
        # If res is None, no solution for this k. Continue.

    # Final fallback
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = MIN_DEN - 1
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
    if s == 1 and len(set(sol)) == len(sol) and all(MIN_DEN <= x <= MAX_DEN for x in sol):
        return sol

    raise RuntimeError("No solution found")
