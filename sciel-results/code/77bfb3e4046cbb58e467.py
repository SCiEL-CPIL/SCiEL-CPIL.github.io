import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    MAX_DEN = 10000000
    MIN_DEN = 5

    # Precompute: what is the minimum number of terms needed?
    # For k terms with distinct denominators >= 5, max sum = 1/5 + 1/6 + ... + 1/(4+k)
    # We need this max sum >= 1 to be feasible.
    
    # Let's check feasibility for small k
    def max_sum_for_k(k, start=5):
        s = Fraction(0)
        for i in range(k):
            s += Fraction(1, start + i)
        return s

    # Find minimum feasible k
    min_feasible_k = 1
    while min_feasible_k < 256:
        if max_sum_for_k(min_feasible_k) >= 1:
            break
        min_feasible_k += 1

    # For k terms, 1/a1 + ... + 1/ak = 1 with 5 <= a1 < ... < ak
    # DFS with strong pruning
    
    def dfs(remaining, k_left, min_den, current, depth=0):
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

        # Lower bound for next denominator d:
        # We need 1/d < remaining (since there are k_left-1 more positive terms)
        # So d > 1/remaining
        # Also d >= min_den
        lower_d = max(min_den, int(1 / float(remaining)) + 1)
        
        # Upper bound: remaining <= 1/d + 1/(d+1) + ... + 1/(d+k_left-1)
        # The max sum of k_left terms starting from d is H(d+k_left-1) - H(d-1)
        # We need this >= remaining
        # As d increases, this max sum decreases
        # So find the largest d where max_sum >= remaining
        
        # Also: remaining >= 1/d (the largest single term)
        # And remaining < k_left / d (sum of k_left terms each <= 1/d)
        # So d < k_left / remaining
        upper_d = min(MAX_DEN, int(k_left / float(remaining)) + 1)
        
        if lower_d > upper_d:
            return None

        # Additional pruning: the minimum possible sum of k_left distinct terms
        # all >= d is 1/d + 1/(d+1) + ... + 1/(d+k_left-1)
        # If this is less than remaining, we can't reach it with these d values
        # But as d increases, this sum decreases, so if it fails at some d,
        # it fails for all larger d -> break
        
        d = lower_d
        while d <= upper_d:
            if time.time() > deadline:
                return "TIMEOUT"
            
            # Prune: max sum of k_left terms starting at d
            # = 1/d + 1/(d+1) + ... + 1/(d+k_left-1)
            max_possible = Fraction(0)
            for i in range(k_left):
                max_possible += Fraction(1, d + i)
            
            if max_possible < remaining:
                break  # For larger d, max_possible only decreases
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining <= 0:
                d += 1
                continue
            
            result = dfs(new_remaining, k_left - 1, d + 1, current + [d], depth + 1)
            if result == "TIMEOUT":
                return "TIMEOUT"
            if result is not None:
                return result
            
            d += 1
        
        return None

    # Search for minimal k starting from min_feasible_k
    for k in range(min_feasible_k, min(min_feasible_k + 15, 257)):
        if time.time() > deadline - 5:
            break
            
        res = dfs(Fraction(1), k, MIN_DEN, [])
        
        if res == "TIMEOUT":
            # Fallback to greedy
            def greedy_solution():
                remaining = Fraction(1)
                denoms = []
                last = MIN_DEN - 1
                while remaining > 0 and len(denoms) < 256:
                    d = max(last + 1, int(1 / float(remaining)) + 1)
                    while Fraction(1, d) > remaining:
                        d += 1
                    if d > MAX_DEN:
                        break
                    denoms.append(d)
                    remaining -= Fraction(1, d)
                    last = d
                return denoms

            sol = greedy_solution()
            s = sum(Fraction(1, x) for x in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(MIN_DEN <= x <= MAX_DEN for x in sol):
                return sol
            raise RuntimeError("Greedy fallback failed")

        if res is not None:
            # Verify
            s = sum(Fraction(1, x) for x in res)
            if s == 1 and len(set(res)) == len(res) and all(MIN_DEN <= x <= MAX_DEN for x in res):
                return res

    # Final fallback: greedy
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = MIN_DEN - 1
        while remaining > 0 and len(denoms) < 256:
            d = max(last + 1, int(1 / float(remaining)) + 1)
            while Fraction(1, d) > remaining:
                d += 1
            if d > MAX_DEN:
                break
            denoms.append(d)
            remaining -= Fraction(1, d)
            last = d
        return denoms

    sol = greedy_solution()
    s = sum(Fraction(1, x) for x in sol)
    if s == 1 and len(set(sol)) == len(sol) and all(MIN_DEN <= x <= MAX_DEN for x in sol):
        return sol

    raise RuntimeError("No solution found")
