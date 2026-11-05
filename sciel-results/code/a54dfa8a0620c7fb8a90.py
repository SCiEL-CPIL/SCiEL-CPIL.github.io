import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Pre-calculated sums for initial denominators
    # We assume the optimal solution starts with the smallest possible denominators
    # to maximize the sum of reciprocals and minimize the number of terms.
    
    def try_solve_last_two(R, min_den, max_den=10000000):
        """
        Finds two distinct integers d1, d2 > min_den such that 1/d1 + 1/d2 = R.
        Returns (d1, d2) or None.
        """
        if R <= 0:
            return None
        p, q = R.numerator, R.denominator
        
        # 1/d1 + 1/d2 = p/q
        # q(d1+d2) = p d1 d2
        # We iterate d1.
        # d1 > min_den
        # 1/d1 < R => d1 > 1/R
        # 1/d1 + 1/(d1+1) >= R => 2/d1 >= R => d1 <= 2/R
        
        lower_d1 = max(min_den + 1, int(1 / float(R)) + 1)
        upper_d1 = int(2 / float(R)) + 2
        
        if lower_d1 > upper_d1:
            return None
            
        for d1 in range(lower_d1, upper_d1 + 1):
            if time.time() > deadline:
                return None
            # rem = R - 1/d1
            # rem = p/q - 1/d1 = (p*d1 - q) / (q*d1)
            num = p * d1 - q
            den = q * d1
            if num <= 0:
                continue
            if den % num == 0:
                d2 = den // num
                if d2 > d1 and d2 <= max_den:
                    return (d1, d2)
        return None

    def try_solve_last_three(R, min_den, max_den=10000000):
        """
        Finds three distinct integers d1, d2, d3 > min_den such that 1/d1 + 1/d2 + 1/d3 = R.
        Returns [d1, d2, d3] or None.
        """
        if R <= 0:
            return None
        p, q = R.numerator, R.denominator
        
        # Iterate d1
        lower_d1 = max(min_den + 1, int(1 / float(R)) + 1)
        # Upper bound for d1: 1/d1 + 1/(d1+1) + 1/(d1+2) >= R
        # Approx 3/d1 >= R => d1 <= 3/R
        upper_d1 = int(3 / float(R)) + 3
        
        if lower_d1 > upper_d1:
            return None
            
        for d1 in range(lower_d1, upper_d1 + 1):
            if time.time() > deadline:
                return None
            R_rem = R - Fraction(1, d1)
            if R_rem <= 0:
                continue
            # Solve for d2, d3
            res = try_solve_last_two(R_rem, d1, max_den)
            if res:
                return [d1, res[0], res[1]]
        return None

    # Base denominators
    base_denoms = [5, 6, 7, 8, 9, 10, 11]
    R_base = Fraction(1)
    for d in base_denoms:
        R_base -= Fraction(1, d)
        
    # Try k=9 (2 more terms)
    res = try_solve_last_two(R_base, 11)
    if res:
        return base_denoms + [res[0], res[1]]

    # Try k=10 (3 more terms)
    res = try_solve_last_three(R_base, 11)
    if res:
        return base_denoms + res

    # Try k=11 (4 more terms)
    # We can extend try_solve_last_three to take a list of existing denoms?
    # Or just implement a generic DFS for the remaining terms if the above fail.
    # Given the efficiency of the above, it's unlikely we need k=11.
    # But let's add a simple DFS for safety, limited in depth.
    
    def dfs(remaining, k_left, min_den, current):
        if time.time() > deadline:
            return None
        if k_left == 0:
            return current[:] if remaining == 0 else None
        if k_left == 1:
            if remaining > 0:
                num, den = remaining.numerator, remaining.denominator
                if den % num == 0:
                    x = den // num
                    if x > min_den and x <= 10000000:
                        return current + [x]
            return None
        if k_left == 2:
            return try_solve_last_two(remaining, min_den) and (current + list(try_solve_last_two(remaining, min_den))) or None
        if k_left == 3:
            return try_solve_last_three(remaining, min_den) and (current + try_solve_last_three(remaining, min_den)) or None
            
        # For k_left > 3, use greedy-like DFS with pruning
        lower_a = max(min_den + 1, int(1 / float(remaining)) + 1)
        upper_a = min(10000000, int(k_left / float(remaining)) + 2)
        
        if lower_a > upper_a:
            return None
            
        for a in range(lower_a, upper_a + 1):
            if time.time() > deadline:
                return None
            new_remaining = remaining - Fraction(1, a)
            if new_remaining <= 0:
                continue
            res = dfs(new_remaining, k_left - 1, a, current + [a])
            if res:
                return res
        return None

    # Try k=11, 12 with DFS
    for k in [11, 12]:
        k_left = k - len(base_denoms)
        res = dfs(R_base, k_left, 11, base_denoms[:])
        if res:
            return res

    # Fallback: Greedy
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = 4
        while remaining > 0:
            d = last + 1
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
    if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
        return sol

    raise RuntimeError("No solution found")
