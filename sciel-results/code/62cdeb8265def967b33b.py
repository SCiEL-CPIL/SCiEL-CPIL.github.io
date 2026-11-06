import math
import time

T12 = math.factorial(12)
# 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
PRIMES = [2, 3, 5, 7, 11]
TOTAL_EXP = [10, 5, 2, 1, 1]

def exp_to_int(e):
    r = 1
    for i, p in enumerate(PRIMES):
        r *= p ** e[i]
    return r

def int_to_exp(n):
    e = [0]*5
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            e[i] += 1
            n //= p
    if n != 1:
        return None
    return e

def valid_solution(factors):
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f < 13 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12

def construct():
    deadline = time.time() + 85
    best_max = T12
    best_factors = [T12]
    
    # Precompute all valid factor values: integers > 12 that divide 12!
    # Their exponent vectors are sub-vectors of TOTAL_EXP
    all_factors = []  # list of (value, exp_vector)
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        v = (2**e2)*(3**e3)*(5**e5)*(7**e7)*(11**e11)
                        if v > 12 and v < T12:  # exclude 1 and T12 itself
                            all_factors.append((v, [e2, e3, e5, e7, e11]))
    all_factors.sort()
    all_factors_by_val = {v: e for v, e in all_factors}
    
    # Initial solution: try a few hand-crafted ones
    initial_candidates = [
        [144, 64, 77, 25, 27],
        [144, 64, 77, 25, 27],
    ]
    for cand in initial_candidates:
        if valid_solution(cand):
            if max(cand) < best_max:
                best_max = max(cand)
                best_factors = cand[:]
    
    # DFS with exponent vectors
    def dfs(rem_exp, min_val, count, factors, best_max_ref):
        nonlocal best_max, best_factors
        if time.time() > deadline:
            return
        
        # Check if rem_exp is zero
        if all(x == 0 for x in rem_exp):
            if count >= 1:
                cur_max = max(factors)
                if cur_max < best_max:
                    best_max = cur_max
                    best_factors = factors[:]
            return
        
        if count >= 12:
            return
        
        # Generate all sub-vectors of rem_exp (non-zero) corresponding to values > min_val
        # and value < current best_max (if count > 0, the new factor must be < best_max to improve)
        # Actually, the new factor must be > min_val and the remaining must be decomposable
        
        # Lower bound: if we need at least 1 more factor, and the smallest possible is min_val+1
        # The product of remaining must be >= (min_val+1) if count+1 is the last factor
        # More precisely: rem product = exp_to_int(rem_exp)
        rem_prod = 1
        for i, p in enumerate(PRIMES):
            rem_prod *= p ** rem_exp[i]
        
        # If rem_prod <= min_val, we can't form a factor > min_val
        if rem_prod <= min_val:
            return
        
        # Try each possible next factor: a sub-vector of rem_exp with value > min_val
        # and value < best_max (to improve)
        # We iterate over all sub-vectors
        def gen_subvectors(rem_exp):
            """Yield all non-zero sub-vectors of rem_exp."""
            def rec(i, cur):
                if i == 5:
                    if any(c > 0 for c in cur):
                        yield cur[:]
                    return
                for k in range(rem_exp[i] + 1):
                    cur[i] = k
                    yield from rec(i+1, cur)
                    cur[i] = 0
            yield from rec(0, [0]*5)
        
        # To improve efficiency, we'll collect valid candidates and sort by value
        candidates = []
        for sub in gen_subvectors(rem_exp):
            v = exp_to_int(sub)
            if v <= min_val or v >= T12:
                continue
            # v must be < best_max if we want to improve (unless this is the last factor)
            # Actually, even if v >= best_max, it might be the last factor and give a valid solution
            # But we only care about improving, so if v >= best_max and count+1 < total, skip
            # For now, collect all and filter later
            candidates.append((v, sub))
        
        candidates.sort()
        
        for v, sub in candidates:
            if time.time() > deadline:
                return
            if v >= best_max and count + 1 < 12:
                # If this isn't the last factor, and v >= best_max, no improvement possible
                # But it could be the last factor... let's check
                new_rem = [rem_exp[i] - sub[i] for i in range(5)]
                if any(x > 0 for x in new_rem):
                    # Not the last factor, so v >= best_max means no improvement
                    continue
                # If it IS the last factor, we can still check
            if v >= best_max and all(x == 0 for x in [rem_exp[i] - sub[i] for i in range(5)]):
                # This is the last factor and v >= best_max, no improvement
                continue
            
            new_rem = [rem_exp[i] - sub[i] for i in range(5)]
            dfs(new_rem, v, count + 1, factors + [v], best_max_ref)
    
    dfs(TOTAL_EXP, 12, 0, [], None)
    
    if best_factors and valid_solution(best_factors):
        return best_factors
    return [T12]
