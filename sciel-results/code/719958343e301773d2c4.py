# EVOLVE-BLOCK-START
def construct():
    from math import factorial
    from functools import lru_cache

    target = factorial(12)
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    target_state = tuple(target_exp)

    def gen_divisors(exp_bounds, idx=0, current=1):
        if idx == len(exp_bounds):
            return [current]
        result = []
        for e in range(exp_bounds[idx] + 1):
            result.extend(gen_divisors(exp_bounds, idx + 1, current * primes[idx] ** e))
        return result

    all_divisors = gen_divisors(target_exp)
    candidates = [d for d in all_divisors if d > 12]
    candidates.sort()

    def exp_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    cand_vecs = [exp_vector(c) for c in candidates]
    
    # dp[state] = (min_max_factor, prev_state, cand_index)
    # We want to minimize the max factor.
    # Initialize with infinity
    INF = float('inf')
    dp = {tuple([0]*5): (0, None, -1)}
    
    # Process candidates in increasing order
    for i, c in enumerate(candidates):
        cv = cand_vecs[i]
        # Iterate over a copy of the current states to avoid using the same candidate twice
        # and to allow updates based on previous states
        current_states = list(dp.keys())
        for state in current_states:
            min_max, prev_state, prev_idx = dp[state]
            new_state = tuple(state[j] + cv[j] for j in range(5))
            if all(new_state[j] <= target_exp[j] for j in range(5)):
                new_max = max(min_max, c)
                if new_state not in dp or dp[new_state][0] > new_max:
                    dp[new_state] = (new_max, state, i)

    if target_state not in dp:
        return [target]

    # Reconstruct
    result_indices = []
    state = target_state
    while state != tuple([0]*5):
        min_max, prev_state, cand_idx = dp[state]
        result_indices.append(cand_idx)
        state = prev_state
    
    result = [candidates[i] for i in result_indices]
    
    # Verify
    prod = 1
    for r in result:
        prod *= r
    if prod != target or len(result) != len(set(result)) or any(r <= 12 for r in result):
        return [target]
        
    return result

# EVOLVE-BLOCK-END
