import math
import time

def construct():
    N = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    # 1. Prime factorization of N
    # 12! = 2^8 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [8, 5, 2, 1, 1]
    target_vec = tuple(exponents)
    
    # 2. Get all divisors of N greater than 12
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            other = N // i
            if other != i and other > 12:
                divisors.append(other)
        i += 1
    
    divisors.sort()
    
    # Helper to get exponent vector of a divisor
    def get_vec(d):
        vec = [0] * 5
        for p, exp in zip(primes, exponents):
            count = 0
            while d % p == 0:
                d //= p
                count += 1
            vec[exp_index] = count # Wait, need index mapping
        return tuple(vec)

    # Correct get_vec
    def get_vec_correct(d):
        vec = [0] * len(primes)
        for idx, p in enumerate(primes):
            count = 0
            while d % p == 0:
                d //= p
                count += 1
            vec[idx] = count
        return tuple(vec)
        
    # Precompute vectors for all divisors
    div_vectors = [(d, get_vec_correct(d)) for d in divisors]
    
    # 3. Iterate M in increasing order
    # We maintain a list of candidates <= M
    # Actually, we can just iterate through divisors and add them one by one to the DP
    # But we need to check if a solution exists with MAX <= current M.
    # If we add divisors in increasing order, after adding d_k, 
    # we have considered all divisors <= d_k.
    # If a solution exists using only divisors <= d_k, it will be found.
    
    # DP State: Set of reachable exponent vectors
    # To allow backtracking, we store the history of reachable sets
    # history[i] = set of vectors reachable using first i divisors
    
    history = [set()]
    history[0].add(tuple([0]*len(primes)))
    
    # We also need to store which divisor was used to reach each state?
    # Or we can just backtrack using the history sets.
    
    for idx, (d, vec) in enumerate(div_vectors):
        if time.time() > deadline:
            break
            
        current_reachable = history[-1]
        next_reachable = set(current_reachable)
        
        for v in current_reachable:
            new_v = tuple(v[i] + vec[i] for i in range(len(primes)))
            # Check if new_v <= target_vec
            valid = True
            for i in range(len(primes)):
                if new_v[i] > target_vec[i]:
                    valid = False
                    break
            if valid:
                next_reachable.add(new_v)
                
        history.append(next_reachable)
        
        # Check if target is reachable
        if target_vec in next_reachable:
            # Reconstruct solution
            result_factors = []
            v = target_vec
            # Backtrack from idx+1 to 1
            for i in range(idx + 1, 0, -1):
                # v is reachable using first i divisors
                # Check if v is reachable using first i-1 divisors
                if v in history[i-1]:
                    # Divisor i was NOT used
                    continue
                else:
                    # Divisor i WAS used
                    d_used = div_vectors[i-1][0]
                    vec_used = div_vectors[i-1][1]
                    result_factors.append(d_used)
                    v = tuple(v[j] - vec_used[j] for j in range(len(primes)))
            
            # Validate
            if len(result_factors) > 0:
                prod = 1
                for f in result_factors:
                    prod *= f
                if prod == N and all(f > 12 for f in result_factors) and len(set(result_factors)) == len(result_factors):
                    return result_factors
            # If validation fails, continue search (should not happen)
            # But to be safe, we can clear history or continue?
            # If it fails, it means a bug in logic.
            # Let's just return the best found so far or fallback.
            # Actually, if target is in set, reconstruction should work.
            # If it doesn't, we break and fallback.
            break
            
    # Fallback
    return [N]
