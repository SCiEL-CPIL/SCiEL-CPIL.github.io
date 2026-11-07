import time
from math import isqrt, gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Calculate M_p: largest power of p <= N
    M = [0] * num_primes
    for i, p in enumerate(primes):
        power = p
        while power * p <= N:
            power *= p
        M[i] = power

    # Calculate V(p,q) and Gain G(p,q)
    gains = [[0] * num_primes for _ in range(num_primes)]
    v_vals = [[0] * num_primes for _ in range(num_primes)]
    
    for i in range(num_primes):
        p = primes[i]
        for j in range(i + 1, num_primes):
            q = primes[j]
            if p * q > N:
                continue
            
            max_val = 0
            pa = p
            while pa <= N:
                pb = q
                while pa * pb <= N:
                    if pa * pb > max_val:
                        max_val = pa * pb
                    pb *= q
                pa *= p
            
            if max_val > 0:
                v_vals[i][j] = max_val
                v_vals[j][i] = max_val
                gain = max_val - M[i] - M[j]
                if gain > 0:
                    gains[i][j] = gain
                    gains[j][i] = gain

    # Precompute all numbers 1..256 with their prime factorization
    # For each number, store the set of prime indices it's divisible by
    num_prime_factors = [[] for _ in range(N + 1)]
    for n in range(2, N + 1):
        for i, p in enumerate(primes):
            if p > n:
                break
            if n % p == 0:
                num_prime_factors[n].append(i)
    
    # Solve Maximum Weight Matching
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        edges = []
        for i in range(num_primes):
            for j in range(i + 1, num_primes):
                if gains[i][j] > 0:
                    edges.append((i, j, gains[i][j]))
        
        if not edges:
            result = [1]
            for i in range(num_primes):
                result.append(M[i])
            return [int(x) for x in result]

        num_edges = len(edges)
        
        c = np.array([-e[2] for e in edges])
        
        rows = []
        cols = []
        vals = []
        
        for k, (i, j, g) in enumerate(edges):
            rows.append(i)
            cols.append(k)
            vals.append(1)
            rows.append(j)
            cols.append(k)
            vals.append(1)
            
        A = sp.csc_matrix((vals, (rows, cols)), shape=(num_primes, num_edges))
        
        bounds = Bounds(lb=0, ub=1)
        constraints = LinearConstraint(A, ub=np.ones(num_primes))
        integrality = np.ones(num_edges)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            matched = [False] * num_primes
            result = [1]
            for k, (i, j, g) in enumerate(edges):
                if x[k] > 0.5:
                    matched[i] = True
                    matched[j] = True
                    result.append(v_vals[i][j])
            
            for i in range(num_primes):
                if not matched[i]:
                    result.append(M[i])
        else:
            raise Exception("MILP failed")
            
    except Exception as e:
        edges = []
        for i in range(num_primes):
            for j in range(i + 1, num_primes):
                if gains[i][j] > 0:
                    edges.append((gains[i][j], i, j))
        
        edges.sort(reverse=True)
        
        matched = [False] * num_primes
        result = [1]
        
        for g, i, j in edges:
            if not matched[i] and not matched[j]:
                matched[i] = True
                matched[j] = True
                result.append(v_vals[i][j])
        
        for i in range(num_primes):
            if not matched[i]:
                result.append(M[i])
    
    # Local search polish: try to replace elements with larger coprime alternatives
    def is_valid_set(s):
        s_set = set(s)
        if len(s_set) != len(s):
            return False
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    def local_search_polish(result, max_time=5.0):
        """Try to improve the solution by replacing elements with larger coprime ones."""
        current_sum = sum(result)
        result_set = set(result)
        
        # Build a list of candidate numbers not in result, sorted descending
        candidates = [n for n in range(N, 0, -1) if n not in result_set and n != 1]
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            for idx in range(len(result)):
                if (time.time() - start_time) > deadline:
                    break
                old_val = result[idx]
                old_factors = num_prime_factors[old_val]
                
                # Try to find a larger number that is coprime to all other elements
                for new_val in candidates:
                    if new_val <= old_val:
                        break
                    # Check if new_val is coprime to all other elements in result
                    new_factors = num_prime_factors[new_val]
                    
                    # Quick check: if new_val shares a prime with old_val, skip
                    # (since old_val is being removed, this is fine, but we need to check others)
                    
                    compatible = True
                    for j in range(len(result)):
                        if j == idx:
                            continue
                        other_val = result[j]
                        if gcd(new_val, other_val) != 1:
                            compatible = False
                            break
                    
                    if compatible:
                        # Replace and check validity
                        result[idx] = new_val
                        if is_valid_set(result):
                            current_sum = sum(result)
                            improved = True
                            break
                        else:
                            result[idx] = old_val
        
        return result
    
    # Apply local search polish
    if (time.time() - start_time) < deadline:
        result = local_search_polish(result)
    
    # Final verification
    if not is_valid_set(result):
        # Fallback to baseline
        result = [1] + M
    
    return [int(x) for x in result]
