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
    # V(p,q) is the largest number <= N of the form p^a * q^b with a,b >= 1
    # If none exists, V=0
    # We only care about pairs where Gain > 0
    gains = [[0] * num_primes for _ in range(num_primes)]
    v_vals = [[0] * num_primes for _ in range(num_primes)]
    
    for i in range(num_primes):
        p = primes[i]
        for j in range(i + 1, num_primes):
            q = primes[j]
            # Quick check: if p*q > N, no power combination p^a q^b (a,b>=1) can be <= N
            if p * q > N:
                continue
            
            max_val = 0
            # Iterate powers of p
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

    # Solve Maximum Weight Matching
    # Using scipy.optimize.milp if available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Collect edges with positive gain
        edges = []
        for i in range(num_primes):
            for j in range(i + 1, num_primes):
                if gains[i][j] > 0:
                    edges.append((i, j, gains[i][j]))
        
        if not edges:
            # No beneficial merges
            result = [1]
            for i in range(num_primes):
                result.append(M[i])
            return [int(x) for x in result]

        num_edges = len(edges)
        
        # Variables x_k for each edge k
        # Objective: maximize sum(gain_k * x_k)
        # milp minimizes, so minimize -sum(gain_k * x_k)
        c = np.array([-e[2] for e in edges])
        
        # Constraints: for each node i, sum(x_k for k incident to i) <= 1
        # A is sparse matrix of shape (num_primes, num_edges)
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
        
        # Bounds: x_k in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Constraints: A @ x <= 1
        constraints = LinearConstraint(A, ub=np.ones(num_primes))
        
        # Integrality: x_k must be integer (0 or 1)
        integrality = np.ones(num_edges)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            matched = [False] * num_primes
            result = [1]
            for k, (i, j, g) in enumerate(edges):
                if x[k] > 0.5: # Matched
                    matched[i] = True
                    matched[j] = True
                    result.append(v_vals[i][j])
            
            for i in range(num_primes):
                if not matched[i]:
                    result.append(M[i])
            
            # Verify
            if len(set(result)) != len(result):
                # Fallback
                result = [1] + M
            else:
                valid = True
                for i in range(len(result)):
                    for j in range(i + 1, len(result)):
                        if gcd(result[i], result[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    result = [1] + M
            
            return [int(x) for x in result]
        else:
            raise Exception("MILP failed")
            
    except Exception as e:
        # Fallback to greedy if MILP fails
        # Sort edges by gain descending
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
        
        # Verify
        if len(set(result)) != len(result):
            result = [1] + M
        else:
            valid = True
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    if gcd(result[i], result[j]) != 1:
                        valid = False
                        break
            if not valid:
                result = [1] + M
        
        return [int(x) for x in result]

# EVOLVE-BLOCK-END
