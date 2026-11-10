import math
import random
import numpy as np

# Check for SciPy MILP availability
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

def construct():
    if HAS_MILP:
        return construct_milp()
    else:
        return construct_fallback()

def construct_milp():
    # 1. Generate primes < 512
    limit = 512
    is_prime = [True] * limit
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    num_primes = len(primes)
    num_vars = 511 # 1 to 511
    
    # 2. Build Constraint Matrix
    # A[i, a] = 1 if primes[i] divides (a+1)
    A = lil_matrix((num_primes, num_vars), dtype=np.int8)
    for i, p in enumerate(primes):
        for a in range(1, 512):
            if a % p == 0:
                A[i, a-1] = 1
                
    # 3. Objective
    # Maximize sum 1/(512-a) x_a
    # Minimize -sum 1/(512-a) x_a
    c = np.zeros(num_vars)
    for a in range(1, 512):
        c[a-1] = -1.0 / (512 - a)
        
    # 4. Solve
    # Constraints: A @ x <= 1
    constraint = LinearConstraint(A, lb=None, ub=np.ones(num_primes))
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(num_vars) # All integer
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.success:
        x = res.x
        chosen = []
        for i in range(num_vars):
            if x[i] > 0.5: # Threshold for binary
                chosen.append(i + 1)
        return chosen
    else:
        # Fallback if solver fails
        return construct_fallback()

def construct_fallback():
    # Original code
    candidates = list(range(511, 0, -1))
    chosen = []
    for a in candidates:
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
       
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
       
    def is_coprime_to_set(x, S):
        for y in S:
            if math.gcd(x, y) > 1:
                return False
        return True
           
    def get_score(S):
        return sum(weights[a] for a in S)
           
    current_score = get_score(chosen)
    improved = True
    max_iters = 1000
    iter_count = 0
       
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
           
        for i, a_out in enumerate(chosen):
            rest = [chosen[j] for j in range(len(chosen)) if j != i]
               
            best_in = -1
            best_gain = 0
               
            for a_in in candidates:
                if a_in in chosen:
                    continue
                if weights[a_in] <= weights[a_out]:
                    continue
                   
                if is_coprime_to_set(a_in, rest):
                    gain = weights[a_in] - weights[a_out]
                    if gain > best_gain:
                        best_gain = gain
                        best_in = a_in
                           
            if best_in != -1:
                chosen[i] = best_in
                current_score += best_gain
                improved = True
                break
           
    return chosen
