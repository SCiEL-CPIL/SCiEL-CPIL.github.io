"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Fallback greedy solution (from previous version)
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    slots = list(primes)
    while True:
        if time.time() > deadline - 5:
            break
        best_gain = 0
        best_i = -1
        best_j = -1
        n = len(slots)
        for i in range(n):
            for j in range(i + 1, n):
                merged = slots[i] * slots[j]
                if merged > MAX_N:
                    continue
                gain = merged - slots[i] - slots[j]
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        if best_i == -1 or best_gain <= 0:
            break
        slots[best_i] = slots[best_i] * slots[best_j]
        slots.pop(best_j)
    fallback_result = slots + [1]
    # Verify fallback
    for i in range(len(fallback_result)):
        for j in range(i + 1, len(fallback_result)):
            if gcd(fallback_result[i], fallback_result[j]) != 1:
                fallback_result = [1] + primes
                break
    
    best_sum = sum(fallback_result)
    best_set = fallback_result

    try:
        # Attempt ILP solution
        # Variables x_1 ... x_256
        # Maximize sum(i * x_i)
        # Constraints: x_i + x_j <= 1 for gcd(i,j) > 1
        
        # Build conflict constraints
        A_rows = []
        # We can use sparse matrix for efficiency
        # But for 256 vars, dense might be okay if not too many constraints.
        # Let's count constraints first or just build.
        
        # Number of pairs is ~32k.
        # Let's build the constraint matrix.
        # Using scipy.sparse for safety if SciPy milp supports it.
        from scipy.sparse import lil_matrix
        
        M = MAX_N
        # x_i corresponds to integer i+1
        # Objective: c_i = i+1
        c = np.arange(1, M + 1)
        
        # Constraints: A x <= 1
        # A is sparse
        # We need to identify all pairs (i, j) with i < j and gcd(i+1, j+1) > 1
        # Let's iterate and add rows
        
        rows = []
        cols = []
        data = []
        
        # Optimization: Precompute prime factors or just use gcd.
        # GCD is fast enough for 256*256.
        for i in range(M):
            val_i = i + 1
            for j in range(i + 1, M):
                val_j = j + 1
                if gcd(val_i, val_j) > 1:
                    # Constraint: x_i + x_j <= 1
                    rows.append(len(rows))
                    rows.append(len(rows))
                    cols.append(i)
                    cols.append(j)
                    data.append(1)
                    data.append(1)
        
        # If no constraints, all can be 1? No, 1 is coprime to all, but others conflict.
        # Actually, if no conflicts, sum is sum(1..256). But there are conflicts.
        
        # Create sparse matrix
        # Number of constraints is len(data) // 2
        num_constraints = len(data) // 2
        A = lil_matrix((num_constraints, M))
        
        for k in range(num_constraints):
            A[k, cols[2*k]] = 1
            A[k, cols[2*k+1]] = 1
        
        # Convert to CSR for milp
        A_csr = A.tocsr()
        
        # Linear constraint: A x <= 1
        # milp expects LinearConstraint(A, lb, ub)
        # We want A x <= 1, so lb = -inf, ub = 1
        lincons = LinearConstraint(A_csr, lb=-np.inf, ub=1)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all variables integer
        integrality = np.ones(M)
        
        # Solve
        res = milp(c=c, constraints=[lincons], integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # res.x is the solution
            x_sol = res.x
            # Select integers where x_i is close to 1
            ilp_set = []
            for i in range(M):
                if x_sol[i] > 0.5:
                    ilp_set.append(i + 1)
            
            # Verify coprimality
            valid = True
            for i in range(len(ilp_set)):
                for j in range(i + 1, len(ilp_set)):
                    if gcd(ilp_set[i], ilp_set[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                ilp_sum = sum(ilp_set)
                if ilp_sum > best_sum:
                    best_sum = ilp_sum
                    best_set = ilp_set
        
    except Exception:
        pass

    # Final safety check on best_set
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            if gcd(best_set[i], best_set[j]) != 1:
                # Fallback to primes + 1 if something went wrong
                return [1] + primes

    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
