import math
import time
import sys
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    
    # 1. Identify all candidate numbers 1..N
    # We will include 1 in the solution if it doesn't conflict (it never does).
    # Actually, 1 is coprime to everything, so x_1 should always be 1 if it maximizes sum.
    # Since 1 > 0, we will always include it.
    # So we can fix x_1 = 1 and solve for 2..N.
    
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # 2. Build the conflict matrix
    # We need a matrix A such that A @ x <= 1 for all conflicting pairs.
    # Instead of building a dense matrix (n x n), we can list the constraints.
    # SciPy's milp accepts LinearConstraint with A_ub.
    # Number of constraints is O(N^2). 256*256/2 = ~32k constraints.
    # This is small enough for HiGHS.
    
    # Optimization: Pre-compute prime factors or just use gcd.
    # GCD is fast.
    
    # Let's collect constraints (i, j) where gcd(candidates[i], candidates[j]) > 1
    # Note: candidates are 2..256.
    
    # We can build the A_ub matrix directly.
    # To save memory and time, we can be smart.
    # But 32k constraints * 255 variables is < 1M floats. Trivial for numpy.
    
    rows = []
    cols = []
    vals = []
    
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            cj = candidates[j]
            if math.gcd(ci, cj) > 1:
                rows.append(len(rows))
                cols.append(i)
                vals.append(1)
                cols.append(j)
                vals.append(1)
    
    if not rows:
        # No conflicts? Unlikely, but handle it.
        # Return all candidates + 1
        return [1] + candidates
    
    # Construct sparse matrix for A_ub
    # scipy.sparse is better for large, but dense is fine for 255x32k?
    # Actually, A_ub shape is (num_constraints, n_vars).
    # num_constraints ~ 32000. n_vars = 255.
    # 32000 * 255 = 8,160,000 elements.
    # This is fine as a dense numpy array, but sparse is more efficient for solvers.
    # scipy.optimize.milp accepts scipy.sparse matrices.
    
    import scipy.sparse as sp
    
    # The rows list above is just indices. Let's build the sparse matrix properly.
    # We need (row, col) pairs for non-zero entries.
    # My loop above appended to lists in a way that's not directly CSR/CSC.
    # Let's restart the matrix building cleanly.
    
    row_idx = []
    col_idx = []
    data = []
    
    constraint_id = 0
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            cj = candidates[j]
            if math.gcd(ci, cj) > 1:
                row_idx.append(constraint_id)
                col_idx.append(i)
                data.append(1)
                
                row_idx.append(constraint_id)
                col_idx.append(j)
                data.append(1)
                
                constraint_id += 1
                
                # Time check
                if time.time() > deadline - 5.0:
                    # Fallback to a simple heuristic if building takes too long
                    # (It shouldn't, but safety first)
                    return _fallback_construct(N)
    
    num_constraints = constraint_id
    if num_constraints == 0:
        return [1] + candidates
        
    A_ub = sp.csr_matrix((data, (row_idx, col_idx)), shape=(num_constraints, n))
    
    # 3. Define the MILP problem
    # Maximize sum(i * x_i)
    # This is equivalent to Minimize -sum(i * x_i)
    c = np.array([-float(ci) for ci in candidates])
    
    # Constraints:
    # 1. A_ub @ x <= 1 (conflicts)
    # 2. 0 <= x_i <= 1 (bounds)
    
    # LinearConstraint for A_ub @ x <= 1
    # lb = 0, ub = 1 for each row
    lincons = LinearConstraint(A_ub, lb=0, ub=1)
    
    # Bounds for x: [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x must be integer (0 or 1)
    integrality = np.ones(n)
    
    # 4. Solve
    try:
        res = milp(c=c, constraints=lincons, integrality=integrality, bounds=bounds)
        
        if time.time() > deadline:
            return _fallback_construct(N)
            
        if not res.success:
            # If solver fails, fallback
            return _fallback_construct(N)
            
        x = res.x
        # Round to nearest integer to handle floating point noise
        x = np.round(x).astype(int)
        
        # 5. Construct result
        result = [1]  # Always include 1
        
        for i in range(n):
            if x[i] == 1:
                result.append(candidates[i])
        
        # Sanity check: Verify pairwise coprimality
        # If invalid, fallback
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) != 1:
                    return _fallback_construct(N)
                    
        return result
        
    except Exception as e:
        # In case of any error, fallback
        return _fallback_construct(N)

def _fallback_construct(N):
    """
    Fallback heuristic if ILP fails or times out.
    Uses the logic from the current program which is decent (score ~1.11).
    """
    import math
    N = 256
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    # Heuristic:
    # 1. Start with prime powers for small primes, but group 2 with 11, 5 with 7.
    # 2. Keep other primes as themselves.
    # This mimics the "Option A" from the current code comments.
    
    # Primes > 17 are singletons.
    large_primes = [p for p in primes if p > 17]
    
    # Small primes handling:
    # 242 (2, 11)
    # 245 (5, 7)
    # 243 (3)
    # 13
    # 17
    # Check if these are valid and distinct.
    # 242 = 2 * 11^2. GCD(242, 245)=1. GCD(242, 243)=1.
    # 245 = 5 * 7^2.
    # 243 = 3^5.
    # 13, 17 are primes.
    # All distinct? 242, 245, 243, 13, 17. Yes.
    # Are they <= 256? Yes.
    
    selected = [1, 242, 245, 243, 13, 17] + large_primes
    
    # Verify
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # If this specific hard-coded set fails (it shouldn't), return all primes
                return [1] + primes
                
    return selected

# Ensure the main entry point is construct
# The problem asks to keep function names and signatures.
# The seed program had `def construct():`
# We have defined `construct` and `_fallback_construct`.

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print(f"Sum: {sum(res)}, Count: {len(res)}")
