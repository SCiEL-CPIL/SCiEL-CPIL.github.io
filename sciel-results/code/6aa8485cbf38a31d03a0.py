import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x7 matrix.
    
    Strategy:
    1. Define a deadline to ensure termination within limits.
    2. Attempt to find a decomposition of rank r < 105 (schoolbook) using random search.
       - Randomly generate V and W with small integer entries.
       - For each i in [0, n), solve the linear system for U[:, i*m:(i+1)*m].
       - Check if U entries are close to half-integers.
       - Verify the full decomposition exactly using Fractions.
    3. If no lower rank is found, return the schoolbook decomposition of rank 105.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s/150s limit
    
    n, m, p = DIMENSIONS
    nmp = n * m * p
    
    # Helper to verify with numpy (fast check)
    def verify_np(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Helper to verify with fractions (exact check)
    def verify_fraction(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank(r, num_attempts=100):
        if time.time() > deadline:
            return None
            
        # Use a fixed seed for reproducibility in the search, but vary it per rank/attempt
        # We want diversity, so we use random seeds but limit the scope.
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random V and W with entries in {-1, 0, 1}
            # This keeps the numbers small and increases the chance of rational solutions
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
            
            # Check if V and W are "useful" (not all zeros, etc.)
            if all(val == 0 for row in V for val in row) or all(val == 0 for row in W for val in row):
                continue

            U = [[0]*r for _ in range(n*m)]
            valid = True
            
            for i in range(n):
                # Unknowns: U[i*m+j, c] for j in range(m), c in range(r)
                # Total unknowns: m * r
                # Equations: for j in range(m), k in range(p):
                #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Total equations: m * p
                
                A_mat = np.zeros((m * p, m * r))
                b_vec = np.ones(m * p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j * r + c] = V[j * p + k][c] * W[k * n + i][c]
                
                # Solve least squares
                try:
                    sol, residuals, rank_A, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to half-integers
                # sol shape is (m*r,)
                # We want to check if 2 * sol is close to an integer
                sol_2x = sol * 2
                sol_rounded = np.round(sol_2x)
                
                # If the difference is small, it's a candidate
                if np.max(np.abs(sol_2x - sol_rounded)) > 1e-6:
                    # Not close to half-integers
                    valid = False
                    break
                
                # Assign the rounded half-integer values to U
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c]) / 2.0
            
            if not valid:
                continue
            
            # Fast verification with numpy
            if verify_np(U, V, W, n, m, p):
                # Convert to exact Fractions for final verification
                U_frac = [[Fraction(U[i][j]).limit_denominator(2) for j in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(2) for j in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(2) for j in range(r)] for i in range(p*n)]
                
                if verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None

    # Try to find a decomposition with rank < 105
    # We start from 104 and go down. If we find one, we return it.
    # Given the difficulty, we limit the search to a few ranks and attempts.
    for r in range(nmp - 1, nmp - 6, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, num_attempts=50)
        if result is not None:
            return result

    # If no lower rank found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
