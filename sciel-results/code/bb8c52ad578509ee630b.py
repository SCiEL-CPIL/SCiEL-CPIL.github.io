"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def trivial_decomp():
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
    
    def verify_decomp(U, V, W):
        """Verify that the decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def try_block_decomposition(r, block_m, block_p, seed=42):
        """
        Try to decompose by treating each (i, k) pair as needing to solve for j.
        For each fixed i and k, we need sum_c U[i*m+j, c] * V[j*p+k, c] = 1/W[k*n+i, c] for each c...
        
        Actually, let's use a different approach. For a block decomposition:
        Split j into blocks of size block_m, and k into blocks of size block_p.
        For each block pair (jm, kp), we use a sub-decomposition.
        
        Let's try: for each i, we have m=4 j's and p=6 k's.
        If we split j into 2 blocks of 2, and k into 3 blocks of 2:
        For block (jm, kp), we need a decomposition of the 2x2 matrix multiplication.
        The rank of 2x2 matrix multiplication is 7 (Strassen).
        
        So for each i, we have 2*3 = 6 blocks, each needing rank 7.
        Total rank per i: 6*7 = 42. For 3 i's: 126. That's worse than 72.
        
        Let's try a different approach. Use the fact that 4 = 2*2 and 6 = 2*3.
        """
        pass
    
    # Let's try a direct approach using the structure of the problem.
    # For matrix multiplication C = A * B where A is n x m and B is m x p,
    # we can use the standard decomposition with rank n*m*p.
    
    # But we can do better. Let's try to use a "sum of products" approach.
    # For each i, C[i,k] = sum_j A[i,j] * B[j,k].
    
    # One idea: use a decomposition where we precompute certain linear combinations.
    
    # For 3x4x6, let's try rank 48.
    # We can think of it as: for each of the 3 rows of A, we need to compute 6 outputs.
    # Each output is a dot product of a 4-vector with 6 different 4-vectors.
    
    # Alternative: use a decomposition based on polynomial interpolation or similar.
    
    # Let's try a specific construction.
    # Rank 48: For each i (3 values), use 16 terms.
    # 16 = 4*4. We can use 4 terms for each of 4 "groups" of k values.
    
    # Actually, let's try to find a decomposition of rank 48 or lower using optimization.
    
    def optimize_decomposition(r, max_time=100, seed=42):
        """
        Use alternating minimization to find a decomposition of rank r.
        Fix two factors, solve for the third using linear algebra.
        """
        if time.time() > deadline - 20:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Reshape for computation
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        def compute_residuals(U, V, W):
            """Compute the residual for each (i, j, k)."""
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residuals[i, j, k] = s - 1.0
            return residuals
        
        def solve_for_W(U, V):
            """Given U and V, solve for W to minimize the residual."""
            # For each (k, i), we want to find W[k*n+i, :] such that
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j.
            # This is a system of m equations in r unknowns for each (k, i).
            
            W_new = np.zeros((p * n, r))
            for k in range(p):
                for i in range(n):
                    # Build the system: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # Let A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then we want A @ W[k*n+i, :] = 1 (vector of ones)
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    
                    # Solve A @ x = 1 using least squares
                    if r >= m:
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = x
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                    else:
                        # Underdetermined, use pseudoinverse
                        try:
                            x = np.linalg.pinv(A) @ np.ones(m)
                            W_new[k*n+i, :] = x
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
            return W_new
        
        def solve_for_V(U, W):
            """Given U and W, solve for V."""
            V_new = np.zeros((m * p, r))
            for j in range(m):
                for k in range(p):
                    # For each i, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # Let A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # Then we want A @ V[j*p+k, :] = 1
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    
                    if r >= n:
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = x
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                    else:
                        try:
                            x = np.linalg.pinv(A) @ np.ones(n)
                            V_new[j*p+k, :] = x
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
            return V_new
        
        def solve_for_U(V, W):
            """Given V and W, solve for U."""
            U_new = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    # For each k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # Let A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then we want A @ U[i*m+j, :] = 1
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    
                    if r >= p:
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = x
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                    else:
                        try:
                            x = np.linalg.pinv(A) @ np.ones(p)
                            U_new[i*m+j, :] = x
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
            return U_new
        
        # Alternating minimization
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(200):
            if time.time() > deadline - 20:
                break
            
            # Solve for W
            W = solve_for_W(U, V)
            # Solve for V
            V = solve_for_V(U, W)
            # Solve for U
            U = solve_for_U(V, W)
            
            # Compute residual
            residuals = compute_residuals(U, V, W)
            residual = np.sum(residuals ** 2)
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-12:
                break
        
        if best_residual < 1e-10:
            # Convert to list of lists
            U_out = [[float(best_U[i, c]) for c in range(r)] for i in range(n * m)]
            V_out = [[float(best_V[i, c]) for c in range(r)] for i in range(m * p)]
            W_out = [[float(best_W[i, c]) for c in range(r)] for i in range(p * n)]
            return {"U": U_out, "V": V_out, "W": W_out}
        
        return None
    
    # Try different ranks
    for r in [48, 42, 36, 30, 24]:
        if time.time() > deadline - 30:
            break
        
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            result = optimize_decomposition(r, seed=seed)
            if result is not None:
                # Verify
                if verify_decomp(result["U"], result["V"], result["W"]):
                    return result
    
    # Fall back to trivial
    return trivial_decomp()
# EVOLVE-BLOCK-END
