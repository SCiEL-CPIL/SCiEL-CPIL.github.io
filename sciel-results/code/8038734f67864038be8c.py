import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Precompute indices for the matrix multiplication tensor
    # Tensor T[i,j,k] = 1 if we multiply (A*B)[i,k] which involves sum_j A[i,j]*B[j,k]
    # But the contract says:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is the tensor for matrix multiplication C = A * B where:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T_{i,j,k} = 1 (for the identity mapping)
    
    # Schoolbook rank is n*m*p = 2*3*6 = 36
    # Lower bounds: 
    # - max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # - But there are better lower bounds
    
    # The tensor is a "matrix multiplication" tensor.
    # For M_{n,m,p}, known bounds:
    # - Schoolbook: n*m*p
    # - Winograd's scheme can sometimes help
    # - For small cases, we can try to find exact decompositions
    
    # Let's try a systematic approach:
    # 1. Start with known decompositions
    # 2. Try to find lower-rank decompositions using linear algebra
    
    # For M_{2,3,6}, the schoolbook rank is 36.
    # Can we do better?
    
    # Approach: We'll try to find a rank-r decomposition by:
    # 1. Randomly initializing U and V
    # 2. Solving for W using least squares
    # 3. Checking if the solution is close to integer/half-integer values
    
    # But first, let's check if there's a known better decomposition.
    # For M_{2,3,6}, I don't immediately know of a sub-schoolbook decomposition.
    # Let's try to search for one.
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    def try_find_decomposition(r, max_attempts=50):
        """Try to find a rank-r decomposition."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Randomly initialize U and V with small integers
            U_np = np.random.randint(-2, 3, size=(n * m, r))
            V_np = np.random.randint(-2, 3, size=(m * p, r))
            
            # Build the linear system for W
            # Variables: W[k*n + i, c] for k in range(p), i in range(n), c in range(r)
            # Total variables: p * n * r
            num_vars = p * n * r
            num_eqs = n * m * p
            
            # Build A matrix
            A = np.zeros((num_eqs, num_vars), dtype=float)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(r):
                            coeff = U_np[i * m + j, c] * V_np[j * p + k, c]
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx, var_idx] = coeff
            
            b = np.ones(num_eqs)
            
            # Solve using least squares
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if solution is close to exact
            residual = np.abs(A @ x - b).max()
            if residual > 1e-6:
                continue
            
            # Try to round to nearest integer/half-integer
            W_vals = np.round(x * 2) / 2
            
            # Verify with rounded values
            W_np = W_vals.reshape(p * n, r)
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                        error = max(error, abs(s - 1.0))
            
            if error > 1e-9:
                continue
            
            # Convert to Fractions for exact representation
            U = [[Fraction(int(U_np[i, c])) for c in range(r)] for i in range(n * m)]
            V = [[Fraction(int(V_np[i, c])) for c in range(r)] for i in range(m * p)]
            W = [[Fraction(W_np[i, c]) for c in range(r)] for i in range(p * n)]
            
            # Final exact verification
            if verify_decomposition(U, V, W, r):
                return U, V, W
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 18
    # But likely we can't do much better than schoolbook for these dimensions
    
    # First, let's try a few ranks below schoolbook
    schoolbook_rank = n * m * p  # 36
    
    # Try ranks from 18 to 35
    for r in range(18, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        
        result = try_find_decomposition(r, max_attempts=20)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If we haven't found anything better, return schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
