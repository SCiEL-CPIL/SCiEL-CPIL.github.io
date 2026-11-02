import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 105
    
    # Precompute schoolbook basis: each term is (i,j,k) -> unit vectors
    # U_basis[term][i*m+j], V_basis[term][j*p+k], W_basis[term][k*n+i]
    num_terms = n * m * p
    U_basis = np.zeros((num_terms, n * m), dtype=np.float64)
    V_basis = np.zeros((num_terms, m * p), dtype=np.float64)
    W_basis = np.zeros((num_terms, p * n), dtype=np.float64)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_basis[t, i * m + j] = 1.0
                V_basis[t, j * p + k] = 1.0
                W_basis[t, k * n + i] = 1.0
                t += 1
    
    # Helper: verify a decomposition exactly using fractions
    def verify_exact(U_mat, V_mat, W_mat):
        """
        U_mat: (n*m, rank) list of lists of Fractions
        V_mat: (m*p, rank) list of lists of Fractions  
        W_mat: (p*n, rank) list of lists of Fractions
        Returns True if sum_r U[:,r] * V[:,r] * W[:,r] == tensor
        """
        rank = len(U_mat[0])
        # Compute the decomposed tensor T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Compare with T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0
        # Actually the matrix multiplication tensor is: T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        # Wait, the tensor for matrix multiplication: T_{i,j,k} where i indexes rows of A, j indexes cols of A/rows of B, k indexes cols of B
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k since C[i,k] = sum_j A[i,j] B[j,k]
        # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Helper: check if a set of random coefficient vectors gives a valid decomposition
    def try_random_decomposition(rank, num_trials=100):
        """
        Try to find rank-r decomposition by expressing each rank-r term as a linear combination
        of schoolbook terms with random integer/half-integer coefficients.
        """
        rng = np.random.default_rng(42)
        
        for trial in range(num_trials):
            # Generate random coefficient matrix: (rank, num_terms)
            # Coefficients are random integers in {-1, 0, 1} or half-integers
            coeff = rng.integers(-2, 3, size=(rank, num_terms)) * 0.5
            # Ensure at least some non-zero entries
            if np.all(coeff == 0):
                continue
            
            # Build U, V, W
            U_mat = np.zeros((n * m, rank))
            V_mat = np.zeros((m * p, rank))
            W_mat = np.zeros((p * n, rank))
            
            for r in range(rank):
                U_mat[:, r] = U_basis.T @ coeff[r, :]
                V_mat[:, r] = V_basis.T @ coeff[r, :]
                W_mat[:, r] = W_basis.T @ coeff[r, :]
            
            # Quick numerical check
            # T_decomp[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Should equal 1 for all i,j,k
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-9:
                # Convert to exact fractions and verify
                U_frac = [[Fraction(U_mat[i, r]).limit_denominator(10000) for r in range(rank)] for i in range(n*m)]
                V_frac = [[Fraction(V_mat[i, r]).limit_denominator(10000) for r in range(rank)] for i in range(m*p)]
                W_frac = [[Fraction(W_mat[i, r]).limit_denominator(10000) for r in range(rank)] for i in range(p*n)]
                
                if verify_exact(U_frac, V_frac, W_frac):
                    return U_frac, V_frac, W_frac, rank
            
            if time.time() > deadline:
                break
        
        return None
    
    # Try to find a decomposition with rank less than schoolbook
    best_result = None
    for target_rank in range(schoolbook_rank - 1, 0, -1):
        if time.time() > deadline - 5:
            break
        result = try_random_decomposition(target_rank, num_trials=50)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U_frac, V_frac, W_frac, rank = best_result
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fall back to schoolbook decomposition
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
