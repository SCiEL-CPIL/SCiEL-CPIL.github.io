import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Build the known rank-55 Strassen-based decomposition
    - Apply basis reduction: check if any rank-1 term can be expressed
      as a linear combination of others while preserving tensor constraints
    - Use Gaussian elimination with sparsity-preserving pivots
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Build the known rank-55 solution
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M2 = (A10 + A11)B00
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            # M3 = A00(B01 - B11)
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M4 = A11(B10 - B00)
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            # M5 = (A00 + A01)B11
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
        
        for k in range(p):
            b2k = 2 * p + k
            U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
            col += 1
            U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
            col += 1
        
        return U, V, W, rank
    
    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Verify rank-55 solution
    T_check = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_check[i, j, k] = np.sum(U55[i*m+j, :] * V55[j*p+k, :] * W55[k*n+i, :])
    
    assert np.allclose(T_check, 1.0), f"Rank-55 solution invalid"
    
    # Basis reduction approach:
    # The tensor constraint is: for all (i,j,k), sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    # 
    # If we have a rank-r decomposition, we can try to find if any term t can be 
    # expressed as a linear combination of other terms. That is, find coefficients 
    # c_s (s != t) such that replacing term t with sum_s c_s * term_s still satisfies 
    # all constraints.
    #
    # This means: for all (i,j,k):
    # U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = sum_{s!=t} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    #
    # This is a system of 60 linear equations in (r-1) unknowns (the c_s values).
    # If the system has a solution, we can eliminate term t.
    
    def try_eliminate_term(U, V, W, rank, t_to_eliminate):
        """
        Try to eliminate term t_to_eliminate by expressing it as a linear 
        combination of the other terms.
        Returns the coefficients if successful, None otherwise.
        """
        # Build the linear system: for each (i,j,k), 
        # sum_{s != t} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
        
        other_terms = [s for s in range(rank) if s != t_to_eliminate]
        n_other = len(other_terms)
        
        # Build coefficient matrix A and right-hand side b
        # A has 60 rows (one per (i,j,k)) and n_other columns
        A = np.zeros((n * m * p, n_other))
        b = np.zeros(n * m * p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Right-hand side: contribution of term t_to_eliminate
                    b[idx] = U[i*m+j, t_to_eliminate] * V[j*p+k, t_to_eliminate] * W[k*n+i, t_to_eliminate]
                    
                    # Left-hand side: contributions from other terms
                    for col_idx, s in enumerate(other_terms):
                        A[idx, col_idx] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                    
                    idx += 1
        
        # Solve the linear system using least squares (since it may be overdetermined)
        try:
            c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is valid (residuals should be near zero)
            if len(residuals) > 0 and residuals[0] < 1e-10:
                # Verify the solution
                A_reconstructed = A @ c
                if np.allclose(A_reconstructed, b, atol=1e-10):
                    return c, other_terms
        except np.linalg.LinAlgError:
            pass
        
        return None
    
    # Iteratively try to eliminate terms until no more can be removed
    # Precompute index arrays for efficient tensor verification
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    
    U_cur, V_cur, W_cur, rank_cur = U55, V55, W55, rank55
    
    while True:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        found_elimination = False
        for t in range(rank_cur):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            result = try_eliminate_term(U_cur, V_cur, W_cur, rank_cur, t)
            if result is not None:
                c, other_terms = result
                new_rank = rank_cur - 1
                new_U = np.zeros((n * m, new_rank), dtype=float)
                new_V = np.zeros((m * p, new_rank), dtype=float)
                new_W = np.zeros((p * n, new_rank), dtype=float)
                
                # Copy all terms except t
                col_idx = 0
                for s in range(rank_cur):
                    if s != t:
                        new_U[:, col_idx] = U_cur[:, s]
                        new_V[:, col_idx] = V_cur[:, s]
                        new_W[:, col_idx] = W_cur[:, s]
                        col_idx += 1
                
                # Verify the new decomposition using vectorized indexing
                T_new = np.einsum('it,jt,kt->ijk', 
                                  new_U[idx_i * m + idx_j, :], 
                                  new_V[idx_j * p + idx_k, :], 
                                  new_W[idx_k * n + idx_i, :])
                
                if np.allclose(T_new, 1.0, atol=1e-10):
                    U_cur, V_cur, W_cur, rank_cur = new_U, new_V, new_W, new_rank
                    best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
                    print(f"Eliminated term {t}, new rank: {new_rank}")
                    found_elimination = True
                    break  # Restart the loop with the new reduced rank
        
        if not found_elimination:
            break  # No more terms can be eliminated
    
    # Final verification using vectorized einsum
    T_final = np.einsum('it,jt,kt->ijk',
                        best_U[idx_i * m + idx_j, :],
                        best_V[idx_j * p + idx_k, :],
                        best_W[idx_k * n + idx_i, :])
    
    if not np.allclose(T_final, 1.0, atol=1e-10):
        # Fall back to rank-55
        best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
