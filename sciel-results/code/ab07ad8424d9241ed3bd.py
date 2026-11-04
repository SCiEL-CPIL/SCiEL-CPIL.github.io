"""Deterministic rank-reduction for 4x5 * 5x6 matrix multiplication tensor."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the schoolbook decomposition (rank = n*m*p = 120)
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=np.int64)
    V = np.zeros((m * p, rank_school), dtype=np.int64)
    W = np.zeros((p * n, rank_school), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    def verify(U_arr, V_arr, W_arr, r):
        """Verify decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
        if time.time() > deadline - 5:
            return False
        U_r = U_arr.reshape(n, m, r)
        V_r = V_arr.reshape(m, p, r)
        W_r = W_arr.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Verify schoolbook
    if not verify(U, V, W, rank_school):
        # Fallback: return schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = rank_school

    # Strategy: For fixed sparse patterns of U and V, solve for W via linear system.
    # The key insight is that if we fix U and V to be sparse (mostly zeros with some 1s),
    # then W is determined by a linear system. We try various sparse patterns and
    # check if the resulting W has integer or half-integer entries.
    
    # A more systematic approach: 
    # For each column c, we can think of it as contributing to a "partial product"
    # The equation is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # If we fix U and V, this is linear in W. But W[k,i,c] only appears in the
    # equation for (i,j,k) where both U[i,j,c] and V[j,k,c] are non-zero.
    #
    # Let's try a structured approach:
    # For each column c, choose a set S_c of (i,j) pairs where U[i,j,c] != 0
    # and a set T_c of (j,k) pairs where V[j,k,c] != 0.
    # Then W[k,i,c] is constrained by equations where (i,j) in S_c and (j,k) in T_c.
    #
    # To keep the system tractable, let's restrict to patterns where each column
    # has a "star" structure: U[i_c, j, c] for all j, and V[j, k_c, c] for all j.
    # This means column c "covers" all (i_c, j, k_c) for j = 0..m-1.
    #
    # Actually, let's try a different approach: use the fact that the matrix
    # multiplication tensor has a certain symmetry and try to exploit block structure.
    #
    # For 4x5x6, let's try to partition the indices into blocks and find
    # decompositions that respect this partitioning.
    
    # Alternative: Try to find decompositions where U, V, W have entries from {0, 1, -1}
    # or {0, 1/2, 1, 3/2, ...} and the rank is less than 120.
    
    # Let's try a greedy approach: start with schoolbook and try to merge columns.
    # If two columns c1 and c2 can be "merged" (i.e., their combined contribution
    # can be represented by a single column), we reduce the rank by 1.
    
    # Actually, let's try a more direct approach: for each target rank r < 120,
    # try to find U, V, W with integer/half-integer entries such that the
    # decomposition holds. We'll use a randomized local search.
    
    np.random.seed(42)
    
    # Try ranks from 119 down to 100
    min_rank_to_try = 90
    
    for target_rank in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Initialize U, V, W with random sparse patterns
            U_try = np.zeros((n * m, target_rank), dtype=np.float64)
            V_try = np.zeros((m * p, target_rank), dtype=np.float64)
            W_try = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # For each column, assign random non-zero entries
            for c in range(target_rank):
                # Choose random (i, j) for U
                i_c = np.random.randint(0, n)
                j_c = np.random.randint(0, m)
                U_try[i_c * m + j_c, c] = 1.0
                
                # Choose random (j, k) for V
                j_c2 = np.random.randint(0, m)
                k_c = np.random.randint(0, p)
                V_try[j_c2 * p + k_c, c] = 1.0
                
                # Set W to satisfy the equation for the "center" (i_c, j_c, k_c)
                # if U[i_c, j_c, c] and V[j_c, k_c, c] are both non-zero
                if U_try[i_c * m + j_c, c] != 0 and V_try[j_c * p + k_c, c] != 0:
                    W_try[k_c * n + i_c, c] = 1.0 / (U_try[i_c * m + j_c, c] * V_try[j_c * p + k_c, c])
            
            # Now try to adjust W to satisfy all equations
            # This is a least squares problem: minimize ||A @ W - b||
            # where A is the coefficient matrix and b is the target tensor
            
            # Build the system
            num_eqs = n * m * p
            num_vars = p * n * target_rank
            
            A = np.zeros((num_eqs, num_vars), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            coeff = U_try[i * m + j, c] * V_try[j * p + k, c]
                            if coeff != 0:
                                var_idx = k * n * target_rank + i * target_rank + c
                                A[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            # Solve the least squares problem
            try:
                W_solution, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is valid
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        continue
                
                W_try = W_solution.reshape(p, n, target_rank)
                
                # Check if W_try is close to half-integers
                W_rounded = np.round(W_try * 2) / 2
                
                # Verify the decomposition with rounded W
                U_r = U_try.reshape(n, m, target_rank)
                V_r = V_try.reshape(m, p, target_rank)
                T_rounded = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_rounded)
                
                if np.allclose(T_rounded, np.ones((n, m, p)), atol=1e-9):
                    # Found a valid decomposition!
                    best_U, best_V, best_W = U_try, V_try, W_rounded
                    best_rank = target_rank
                    found = True
                    break
            except Exception:
                continue
        
        if found:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
