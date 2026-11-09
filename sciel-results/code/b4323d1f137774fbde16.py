"""Matrix multiplication 2,3,6 - rank-minimal tensor decomposition."""
import random
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Build the target tensor T as a 3D array
    # T[x, y, z] = 1 if x = i*m+j, y = j*p+k, z = k*n+i for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Verification function: check if U, V, W decompose T exactly
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Reconstruct tensor: T_recon[x,y,z] = sum_r U[x,r]*V[y,r]*W[z,r]
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T)) < tol:
                return True
        except:
            pass
        return False
    
    # Convert to list of lists with integer/half-integer values
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    # half-integer
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # For 2x3 x 3x6, let's think about the structure
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02, b03, b04, b05],
    #      [b10, b11, b12, b13, b14, b15],
    #      [b20, b21, b22, b23, b24, b25]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # C has 2*6=12 entries. Each C[i,k] is a sum of 3 products.
    # Total 36 products.
    
    # The tensor rank of matrix multiplication is studied.
    # For 2x2 x 2x2, rank is 7 (Strassen).
    # For other small cases, there are known results.
    
    # Let's try a systematic search: parameterize U, V, W with sparse structure
    # and use the fact that entries should be small integers or half-integers.
    
    # Alternative: use the fact that the problem is small and try to solve
    # the system of equations for a given rank.
    
    # For rank r, we have r variables per matrix, so 3*r unknowns.
    # Constraints: (n*m)*(m*p)*(p*n) = 6*18*12 = 1296 equations.
    # But only 36 are non-zero (the support), and the rest must be 0.
    # Actually all 1296 entries of T_recon must match T.
    
    # This is a system of polynomial equations (degree 3 in the unknowns).
    # Hard to solve directly.
    
    # Better approach: use the schoolbook and try to find linear dependencies
    # among the "rank-1 tensors" U_col_r ⊗ V_col_r ⊗ W_col_r.
    
    # If we can find coefficients c_r such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0,
    # then we can eliminate one column.
    
    # Let's check if the schoolbook columns are linearly independent as 3-tensors.
    
    # Flatten each rank-1 tensor: U_col_r ⊗ V_col_r ⊗ W_col_r is a vector of size (n*m)*(m*p)*(p*n)
    # Check linear independence of the 36 such vectors.
    
    if time.time() < deadline:
        # Build the matrix of flattened rank-1 tensors
        tensor_dim = (n*m) * (m*p) * (p*n)
        rank1_tensors = []
        for r in range(schoolbook_rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # Outer product
            outer = np.einsum('x,y,z->xyz', u_col, v_col, w_col)
            rank1_tensors.append(outer.flatten())
        
        M = np.array(rank1_tensors).T  # shape: tensor_dim x rank
        # Check rank of M
        try:
            rank_M = np.linalg.matrix_rank(M, tol=1e-9)
            print(f"Schoolbook columns: {schoolbook_rank}, linear rank: {rank_M}")
        except:
            rank_M = schoolbook_rank
        
        # If rank_M < schoolbook_rank, we have linear dependencies
        # and can reduce the rank.
    
    # For now, return the schoolbook solution which is valid
    # Try a few more strategies if time permits
    
    # Try random perturbation and SVD-based rank reduction
    if time.time() < deadline - 10:
        # Try to find a lower rank by solving a least squares problem
        # for candidate ranks
        min_possible = max(n*p, m*(n+p))  # rough lower bound
        min_possible = max(min_possible, 1)
        
        for target_rank in range(best_rank - 1, min_possible - 1, -1):
            if time.time() > deadline - 5:
                break
            
            # Try to find U, V, W of rank target_rank
            # Use random initialization and gradient-free search
            # This is very hard, so we'll just try a few random attempts
            
            attempts = 0
            max_attempts = 5
            found = False
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                attempts += 1
                
                # Initialize with small random integers
                U_try = np.zeros((n*m, target_rank))
                V_try = np.zeros((m*p, target_rank))
                W_try = np.zeros((p*n, target_rank))
                
                # Fill with random -1, 0, 1
                for r in range(target_rank):
                    # Choose random non-zero entries
                    num_nonzero_u = random.randint(1, 3)
                    idx_u = random.sample(range(n*m), num_nonzero_u)
                    for idx in idx_u:
                        U_try[idx, r] = random.choice([-1, 1])
                    
                    num_nonzero_v = random.randint(1, 3)
                    idx_v = random.sample(range(m*p), num_nonzero_v)
                    for idx in idx_v:
                        V_try[idx, r] = random.choice([-1, 1])
                    
                    num_nonzero_w = random.randint(1, 3)
                    idx_w = random.sample(range(p*n), num_nonzero_w)
                    for idx in idx_w:
                        W_try[idx, r] = random.choice([-1, 1])
                
                if verify(U_try, V_try, W_try):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    found = True
                    break
            
            if found:
                break
    
    # Return the best solution found
    U_out = to_clean_list(best_U)
    V_out = to_clean_list(best_V)
    W_out = to_clean_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
