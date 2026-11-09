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
    
    # Verification function
    def verify(U, V, W, tol=1e-9):
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            return np.max(np.abs(T_recon - T)) < tol
        except:
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
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # Try to reduce rank by finding linear dependencies among rank-1 tensors
    def try_reduce_rank(U, V, W, tol=1e-8):
        """Iteratively try to remove columns that are linear combinations of others."""
        r = U.shape[1]
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        
        reduced = True
        while reduced and r > 1:
            reduced = False
            for r_to_remove in range(r):
                if time.time() > deadline - 5:
                    return U, V, W, r
                
                mask = [i for i in range(r) if i != r_to_remove]
                
                # Set up linear system: A * c = b
                # b[x,y,z] = U[x,r_to_remove]*V[y,r_to_remove]*W[z,r_to_remove]
                # A[x,y,z,j] = U[x,mask[j]]*V[y,mask[j]]*W[z,mask[j]]
                
                dim = U.shape[0] * V.shape[0] * W.shape[0]
                b = np.zeros(dim)
                A = np.zeros((dim, r-1))
                
                # Compute b
                b_vec = (U[:, r_to_remove][:, None, None] * 
                         V[:, r_to_remove][None, :, None] * 
                         W[:, r_to_remove][None, None, :]).flatten()
                
                # Compute A
                for j, col in enumerate(mask):
                    A[:, j] = (U[:, col][:, None, None] * 
                               V[:, col][None, :, None] * 
                               W[:, col][None, None, :]).flatten()
                
                # Solve least squares
                try:
                    c, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=1e-10)
                    if len(residuals) > 0 and residuals[0] < 1e-6:
                        b_approx = A @ c
                        if np.max(np.abs(b_approx - b_vec)) < 1e-6:
                            # Check the full tensor reconstruction
                            U_new = np.delete(U, r_to_remove, axis=1)
                            V_new = np.delete(V, r_to_remove, axis=1)
                            W_new = np.delete(W, r_to_remove, axis=1)
                            if verify(U_new, V_new, W_new):
                                U, V, W = U_new, V_new, W_new
                                r -= 1
                                reduced = True
                                break
                except:
                    pass
            if time.time() > deadline - 5:
                break
        
        return U, V, W, r
    
    # Apply rank reduction
    if time.time() < deadline - 10:
        U_red, V_red, W_red, r_red = try_reduce_rank(U, V, W)
        if r_red < best_rank:
            best_rank = r_red
            best_U, best_V, best_W = U_red, V_red, W_red
    
    # Try random search for lower ranks if time permits
    if time.time() < deadline - 10:
        values = [-1, -0.5, 0.5, 1]
        for target_rank in range(best_rank - 1, max(best_rank - 5, 15), -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            max_attempts = 50
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                U_try = np.zeros((n*m, target_rank))
                V_try = np.zeros((m*p, target_rank))
                W_try = np.zeros((p*n, target_rank))
                
                for r in range(target_rank):
                    nnz_u = random.randint(1, 2)
                    idx_u = random.sample(range(n*m), nnz_u)
                    for idx in idx_u:
                        U_try[idx, r] = random.choice(values)
                    
                    nnz_v = random.randint(1, 3)
                    idx_v = random.sample(range(m*p), nnz_v)
                    for idx in idx_v:
                        V_try[idx, r] = random.choice(values)
                    
                    nnz_w = random.randint(1, 2)
                    idx_w = random.sample(range(p*n), nnz_w)
                    for idx in idx_w:
                        W_try[idx, r] = random.choice(values)
                
                if verify(U_try, V_try, W_try):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    found = True
                    break
            
            if found:
                break
    
    return {"U": to_clean_list(best_U), "V": to_clean_list(best_V), "W": to_clean_list(best_W)}
# EVOLVE-BLOCK-END
