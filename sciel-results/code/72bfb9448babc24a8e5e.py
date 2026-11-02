import numpy as np
import time
import random
import itertools
from scipy.linalg import lstsq

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] where:
    i: A index (0..n*m-1), i = i_row * m + i_col
    j: B index (0..m*p-1), j = j_row * p + j_col  
    k: C index (0..p*n-1), k = k_col * n + k_row
    
    T[i,j,k] = 1 if A[i] * B[j] contributes to C[k]
    i.e., i_col == j_row and j_col == k_col and i_row == k_row
    """
    T = np.zeros((n*m, m*p, p*n))
    for i_row in range(n):
        for i_col in range(m):
            i = i_row * m + i_col
            for j_row in range(m):
                for j_col in range(p):
                    j = j_row * p + j_col
                    if i_col == j_row:
                        k_row = i_row
                        k_col = j_col
                        k = k_col * n + k_row
                        T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, target_T, tol=1e-10):
    """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == target_T"""
    rank = U.shape[1]
    reconstructed = np.zeros_like(target_T)
    for r in range(rank):
        reconstructed += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(reconstructed, target_T, atol=tol)

def _try_rank_reduction(U, V, W, target_T, start_rank, max_reduction=3, time_budget=60):
    """Try to reduce rank by checking if any column is linearly dependent."""
    start_time = time.time()
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = start_rank
    
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    best_rank = current_rank
    
    # Try removing one column at a time
    for r in range(current_rank):
        if time.time() - start_time > time_budget:
            break
            
        # Try to express column r as a combination of others
        # Remove column r and check if we can find coefficients to restore it
        U_reduced = np.delete(current_U, r, axis=1)
        V_reduced = np.delete(current_V, r, axis=1)
        W_reduced = np.delete(current_W, r, axis=1)
        
        # The contribution of column r is: U[:,r] outer V[:,r] outer W[:,r]
        # We need: U[:,r] outer V[:,r] outer W[:,r] = sum_s alpha_s * U[:,s] outer V[:,s] outer W[:,s]
        # This is a rank-1 tensor approximation problem
        
        # Simpler approach: just check if the tensor without column r is still valid
        # (This only works if column r contributes zero, which is unlikely)
        
        # Better: try to solve for coefficients
        # Flatten the target contribution
        target_contrib = np.einsum('i,j,k->ijk', current_U[:,r], current_V[:,r], current_W[:,r])
        
        # Build matrix where each column is the flattened contribution of another column
        # M[:, s] = vec(U[:,s] outer V[:,s] outer W[:,s])
        n_terms = current_rank - 1
        if n_terms == 0:
            continue
            
        M = np.zeros((target_T.size, n_terms))
        for s in range(current_rank):
            if s == r:
                continue
            M[:, s - (1 if s > r else 0)] = np.einsum('i,j,k->ijk', 
                current_U[:,s], current_V[:,s], current_W[:,s]).flatten()
        
        # Solve M @ alpha = target_contrib.flatten()
        try:
            alpha, residuals, rank_m, sv = np.linalg.lstsq(M, target_contrib.flatten(), rcond=None)
            residual = np.linalg.norm(M @ alpha - target_contrib.flatten())
            
            if residual < 1e-8:
                # Column r is redundant! Remove it.
                current_U = np.delete(current_U, r, axis=1)
                current_V = np.delete(current_V, r, axis=1)
                current_W = np.delete(current_W, r, axis=1)
                current_rank -= 1
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                best_rank = current_rank
                # Restart search with reduced rank
                r = -1  # Reset loop
        except:
            pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 2, 2, 7
    
    # Build target tensor for verification
    target_T = _build_target_tensor(n, m, p)
    
    # Construct the known valid Rank 25 decomposition:
    # 3 blocks of Strassen (rank 7) for columns 0-5 of B/C
    # 1 block of Schoolbook (rank 4) for column 6 of B/C
    base_rank = 25
    U = np.zeros((4, base_rank))
    V = np.zeros((14, base_rank))
    W = np.zeros((14, base_rank))
    
    # Standard Strassen for 2x2 * 2x2 = C (2x2)
    # C00 = M1 - M3 - M4 + M5
    # C01 = M2 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10)(B00+B01)
    # M7 = (A01-A11)(B10+B11) -- not used in standard Strassen output
    
    # Actually, let me use the standard Strassen formulation:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b01-b11)
    # m4 = (a11)(b10-b00)
    # m5 = (a00+a01)(b11)
    # m6 = (a10)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1 - m3 - m4 + m5
    # c01 = m2 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 - m3 + m6
    
    strassen_U = np.array([
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [0, 0, 1, 0],  # M6: A10
        [0, 1, 0, -1]  # M7: A01 - A11
    ], dtype=float)
    
    # B coefficients for each M
    # M1: B00+B11 -> indices 0,3
    # M2: B00 -> index 0
    # M3: B01-B11 -> indices 1,3
    # M4: B10-B00 -> indices 2,0
    # M5: B11 -> index 3
    # M6: B00+B01 -> indices 0,1
    # M7: B10+B11 -> indices 2,3
    strassen_V_local = [
        {0: 1, 3: 1},   # M1
        {0: 1},         # M2
        {1: 1, 3: -1},  # M3
        {2: 1, 0: -1},  # M4
        {3: 1},         # M5
        {0: 1, 1: 1},   # M6
        {2: 1, 3: 1}    # M7
    ]
    
    # C coefficients for each M
    # c00 = m1 - m3 - m4 + m5
    # c01 = m2 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 - m3 + m6
    # M1: +c00, +c11
    # M2: +c01, +c10, -c11
    # M3: -c00, -c11
    # M4: -c00, +c10
    # M5: +c00, +c01
    # M6: +c11
    # M7: not used
    strassen_W_local = [
        {0: 1, 3: 1},    # M1: c00 + c11
        {1: 1, 2: 1, 3: -1},  # M2: c01 + c10 - c11
        {0: -1, 3: -1},  # M3: -c00 - c11
        {0: -1, 2: 1},   # M4: -c00 + c10
        {0: 1, 1: 1},    # M5: c00 + c01
        {3: 1},          # M6: c11
        {}               # M7: unused
    ]
    
    col = 0
    for b in range(3):
        for r in range(7):
            U[:, col] += strassen_U[r, :]
            
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx, col] += coeff
            
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx, col] += coeff
            
            col += 1
            
    # Schoolbook for last block (col 6)
    # A (2x2) * B_col6 (2x1) -> C_col6 (2x1)
    # C06 = A00*B06 + A01*B16
    # C16 = A10*B06 + A11*B16
    
    schoolbook_U = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    schoolbook_V_local = [
        {0: 1}, # B06
        {1: 1}, # B16
        {0: 1}, # B06
        {1: 1}  # B16
    ]
    
    schoolbook_W_local = [
        {0: 1}, # C06
        {0: 1}, # C06
        {1: 1}, # C16
        {1: 1}  # C16
    ]
    
    for r in range(4):
        U[:, col] += schoolbook_U[r, :]
        
        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0:
                v_idx = 0 * p + 6
            elif loc_idx == 1:
                v_idx = 1 * p + 6
            V[v_idx, col] += coeff
            
        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0:
                w_idx = 6 * n + 0
            elif loc_idx == 1:
                w_idx = 6 * n + 1
            W[w_idx, col] += coeff
            
        col += 1

    # Verify the decomposition
    is_valid = _verify_decomposition(U, V, W, target_T)
    if not is_valid:
        # If invalid, fall back to pure schoolbook
        U = np.zeros((4, 28))
        V = np.zeros((14, 28))
        W = np.zeros((14, 28))
        for i_row in range(n):
            for i_col in range(m):
                for j_col in range(p):
                    idx = (i_row * m + i_col) * p + j_col
                    U[i_row * m + i_col, idx] = 1
                    V[i_col * p + j_col, idx] = 1
                    W[j_col * n + i_row, idx] = 1
        is_valid = _verify_decomposition(U, V, W, target_T)
    
    # Try to reduce rank
    if time.time() < deadline - 30 and is_valid:
        try:
            U_reduced, V_reduced, W_reduced, reduced_rank = _try_rank_reduction(
                U, V, W, target_T, base_rank, time_budget=60
            )
            if reduced_rank < base_rank:
                U, V, W = U_reduced, V_reduced, W_reduced
                base_rank = reduced_rank
        except Exception as e:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
