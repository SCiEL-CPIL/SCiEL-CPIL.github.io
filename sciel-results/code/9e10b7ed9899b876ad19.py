"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time
from itertools import combinations

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def _remaining_time():
    return _TIMEOUT - (time.time() - _START_TIME)

# Precompute the target tensor flattened for fast verification
def _build_target_tensor(n, m, p):
    """Build and return the flattened target tensor."""
    T = np.zeros(n * m * m * p * p * n, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u * (m * p) * (p * n) + v * (p * n) + w] = 1.0
    return T

_TARGET_TENSOR = _build_target_tensor(2, 3, 8)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the tensor decomposition is exact using vectorized ops."""
    if not U:
        return False
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # T_approx[u, v, w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # Flatten: T_approx_flat[idx] where idx = u*(m*p)*(p*n) + v*(p*n) + w
    # = sum_r U[u,r] * V[v,r] * W[w,r]
    
    # Vectorized: for each r, outer product
    # T_approx = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Reshape to flat: sum_r (U[:,r] * V[:,r] * W[:,r]) reshaped
    
    T_approx = np.zeros(n * m * m * p * p * n, dtype=np.float64)
    for r in range(rank):
        # U[:,r] has shape (n*m,), V[:,r] has shape (m*p,), W[:,r] has shape (p*n,)
        # Outer product flattened
        T_approx += (U_np[:, r][:, np.newaxis, np.newaxis] * 
                     V_np[:, r][np.newaxis, :, np.newaxis] * 
                     W_np[:, r][np.newaxis, np.newaxis, :]).ravel()
    
    return np.allclose(T_approx, _TARGET_TENSOR, atol=1e-9)

def _construct_strassen_238():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    Uses 4 blocks of 2x3x2.
    Each 2x3x2 block uses Strassen (rank 7) for 2x2x2 part and Outer Product (rank 4) for 2x1x2 part.
    Total rank: 4 * (7 + 4) = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            W[idx][col] += c

    current_col = 0
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        a_idx = 0
        b_idx = 1
        c_idx = 3
        d_idx = 4
        
        # M1 = (a+d)(e+h)
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        
        # M2 = (c+d)e
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        
        # M3 = a(f-h)
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        
        # M4 = d(g-e)
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        
        # M5 = (a+b)h
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        
        # M6 = (c-a)(e+f)
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        
        # M7 = (b-d)(g+h)
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: j = 2 (2x1x2)
        a2_idx = 0 * m + 2
        c2_idx = 1 * m + 2
        
        e2_idx = 2 * p + k0
        f2_idx = 2 * p + k1
        
        # C[0, k0] = A[0,2] * B[2, k0]
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        
        # C[0, k1] = A[0,2] * B[2, k1]
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        
        # C[1, k0] = A[1,2] * B[2, k0]
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        
        # C[1, k1] = A[1,2] * B[2, k1]
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}

def _try_structured_rank_reduction(U, V, W, target_rank, n, m, p):
    """
    Try to reduce the rank of a decomposition by finding linear dependencies
    among the rank-1 terms. This is a local optimization approach.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    # Convert to numpy
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try to find a subset of columns that can be removed or combined
    # Strategy: try removing each column and checking if the residual
    # can be absorbed into other columns
    
    best_result = None
    
    # Try greedy removal: for each column, see if removing it and
    # adjusting others still gives a valid decomposition
    for col_to_remove in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 30:
            break
        
        # Remove this column
        mask = [i for i in range(rank) if i != col_to_remove]
        U_sub = U_np[:, mask]
        V_sub = V_np[:, mask]
        W_sub = W_np[:, mask]
        
        # Check if this still decomposes correctly
        T_approx = np.zeros(n * m * m * p * p * n, dtype=np.float64)
        for r in range(len(mask)):
            T_approx += (U_sub[:, r][:, np.newaxis, np.newaxis] * 
                         V_sub[:, r][np.newaxis, :, np.newaxis] * 
                         W_sub[:, r][np.newaxis, np.newaxis, :]).ravel()
        
        if np.allclose(T_approx, _TARGET_TENSOR, atol=1e-9):
            # Found a valid lower-rank decomposition!
            new_rank = len(mask)
            if best_result is None or new_rank < len(best_result[0][0]):
                U_new = [[U_sub[i, r] for r in range(new_rank)] for i in range(n * m)]
                V_new = [[V_sub[i, r] for r in range(new_rank)] for i in range(m * p)]
                W_new = [[W_sub[i, r] for r in range(new_rank)] for i in range(p * n)]
                best_result = (U_new, V_new, W_new)
    
    return best_result

def _construct_schoolbook():
    """
    Constructs rank 48 schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _try_rank_decomposition(n, m, p, max_rank, seed=42):
    """
    Try to find a rank decomposition using structured search.
    Attempts to find rank 43 by exploiting linear dependencies in the rank-44 construction.
    """
    if time.time() - _START_TIME > _TIMEOUT - 40:
        return None
    
    # Start with the rank-44 construction and try to reduce it
    cand = _construct_strassen_238()
    U, V, W = cand["U"], cand["V"], cand["W"]
    rank = len(U[0])
    
    if rank <= max_rank:
        return (rank, U, V, W)
    
    # Try structured reduction: look for pairs of columns that can be merged
    # by finding if one column is a linear combination of others
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try removing each column and see if we can compensate
    for col_to_remove in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 40:
            break
        
        # Create mask excluding this column
        mask = [i for i in range(rank) if i != col_to_remove]
        
        # Check if remaining columns still span the tensor
        T_approx = np.zeros(n * m * m * p * p * n, dtype=np.float64)
        for r in range(len(mask)):
            T_approx += (U_np[:, mask[r]][:, np.newaxis, np.newaxis] * 
                         V_np[:, mask[r]][np.newaxis, :, np.newaxis] * 
                         W_np[:, mask[r]][np.newaxis, np.newaxis, :]).ravel()
        
        if np.allclose(T_approx, _TARGET_TENSOR, atol=1e-9):
            # Found valid lower rank!
            new_rank = len(mask)
            U_new = [[U_np[i, mask[r]] for r in range(new_rank)] for i in range(n * m)]
            V_new = [[V_np[i, mask[r]] for r in range(new_rank)] for i in range(m * p)]
            W_new = [[W_np[i, mask[r]] for r in range(new_rank)] for i in range(p * n)]
            return (new_rank, U_new, V_new, W_new)
    
    # Try a different approach: use the block structure to find cross-block dependencies
    # The 4 blocks each contribute 11 terms. Check if any cross-block combination works.
    
    # For now, return None if no simple reduction found
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Candidate 1: Rank 44 Strassen-based
    cand1 = _construct_strassen_238()
    if _verify_decomposition(cand1["U"], cand1["V"], cand1["W"], n, m, p):
        # Try to see if we can do better
        if _remaining_time() > 50:
            # Try structured search for better decomposition
            result = _try_rank_decomposition(n, m, p, 43)
            if result is not None and result[0] < 44:
                rank, U, V, W = result
                # Convert to list format with proper integer/half-integer detection
                U_list = []
                for i in range(len(U)):
                    row = []
                    for r in range(len(U[0])):
                        val = U[i][r]
                        if abs(val - round(val)) < 1e-9:
                            row.append(int(round(val)))
                        else:
                            # Check if it's a half-integer
                            doubled = val * 2
                            if abs(doubled - round(doubled)) < 1e-9:
                                row.append(float(round(doubled)) / 2.0)
                            else:
                                row.append(float(val))
                    U_list.append(row)
                
                V_list = []
                for i in range(len(V)):
                    row = []
                    for r in range(len(V[0])):
                        val = V[i][r]
                        if abs(val - round(val)) < 1e-9:
                            row.append(int(round(val)))
                        else:
                            doubled = val * 2
                            if abs(doubled - round(doubled)) < 1e-9:
                                row.append(float(round(doubled)) / 2.0)
                            else:
                                row.append(float(val))
                    V_list.append(row)
                
                W_list = []
                for i in range(len(W)):
                    row = []
                    for r in range(len(W[0])):
                        val = W[i][r]
                        if abs(val - round(val)) < 1e-9:
                            row.append(int(round(val)))
                        else:
                            doubled = val * 2
                            if abs(doubled - round(doubled)) < 1e-9:
                                row.append(float(round(doubled)) / 2.0)
                            else:
                                row.append(float(val))
                    W_list.append(row)
                
                cand_better = {"U": U_list, "V": V_list, "W": W_list}
                if _verify_decomposition(cand_better["U"], cand_better["V"], cand_better["W"], n, m, p):
                    return cand_better
        
        return cand1
        
    # Candidate 2: Rank 48 Schoolbook (Fallback)
    cand2 = _construct_schoolbook()
    if _verify_decomposition(cand2["U"], cand2["V"], cand2["W"], n, m, p):
        return cand2
        
    # Should not happen, but return schoolbook regardless if verification fails due to bug
    return cand2
