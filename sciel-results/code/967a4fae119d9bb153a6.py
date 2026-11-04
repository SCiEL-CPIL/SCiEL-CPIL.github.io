import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _build_strassen_4x4x4():
    """Build rank-49 Strassen decomposition for 4x4x4."""
    n, m, p = 4, 4, 4
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    sub_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    cm_coeffs_list = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    c_block_coeffs_list = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    U = []
    V = []
    W = []
    for m_idx in range(7):
        x_def, y_def = strassen_terms[m_idx]
        for t_idx in range(7):
            x_sub, y_sub = sub_terms[t_idx]
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            w_coeffs = [0.0] * (p*n)
            cm_coeffs = cm_coeffs_list[t_idx]
            c_block_coeffs = c_block_coeffs_list[m_idx]
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            U.append(u_coeffs)
            V.append(v_coeffs)
            W.append(w_coeffs)
    return np.array(U).T, np.array(V).T, np.array(W).T

def _build_2x2x3_decomposition():
    """Build rank-11 decomposition for 2x2x3."""
    n, m, p = 2, 2, 3
    strassen_2x2x2_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    cm_coeffs_list = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    schoolbook_terms = [
        ({(0,0): 1}, {(0,0): 1}, (0,0)),
        ({(0,1): 1}, {(1,0): 1}, (0,0)),
        ({(1,0): 1}, {(0,0): 1}, (1,0)),
        ({(1,1): 1}, {(1,0): 1}, (1,0))
    ]
    
    U = []
    V = []
    W = []
    for t_idx in range(7):
        x_sub, y_sub = strassen_2x2x2_terms[t_idx]
        u_coeffs = [0.0] * (n*m)
        v_coeffs = [0.0] * (m*p)
        for (li, lj), c_x in x_sub.items():
            i = li
            j = lj
            u_idx = i * m + j
            u_coeffs[u_idx] += c_x
        for (li, lj), c_y in y_sub.items():
            i = li
            j = lj
            v_idx = i * p + j
            v_coeffs[v_idx] += c_y
        w_coeffs = [0.0] * (p*n)
        cm_coeffs = cm_coeffs_list[t_idx]
        for (li, lj), cm_c in cm_coeffs.items():
            if cm_c == 0.0: continue
            i_final = li
            k_final = lj
            w_idx = k_final * n + i_final
            w_coeffs[w_idx] += cm_c
        U.append(u_coeffs)
        V.append(v_coeffs)
        W.append(w_coeffs)
    
    for t_idx in range(4):
        x_sub, y_sub, c_pos = schoolbook_terms[t_idx]
        u_coeffs = [0.0] * (n*m)
        v_coeffs = [0.0] * (m*p)
        for (li, lj), c_x in x_sub.items():
            i = li
            j = lj
            u_idx = i * m + j
            u_coeffs[u_idx] += c_x
        for (li, lj), c_y in y_sub.items():
            i = li
            j = 2
            v_idx = i * p + j
            v_coeffs[v_idx] += c_y
        w_coeffs = [0.0] * (p*n)
        i_final = c_pos[0]
        k_final = 2
        w_idx = k_final * n + i_final
        w_coeffs[w_idx] = 1.0
        U.append(u_coeffs)
        V.append(v_coeffs)
        W.append(w_coeffs)
    
    return np.array(U).T, np.array(V).T, np.array(W).T

def _place_2x2x3_in_4x4x3(U_2x2x3, V_2x2x3, W_2x2x3, a_row_block, a_col_block, b_row_block, c_row_block):
    """Place a 2x2x3 decomposition into the 4x4x3 tensor at specified block positions."""
    n, m, p = 4, 4, 7
    U = np.zeros((n*m, U_2x2x3.shape[1]))
    V = np.zeros((m*p, V_2x2x3.shape[1]))
    W = np.zeros((p*n, W_2x2x3.shape[1]))
    
    for r in range(U_2x2x3.shape[1]):
        for u_idx in range(4):
            i = u_idx // 2
            j = u_idx % 2
            val = U_2x2x3[u_idx, r]
            if val != 0:
                I = 2 * a_row_block + i
                J = 2 * a_col_block + j
                U[I * m + J, r] = val
        
        for v_idx in range(6):
            i = v_idx // 3
            j = v_idx % 3
            val = V_2x2x3[v_idx, r]
            if val != 0:
                I = 2 * b_row_block + i
                J = 4 + j
                V[I * p + J, r] = val
        
        for w_idx in range(6):
            k = w_idx // 2
            i = w_idx % 2
            val = W_2x2x3[w_idx, r]
            if val != 0:
                K = 4 + k
                I = 2 * c_row_block + i
                W[K * n + I, r] = val
    
    return U, V, W

def _build_structured_decomposition():
    """Build the structured rank-93 decomposition."""
    n, m, p = 4, 4, 7
    
    U1, V1, W1 = _build_strassen_4x4x4()
    
    U_2x2x3, V_2x2x3, W_2x2x3 = _build_2x2x3_decomposition()
    
    placements = [
        (0, 0, 0, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (1, 1, 1, 1),
    ]
    
    U_parts = [U1]
    V_parts = [V1]
    W_parts = [W1]
    
    for a_row_block, a_col_block, b_row_block, c_row_block in placements:
        U_placed, V_placed, W_placed = _place_2x2x3_in_4x4x3(
            U_2x2x3, V_2x2x3, W_2x2x3,
            a_row_block, a_col_block, b_row_block, c_row_block
        )
        U_parts.append(U_placed)
        V_parts.append(V_placed)
        W_parts.append(W_placed)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that U, V, W correctly decompose the 4x4x7 matrix multiplication tensor."""
    n, m, p = 4, 4, 7
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check it equals delta[i,i'] * delta[j,j'] * delta[k,k']
    
    # For each output position (i, k), check that the linear map from input (j) is correct
    # T[i,j,k] should be 1 if i==i', j==j', k==k' for the appropriate indices
    # Actually, the matrix multiplication tensor is: T[i,j,k] = 1 if the multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k] means T[i,j,k] = 1 for all valid (i,j,k)
    
    # The tensor for matmul: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's verify by computing the actual matmul
    # For random A, B, check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] gives correct result
    
    # Simpler: check the tensor directly
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all valid i,j,k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
    
    return max_err

def _als_reduce_rank(U, V, W, target_rank, max_iter=500, tol=1e-10):
    """
    Try to reduce rank from current to target_rank using alternating least squares
    with exact rational reconstruction.
    """
    n, m, p = 4, 4, 7
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W
    
    # Initialize with random projection of existing factors
    np.random.seed(42)
    P = np.random.randn(current_rank, target_rank)
    U_new = U @ P
    V_new = V @ P
    W_new = W @ P
    
    # ALS iterations
    for it in range(max_iter):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
        
        # Update U: for each column i*m+j, solve least squares
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should give tensor
        # For fixed (i,j), we need: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # This is a linear system in U[i*m+j, :]
        
        for i in range(n):
            for j in range(m):
                # Build the system: for each k, sum_r U[u_idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is: U[u_idx, :] @ diag(V[j*p+k,:]*W[k*n+i,:]) ... no
                # It's: for each k, dot(U[u_idx,:], V[j*p+k,:] * W[k*n+i,:]) = 1
                # So: M @ U[u_idx,:] = 1 where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                M = V[j*p:(j*p+p), :] * W[(i):(i+p*n):n, :]  # shape (p, rank)
                # Actually W[k*n+i, r] for k=0..p-1
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve M @ u = 1
                try:
                    u_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = u_sol
                except:
                    pass
        
        # Update V similarly
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = v_sol
                except:
                    pass
        
        # Update W similarly
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = w_sol
                except:
                    pass
        
        # Check convergence
        max_err = _verify_decomposition(U_new, V_new, W_new)
        if max_err < tol:
            break
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS
    
    # Build the structured rank-93 decomposition
    U, V, W = _build_structured_decomposition()
    
    current_rank = U.shape[1]
    
    # Try to reduce rank using ALS
    # Start from slightly below current and work down
    for target_rank in range(current_rank - 1, max(60, current_rank - 20), -1):
        if time.time() - _START_TIME > TIME_LIMIT - 20:
            break
        
        U_try, V_try, W_try = _als_reduce_rank(U, V, W, target_rank, max_iter=200, tol=1e-10)
        
        # Verify
        max_err = _verify_decomposition(U_try, V_try, W_try)
        if max_err < 1e-8:
            U, V, W = U_try, V_try, W_try
            current_rank = target_rank
    
    # Convert to lists
    U_list = [[float(U[i, r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
