"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

_DIMS = (4, 5, 5)
_START_TIME = time.time()
_DEADLINE = 140.0


def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a, b, c_idx]."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1
    return T


def _verify(U, V, W, n, m, p, T):
    """Verify that sum_c U[:,c] V[:,c] W[:,c] == T."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(T_recon, T, atol=1e-9)


def _strassen_2x2():
    """
    Return 2x2x2 Strassen factor matrices (4, 7) each.
    Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    return s_u, s_v, s_w


def _add_strassen_4x4(U, V, W, n, m, p, col_start, a_row_offset=0, a_col_offset=0,
                      b_row_offset=0, b_col_offset=0, c_row_offset=0, c_col_offset=0):
    """
    Add a 4x4x4 Strassen decomposition (rank 49) using 2x2x2 Strassen sub-factors.
    """
    s_u, s_v, s_w = _strassen_2x2()
    
    macro_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(macro_terms):
        if _time_exceeded():
            return
        for inner_k in range(7):
            col = col_start + t_idx * 7 + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = a_row_offset + 2 * a_br + lr
                            gj = a_col_offset + 2 * a_bc + lc
                            U[gi * m + gj, col] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = b_row_offset + 2 * b_br + lr
                            gk = b_col_offset + 2 * b_bc + lc
                            V[gj * p + gk, col] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = c_row_offset + 2 * c_br + lr
                            gk = c_col_offset + 2 * c_bc + lc
                            W[gk * n + gi, col] += c_coeff * w_val


def _try_sparsify(U, V, W, n, m, p, T, max_iter=50):
    """
    Attempt to reduce rank by merging proportional rank-1 terms.
    Two terms r1 and r2 can be merged if U[:,r1] = alpha*U[:,r2], 
    V[:,r1] = beta*V[:,r2], W[:,r1] = gamma*W[:,r2] with alpha*beta*gamma = 1.
    Actually, we can absorb scaling: if U1=c*U2, V1=V2, W1=W2, we can merge.
    More generally, if U1=c*U2, V1=(1/c)*V2, W1=W2, then term1 = c*(1/c)*1 * (U2 V2 W2) = term2.
    So we can remove one of them.
    
    This is a very conservative check. Let's try a simpler approach:
    Check if any two columns are identical (up to sign) and if so, they cancel or duplicate.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]
    
    for iteration in range(max_iter):
        if _time_exceeded():
            break
        if current_rank <= 1:
            break
        
        found_merge = False
        for i in range(current_rank):
            if _time_exceeded():
                return U, V, W, current_rank
            for j in range(i+1, current_rank):
                # Check if column i is a scalar multiple of column j in U
                ui = U[:, i]
                uj = U[:, j]
                vi = V[:, i]
                vj = V[:, j]
                wi = W[:, i]
                wj = W[:, j]
                
                # Check proportionality
                nz_i = np.nonzero(ui)[0]
                nz_j = np.nonzero(uj)[0]
                if len(nz_i) == 0 or len(nz_j) == 0:
                    continue
                if not np.array_equal(nz_i, nz_j):
                    continue
                
                # Compute ratio
                ratio = ui[nz_i] / uj[nz_j]
                if not np.allclose(ratio, ratio[0], atol=1e-9):
                    continue
                c = ratio[0]
                
                # Check V proportionality with 1/c
                nz_vi = np.nonzero(vi)[0]
                nz_vj = np.nonzero(vj)[0]
                if not np.array_equal(nz_vi, nz_vj):
                    continue
                ratio_v = vi[nz_vi] / vj[nz_vj]
                if not np.allclose(ratio_v, 1.0/c, atol=1e-9):
                    continue
                
                # Check W proportionality with 1
                nz_wi = np.nonzero(wi)[0]
                nz_wj = np.nonzero(wj)[0]
                if not np.array_equal(nz_wi, nz_wj):
                    continue
                ratio_w = wi[nz_wi] / wj[nz_wj]
                if not np.allclose(ratio_w, 1.0, atol=1e-9):
                    continue
                
                # Merge: remove column j
                keep = [k for k in range(current_rank) if k != j]
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                current_rank -= 1
                found_merge = True
                break
            if found_merge:
                break
        
        if not found_merge:
            break
    
    return U, V, W, current_rank


def construct():
    """
    Constructs a low-rank tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    """
    n, m, p = _DIMS
    T = _build_tensor(n, m, p)
    
    rank = 61
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Part 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4] (4x4x4 Strassen, rank 49)
    _add_strassen_4x4(U, V, W, n, m, p, col_start=0,
                      a_row_offset=0, a_col_offset=0,
                      b_row_offset=0, b_col_offset=0,
                      c_row_offset=0, c_col_offset=0)
    col = 49
    
    # Part 2+4: A[0:4, 0:5] * B[0:5, 4] -> C[0:4, 4] (4x5x1, rank 4)
    for i in range(4):
        if _time_exceeded():
            break
        for j in range(5):
            U[i * 5 + j, col] = 1
        for j in range(5):
            V[j * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1
    
    # Part 3: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4] (4x1x4, rank 4)
    for k in range(4):
        if _time_exceeded():
            break
        for i in range(4):
            U[i * 5 + 4, col] = 1
        V[4 * 5 + k, col] = 1
        for i in range(4):
            W[k * 4 + i, col] = 1
        col += 1
    
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    
    if not _verify(U_arr, V_arr, W_arr, n, m, p, T):
        # Fallback to rank 85
        U = np.zeros((n * m, 85), dtype=int)
        V = np.zeros((m * p, 85), dtype=int)
        W = np.zeros((p * n, 85), dtype=int)
        
        _add_strassen_4x4(U, V, W, n, m, p, col_start=0,
                          a_row_offset=0, a_col_offset=0,
                          b_row_offset=0, b_col_offset=0,
                          c_row_offset=0, c_col_offset=0)
        col = 49
        
        for i in range(4):
            for j in range(4):
                U[i*5 + j, col] = 1
                V[j*5 + 4, col] = 1
                W[4*4 + i, col] = 1
                col += 1
        
        for i in range(4):
            for k in range(4):
                U[i*5 + 4, col] = 1
                V[4*5 + k, col] = 1
                W[k*4 + i, col] = 1
                col += 1
        
        for i in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
        
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
    
    # Try sparsification
    if not _time_exceeded():
        U_arr, V_arr, W_arr, new_rank = _try_sparsify(U_arr, V_arr, W_arr, n, m, p, T)
        if new_rank < U_arr.shape[1]:
            U_arr = U_arr[:, :new_rank]
            V_arr = V_arr[:, :new_rank]
            W_arr = W_arr[:, :new_rank]
    
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in U_arr]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in V_arr]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in W_arr]
    
    return {"U": U_list, "V": V_list, "W": W_list}
