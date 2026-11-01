import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Rank 44 decomposition: 4 blocks of rank 11
    U, V, W = _four_block_decomposition(n, m, p)
    
    # Verify
    if _verify_fast(n, m, p, U, V, W):
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Fallback to schoolbook rank 48
    U0, V0, W0 = _schoolbook(n, m, p)
    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _four_block_decomposition(n, m, p):
    n, m, p = 2, 3, 8
    block_size = 2
    rank_per_block = 11
    total_rank = 44

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for b in range(4):
        k_offset = b * block_size
        col_offset = b * rank_per_block

        # Strassen 7 terms for 2x2 * 2x2 (A_left, B_top)
        # A_left: rows 0,1; cols 0,1
        # B_top: rows 0,1; cols k_offset, k_offset+1
        
        # M1: (A00+A11)(B00+B11)
        U[0, col_offset+0] += 1; U[4, col_offset+0] += 1
        V[0*p+k_offset, col_offset+0] += 1; V[1*p+k_offset+1, col_offset+0] += 1
        W[(k_offset+0)*n+0, col_offset+0] += 1; W[(k_offset+1)*n+1, col_offset+0] += 1
        
        # M2: (A10+A11)B00
        U[3, col_offset+1] += 1; U[4, col_offset+1] += 1
        V[0*p+k_offset, col_offset+1] += 1
        W[(k_offset+0)*n+1, col_offset+1] += 1
        
        # M3: A00(B10-B11)
        U[0, col_offset+2] += 1
        V[1*p+k_offset, col_offset+2] += 1; V[1*p+k_offset+1, col_offset+2] -= 1
        W[(k_offset+1)*n+0, col_offset+2] += 1; W[(k_offset+1)*n+1, col_offset+2] += 1
        
        # M4: A11(B01-B00)
        U[4, col_offset+3] += 1
        V[0*p+k_offset+1, col_offset+3] += 1; V[0*p+k_offset, col_offset+3] -= 1
        W[(k_offset+0)*n+0, col_offset+3] += 1; W[(k_offset+0)*n+1, col_offset+3] += 1
        
        # M5: (A00+A01)B11
        U[0, col_offset+4] += 1; U[1, col_offset+4] += 1
        V[1*p+k_offset+1, col_offset+4] += 1
        W[(k_offset+0)*n+0, col_offset+4] -= 1; W[(k_offset+1)*n+0, col_offset+4] += 1
        
        # M6: (A10-A00)(B00+B01)
        U[3, col_offset+5] += 1; U[0, col_offset+5] -= 1
        V[0*p+k_offset, col_offset+5] += 1; V[0*p+k_offset+1, col_offset+5] += 1
        W[(k_offset+1)*n+1, col_offset+5] += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1, col_offset+6] += 1; U[4, col_offset+6] -= 1
        V[1*p+k_offset, col_offset+6] += 1; V[1*p+k_offset+1, col_offset+6] += 1
        W[(k_offset+0)*n+0, col_offset+6] += 1

        # Outer product 4 terms: A_right(2x1) * B_bottom(1x2)
        # A_right: rows 0,1; col 2
        # B_bottom: row 2; cols k_offset, k_offset+1
        
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1

    return U, V, W

def _verify_fast(n, m, p, U, V, W):
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Vectorized computation
    # C_comp[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_l V[l*p+k, c] B[l,k])
    
    # Reshape U, V, W for broadcasting
    # U: (n*m, rank) -> (n, m, rank)
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # u_val[i, j, c] = U[i, j, c] * A[i, j]
    u_val = U_reshaped * A[:, :, np.newaxis]
    
    # v_val[l, k, c] = V[l, k, c] * B[l, k]
    v_val = V_reshaped * B[:, :, np.newaxis]
    
    # Sum over j for u: sum_j U[i,j,c] A[i,j] -> shape (n, rank)
    u_sum = np.sum(u_val, axis=1) # (n, rank)
    
    # Sum over l for v: sum_l V[l,k,c] B[l,k] -> shape (p, rank)
    v_sum = np.sum(v_val, axis=0) # (p, rank)
    
    # C_comp[i, k] = sum_c W[k, i, c] * u_sum[i, c] * v_sum[k, c]
    # W_reshaped is (p, n, rank)
    # We need W[k, i, c] * u_sum[i, c] * v_sum[k, c]
    
    # Expand dims for broadcasting
    # u_sum: (n, 1, rank)
    # v_sum: (1, p, rank)
    # W: (p, n, rank)
    
    term = W_reshaped * u_sum[:, np.newaxis, :] * v_sum[np.newaxis, :, :]
    
    # Sum over c (rank)
    C_comp = np.sum(term, axis=2)
    
    # C_comp is (p, n). Need (n, p).
    C_comp = C_comp.T
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True

def _to_list(arr):
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result
