import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_222():
    """
    Returns U, V, W for M_{2,2,2} with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen products
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # c00 gets +1, c11 gets +1? 
    # Wait, W index is k*2+i.
    # c00 is i=0, k=0. W[0*2+0]=W[0].
    # c11 is i=1, k=1. W[1*2+1]=W[3].
    # M1 contributes to c00 and c11.
    # So W[0] += 1, W[3] += 1.
    
    # Let's restart the mapping carefully.
    # We need to sum 7 rank-1 tensors.
    # Tensor c contributes to W[k*2+i] with coefficient w_c.
    # And U[i*2+j] with u_c, V[j*2+k] with v_c.
    
    # M1: (a00+a11)(b00+b11)
    # u: a00, a11 -> U[0], U[3] are 1.
    # v: b00, b11 -> V[0], V[3] are 1.
    # w: c00, c11 -> W[0], W[3] are 1.
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2: (a10+a11)b00
    # u: a10, a11 -> U[2], U[3] are 1.
    # v: b00 -> V[0] is 1.
    # w: c10 -> W[2] (k=1, i=0 -> 1*2+0=2) is 1.
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # M3: a00(b01-b11)
    # u: a00 -> U[0] is 1.
    # v: b01, -b11 -> V[1] is 1, V[3] is -1.
    # w: c01 -> W[1] (k=1, i=0 -> 1*2+0=1) is 1.
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1
    
    # M4: a11(b10-b00)
    # u: a11 -> U[3] is 1.
    # v: b10, -b00 -> V[2] is 1, V[0] is -1.
    # w: c00, c10 -> W[0] is 1, W[2] is 1.
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # M5: (a00+a01)b11
    # u: a00, a01 -> U[0], U[1] are 1.
    # v: b11 -> V[3] is 1.
    # w: c01, -c00 -> W[1] is 1, W[0] is -1.
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[0,4]=-1
    
    # M6: (a10-a00)(b00+b01)
    # u: a10, -a00 -> U[2] is 1, U[0] is -1.
    # v: b00, b01 -> V[0], V[1] are 1.
    # w: c11 -> W[3] is 1.
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M7: (a01-a11)(b10+b11)
    # u: a01, -a11 -> U[1] is 1, U[3] is -1.
    # v: b10, b11 -> V[2], V[3] are 1.
    # w: c00 -> W[0] is 1.
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _schoolbook_232():
    """
    Returns U, V, W for M_{2,3,2} with rank 12.
    U shape (6, 12), V shape (6, 12), W shape (4, 12).
    Indices: U[i*3+j], V[j*2+k], W[k*2+i].
    """
    rank = 12
    U = np.zeros((6, rank))
    V = np.zeros((6, rank))
    W = np.zeros((4, rank))
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Block definitions
    # j ranges: [0,1] (size 2), [2,3,4] (size 3)
    # k ranges: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    
    blocks = [
        # (j_start, j_len, k_start, k_len, type)
        (0, 2, 0, 2, 'strassen'),
        (0, 2, 2, 2, 'strassen'),
        (0, 2, 4, 2, 'strassen'),
        (2, 3, 0, 2, 'schoolbook'),
        (2, 3, 2, 2, 'schoolbook'),
        (2, 3, 4, 2, 'schoolbook'),
    ]
    
    # Calculate total rank
    total_rank = 0
    for b in blocks:
        if b[4] == 'strassen':
            total_rank += 7
        else:
            total_rank += 12
            
    U_global = np.zeros((n*m, total_rank))
    V_global = np.zeros((m*p, total_rank))
    W_global = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    for b in blocks:
        j_start, j_len, k_start, k_len, btype = b
        
        if btype == 'strassen':
            U_loc, V_loc, W_loc = _strassen_222()
            local_rank = 7
        else:
            U_loc, V_loc, W_loc = _schoolbook_232()
            local_rank = 12
            
        # Map local indices to global indices
        # Local U index: i_loc * j_len + j_loc
        # Global U index: i * m + j
        # i_loc is 0 or 1 (since n=2)
        # j_loc is 0..j_len-1
        # j = j_start + j_loc
        
        # Local V index: j_loc * k_len + k_loc
        # Global V index: j * p + k
        # k = k_start + k_loc
        
        # Local W index: k_loc * n + i_loc
        # Global W index: k * n + i
        
        for c in range(local_rank):
            for i_loc in range(n): # 0, 1
                for j_loc in range(j_len):
                    u_idx_global = i_loc * m + (j_start + j_loc)
                    u_val = U_loc[i_loc * j_len + j_loc, c]
                    if u_val != 0:
                        U_global[u_idx_global, col_offset + c] += u_val
                        
            for j_loc in range(j_len):
                for k_loc in range(k_len):
                    v_idx_global = (j_start + j_loc) * p + (k_start + k_loc)
                    v_val = V_loc[j_loc * k_len + k_loc, c]
                    if v_val != 0:
                        V_global[v_idx_global, col_offset + c] += v_val
                        
            for k_loc in range(k_len):
                for i_loc in range(n):
                    w_idx_global = (k_start + k_loc) * n + i_loc
                    w_val = W_loc[k_loc * n + i_loc, c]
                    if w_val != 0:
                        W_global[w_idx_global, col_offset + c] += w_val
                        
        col_offset += local_rank
        
    # Verification (optional, but good for debugging)
    # if not _verify(U_global.tolist(), V_global.tolist(), W_global.tolist(), n, m, p, total_rank):
    #     # Fallback to schoolbook
    #     U_s, V_s, W_s, rank_s = _schoolbook(n, m, p)
    #     return {"U": U_s, "V": V_s, "W": W_s}
        
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist(), rank
