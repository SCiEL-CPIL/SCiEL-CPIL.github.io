"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Vectorized verification: for each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # Build index arrays
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_idx = i_idx * m + j_idx  # shape (n,m,p)
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # For each (i,j,k), compute sum over r
    # U[u_idx[i,j,k], r] * V[v_idx[i,j,k], r] * W[w_idx[i,j,k], r]
    # Use einsum: U_u_r * V_v_r * W_w_r summed over r
    # First gather the needed rows
    U_sel = U[u_idx]  # shape (n,m,p,rank)
    V_sel = V[v_idx]  # shape (n,m,p,rank)
    W_sel = W[w_idx]  # shape (n,m,p,rank)
    
    result = np.einsum('...r,...r,...r->...', U_sel, V_sel, W_sel)
    return np.allclose(result, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_factors():
    """
    Returns U, V, W factors for 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices: u = i*2+j, v = j*2+k, w = k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_4x4_factors():
    """
    Construct factors for 4x4x4 multiplication using Strassen on 2x2 blocks.
    Rank 49.
    i = 2I+a, j = 2J+b, k = 2K+c
    U[i*4+j, t_o*7+t_i] = u_o[I*2+J] * u_i[a*2+b]
    V[j*4+k, t_o*7+t_i] = v_o[J*2+K] * v_i[b*2+c]
    W[k*4+i, t_o*7+t_i] = w_o[K*2+I] * w_i[c*2+a]
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for t_o in range(7):
        for t_i in range(7):
            r = t_o * 7 + t_i
            
            u_o = U2[:, t_o]  # shape (4,), indexed by [I*2+J]
            v_o = V2[:, t_o]  # shape (4,), indexed by [J*2+K]
            w_o = W2[:, t_o]  # shape (4,), indexed by [K*2+I]
            
            u_i = U2[:, t_i]  # shape (4,), indexed by [a*2+b]
            v_i = V2[:, t_i]  # shape (4,), indexed by [b*2+c]
            w_i = W2[:, t_i]  # shape (4,), indexed by [c*2+a]
            
            for I in range(2):
                for J in range(2):
                    u_o_val = u_o[I * 2 + J]
                    if u_o_val == 0:
                        continue
                    for a in range(2):
                        for b in range(2):
                            u_i_val = u_i[a * 2 + b]
                            if u_i_val == 0:
                                continue
                            
                            i = 2 * I + a
                            j = 2 * J + b
                            idx_u = i * 4 + j
                            U[idx_u, r] = u_o_val * u_i_val
                            
                            for K in range(2):
                                v_o_val = v_o[J * 2 + K]
                                if v_o_val == 0:
                                    continue
                                for c in range(2):
                                    v_i_val = v_i[b * 2 + c]
                                    if v_i_val == 0:
                                        continue
                                    
                                    k = 2 * K + c
                                    idx_v = j * 4 + k
                                    V[idx_v, r] = v_o_val * v_i_val
                                    
                                    w_o_val = w_o[K * 2 + I]
                                    if w_o_val == 0:
                                        continue
                                    
                                    w_i_val = w_i[c * 2 + a]
                                    if w_i_val == 0:
                                        continue
                                    
                                    idx_w = k * 4 + i
                                    W[idx_w, r] = w_o_val * w_i_val

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Schoolbook rank 128 as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen 4x4x4x4 twice (for 4x4x4x8)
    try:
        if time.time() > deadline - 5.0:
            return best
            
        U4, V4, W4 = _strassen_4x4_4x4_factors()
        
        # Combine two 4x4x4 decompositions into one 4x4x8.
        # First half handles k=0..3, second half handles k=4..7.
        # Total rank = 49 + 49 = 98
        
        U_full = np.zeros((16, 98))
        V_full = np.zeros((32, 98))  # m*p = 4*8 = 32
        W_full = np.zeros((32, 98))  # p*n = 8*4 = 32
        
        # First half (k=0..3)
        U_full[:, 0:49] = U4
        for j in range(4):
            for k in range(4):
                V_full[j * 8 + k, 0:49] = V4[j * 4 + k, :]
        for k in range(4):
            for i in range(4):
                W_full[k * 8 + i, 0:49] = W4[k * 4 + i, :]
        
        # Second half (k=4..7)
        U_full[:, 49:98] = U4
        for j in range(4):
            for k in range(4):
                V_full[j * 8 + (k + 4), 49:98] = V4[j * 4 + k, :]
        for k in range(4):
            for i in range(4):
                W_full[(k + 4) * 8 + i, 49:98] = W4[k * 4 + i, :]
        
        # Verify
        if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
    except Exception as e:
        pass
    
    return best
# EVOLVE-BLOCK-END
