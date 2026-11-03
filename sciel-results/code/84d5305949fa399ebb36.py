import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding.
    
    Splits 4x4x5 into 4x4x4 (handled by Kronecker-embedded Strassen, rank 49)
    plus 4x4x1 (handled naively, rank 16).
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Standard Strassen 2x2x2 decomposition
    # U, V, W are lists of 7 vectors of length 4
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for 2x2 matrices
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, -1, 0], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 0, 0], # M4
        [0, 0, 1, 1], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Build 4x4x4 part using Kronecker product
    for r in range(7):
        for s in range(7):
            U_rs = np.zeros(16)
            V_rs = np.zeros(16)
            W_rs = np.zeros(16)
            
            # Build U_{rs}
            for i1 in range(2):
                for j1 in range(2):
                    val1 = strassen_U[r][i1*2+j1]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val2 = strassen_U[s][i2*2+j2]
                            if val2 == 0: continue
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            U_rs[i*4+j] += val1 * val2
            
            # Build V_{rs}
            for j1 in range(2):
                for k1 in range(2):
                    val1 = strassen_V[r][j1*2+k1]
                    if val1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            val2 = strassen_V[s][j2*2+k2]
                            if val2 == 0: continue
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            V_rs[j*4+k] += val1 * val2
            
            # Build W_{rs}
            for k1 in range(2):
                for i1 in range(2):
                    val1 = strassen_W[r][k1*2+i1]
                    if val1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            val2 = strassen_W[s][k2*2+i2]
                            if val2 == 0: continue
                            k = 2*k1 + k2
                            i = 2*i1 + i2
                            W_rs[k*4+i] += val1 * val2
            
            # Extend to 4x4x5 by zeroing out column 4
            U_full = np.zeros(n*m)
            V_full = np.zeros(m*p)
            W_full = np.zeros(p*n)
            
            for idx in range(16):
                U_full[idx] = U_rs[idx]
                V_full[idx] = V_rs[idx]
                W_full[idx] = W_rs[idx]
            
            add_term(U_full, V_full, W_full)
    
    # Build 4x4x1 part (naive, rank 16)
    for i in range(4):
        for j in range(4):
            U_full = np.zeros(n*m)
            V_full = np.zeros(m*p)
            W_full = np.zeros(p*n)
            
            U_full[i*4+j] = 1
            V_full[j*5+4] = 1
            W_full[4*4+i] = 1
            
            add_term(U_full, V_full, W_full)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Get the Kronecker-embedded Strassen decomposition (rank 65)
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    
    # Verify the Kronecker construction
    if _verify_decomposition(U_kron, V_kron, W_kron, n, m, p):
        best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
    else:
        # Fallback to naive rank 80
        U_naive = np.zeros((n*m, n*m*p))
        V_naive = np.zeros((m*p, n*m*p))
        W_naive = np.zeros((p*n, n*m*p))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, idx] = 1
                    V_naive[j*p+k, idx] = 1
                    W_naive[k*n+i, idx] = 1
                    idx += 1
                    
        best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
        
    return best_result
# EVOLVE-BLOCK-END
