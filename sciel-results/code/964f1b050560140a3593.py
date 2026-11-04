import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0
_MARGIN = 5.0


def _time_ok():
    return time.time() < _DEADLINE - _MARGIN


def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int8)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int8)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int8)
    
    R = 49
    
    # Build 4x4 decomposition via Kronecker product of 2x2 Strassen
    # Index mapping: for 4x4, i = i1*2 + i2, j = j1*2 + j2
    # U_4[i*4+j, a*7+b] = U2[i1*2+j1, a] * U2[i2*2+j2, b]
    
    is_ = np.array([0, 1, 2, 3])
    js = np.array([0, 1, 2, 3])
    ks = np.array([0, 1, 2, 3])
    
    i1s = is_ // 2
    i2s = is_ % 2
    j1s = js // 2
    j2s = js % 2
    k1s = ks // 2
    k2s = ks % 2
    
    # U construction: U[i,j] = outer(U2[idx1], U2[idx2]) where idx1=i1*2+j1, idx2=i2*2+j2
    idx1_u = i1s[:, None] * 2 + j1s[None, :]  # (4,4)
    idx2_u = i2s[:, None] * 2 + j2s[None, :]  # (4,4)
    
    U_p1 = U2[idx1_u]  # (4,4,7)
    U_p2 = U2[idx2_u]  # (4,4,7)
    U_tensor = U_p1[:, :, :, None] * U_p2[:, :, None, :]  # (4,4,7,7)
    U = U_tensor.reshape(16, 49).astype(int)
    
    if not _time_ok():
        return {"U": U.tolist(), "V": np.zeros((16, 49), dtype=int).tolist(), 
                "W": np.zeros((16, 49), dtype=int).tolist()}
    
    # V construction: V[j,k] = outer(V2[idx1], V2[idx2]) where idx1=j1*2+k1, idx2=j2*2+k2
    idx1_v = j1s[:, None] * 2 + k1s[None, :]  # (4,4)
    idx2_v = j2s[:, None] * 2 + k2s[None, :]  # (4,4)
    
    V_p1 = V2[idx1_v]  # (4,4,7)
    V_p2 = V2[idx2_v]  # (4,4,7)
    V_tensor = V_p1[:, :, :, None] * V_p2[:, :, None, :]  # (4,4,7,7)
    V = V_tensor.reshape(16, 49).astype(int)
    
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": np.zeros((16, 49), dtype=int).tolist()}
    
    # W construction: W[k,i] = outer(W2[idx1], W2[idx2]) where idx1=k1*2+i1, idx2=k2*2+i2
    idx1_w = k1s[:, None] * 2 + i1s[None, :]  # (4,4)
    idx2_w = k2s[:, None] * 2 + i2s[None, :]  # (4,4)
    
    W_p1 = W2[idx1_w]  # (4,4,7)
    W_p2 = W2[idx2_w]  # (4,4,7)
    W_tensor = W_p1[:, :, :, None] * W_p2[:, :, None, :]  # (4,4,7,7)
    W = W_tensor.reshape(16, 49).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
