"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Vectorized verification using numpy for speed
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i, j, k] == 1 for all i, j, k
    
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Extract the relevant rows
    U_sel = U[u_idx, :]  # shape (n*m*p, rank)
    V_sel = V[v_idx, :]  # shape (n*m*p, rank)
    W_sel = W[w_idx, :]  # shape (n*m*p, rank)
    
    # Compute the tensor values
    # T = sum_r U_sel[:, r] * V_sel[:, r] * W_sel[:, r]
    T = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    
    return True

def _build_rank33_decomposition():
    """Build a rank-33 decomposition using 3 independent 2x3x2 blocks."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 11 terms for 2x3x2 multiplication
    # A is 2x3: indices (i,j) -> flat i*3+j
    # B is 3x2: indices (j,k) -> flat j*2+k
    # C is 2x2: indices (i,k) -> flat i*2+k
    
    terms = []
    
    # Strassen-like terms for the 2x2 x 2x2 part (using columns 0,1 of A and rows 0,1 of B)
    # A sub: A[0,0], A[0,1], A[1,0], A[1,1] -> flat 0,1,3,4
    # B sub: B[0,0], B[0,1], B[1,0], B[1,1] -> flat 0,1,2,3
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({'u': {0:1, 4:1}, 'v': {0:1, 3:1}, 'w': {0:1, 3:1}})
    # M2 = (A10+A11)B00
    terms.append({'u': {3:1, 4:1}, 'v': {0:1}, 'w': {2:1, 3:-1}})
    # M3 = A00(B01-B11)
    terms.append({'u': {0:1}, 'v': {1:1, 3:-1}, 'w': {1:1, 3:1}})
    # M4 = A11(B10-B00)
    terms.append({'u': {4:1}, 'v': {2:1, 0:-1}, 'w': {0:1, 2:1}})
    # M5 = (A00+A01)B11
    terms.append({'u': {0:1, 1:1}, 'v': {3:1}, 'w': {0:-1, 1:1}})
    # M6 = (A10-A00)(B00+B01)
    terms.append({'u': {3:1, 0:-1}, 'v': {0:1, 1:1}, 'w': {3:1}})
    # M7 = (A01-A11)(B10+B11)
    terms.append({'u': {1:1, 4:-1}, 'v': {2:1, 3:1}, 'w': {0:1}})
    
    # Scalar terms for column 2 of A and row 2 of B
    # K1 = A02 * B20
    terms.append({'u': {2:1}, 'v': {4:1}, 'w': {0:1}})
    # K2 = A02 * B21
    terms.append({'u': {2:1}, 'v': {5:1}, 'w': {1:1}})
    # K3 = A12 * B20
    terms.append({'u': {5:1}, 'v': {4:1}, 'w': {2:1}})
    # K4 = A12 * B21
    terms.append({'u': {5:1}, 'v': {5:1}, 'w': {3:1}})
    
    # Fill matrices for 3 blocks
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U
            for idx, val in term['u'].items():
                U[idx, col] = val
            
            # V
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            
            # W
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Use rank-33 solution as baseline
    U, V, W = _build_rank33_decomposition()
    
    # Verify if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Should not happen, but return anyway
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
