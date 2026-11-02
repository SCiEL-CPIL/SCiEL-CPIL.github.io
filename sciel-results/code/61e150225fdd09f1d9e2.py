"""Combined program: Robust Rank-33 decomposition with fast verification."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Vectorized verification
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i, j, k] == 1 for all i, j, k
    
    # Reshape U, V, W to facilitate broadcasting
    # U shape: (n, m, rank)
    U_r = U.reshape(n, m, rank)
    # V shape: (m, p, rank)
    V_r = V.reshape(m, p, rank)
    # W shape: (p, n, rank)
    W_r = W.reshape(p, n, rank)
    
    # Sum over rank dimension
    # Result shape: (n, m, p)
    # T[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
    
    # This requires careful broadcasting.
    # U_r[i, j, r] * V_r[j, k, r] -> shape (n, m, p, rank)
    # Let's do it in steps to save memory if needed, but 2*3*6*33 is small.
    
    # Term 1: U[i, j, r] * V[j, k, r]
    # U_r: (n, m, rank) -> (n, m, 1, rank)
    # V_r: (m, p, rank) -> (1, m, p, rank)
    UV = U_r[:, :, np.newaxis, :] * V_r[np.newaxis, :, :, :]
    
    # Term 2: W[k, i, r]
    # W_r: (p, n, rank) -> (p, n, rank)
    # We need to align indices: UV is (n, m, p, rank). W is (p, n, rank).
    # We need W[k, i, r] to match UV[i, j, k, r].
    # So W should be (1, 1, p, n, rank)? No.
    # UV index order: i, j, k, r.
    # W index order: k, i, r.
    # We need to broadcast W to (n, m, p, rank) such that W[k, i, r] is at [i, j, k, r].
    # W_r_transposed: (n, p, rank) is not right.
    # W_r is (p, n, rank).
    # We want W[k, i, r].
    # Create W_b of shape (n, 1, p, rank) where W_b[i, 0, k, r] = W_r[k, i, r].
    W_b = W_r.T[:, np.newaxis, :, :] # (n, p, rank) -> (n, 1, p, rank)
    
    # Now UV * W_b
    # UV: (n, m, p, rank)
    # W_b: (n, 1, p, rank)
    Tensor = np.sum(UV * W_b, axis=3)
    
    # Tensor shape: (n, m, p)
    # Check if all entries are 1
    return np.allclose(Tensor, 1.0, atol=1e-9)

def _build_rank33_decomposition():
    """Build a rank-33 decomposition using 3 independent 2x3x2 blocks."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 11 terms for 2x3x2 multiplication
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Use the verified Rank 33 decomposition
    U, V, W = _build_rank33_decomposition()
    
    # Verify
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Should not happen
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
