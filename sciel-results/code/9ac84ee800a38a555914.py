import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """Return U2, V2, W2 as (7, 4) arrays for 2x2x2 Strassen decomposition."""
    # Row-major flattening: index = r*2 + c
    U2 = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ], dtype=int)
    
    return U2, V2, W2

_U2, _V2, _W2 = _strassen_2x2_factors()

def _build_4x4x4_factors():
    """Build rank-49 decomposition for 4x4x4 via Kronecker product of two Strassen."""
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U4[I, J, col] = U2[s, I//2, J//2] * U2[t, I%2, J%2]
            for I in range(4):
                for J in range(4):
                    r1, c1 = I // 2, J // 2
                    r0, c0 = I % 2, J % 2
                    U4[I * 4 + J, col] = _U2[s, r1 * 2 + c1] * _U2[t, r0 * 2 + c0]
            
            # V4[J, K, col] = V2[s, J//2, K//2] * V2[t, J%2, K%2]
            for J in range(4):
                for K in range(4):
                    r1, c1 = J // 2, K // 2
                    r0, c0 = J % 2, K % 2
                    V4[J * 4 + K, col] = _V2[s, r1 * 2 + c1] * _V2[t, r0 * 2 + c0]
            
            # W4[I, K, col] = W2[s, I//2, K//2] * W2[t, I%2, K%2]
            for I in range(4):
                for K in range(4):
                    r1, c1 = I // 2, K // 2
                    r0, c0 = I % 2, K % 2
                    W4[I * 4 + K, col] = _W2[s, r1 * 2 + c1] * _W2[t, r0 * 2 + c0]
    
    return U4, V4, W4

_U4, _V4, _W4 = _build_4x4x4_factors()

def _build_4x4x8():
    """Build rank-98 decomposition for 4x4x8 by duplicating 4x4x4 factors."""
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # Left block (k=0..3)
    U[:, 0:49] = _U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = _V4[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = _W4[i * 4 + k]
    
    # Right block (k=4..7)
    U[:, 49:98] = _U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = _V4[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = _W4[i * 4 + k]
    
    return U, V, W

def _verify_fast(U, V, W, atol=1e-10):
    """Fast verification using einsum."""
    n, m, p = 4, 4, 8
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank).astype(float)
    V3 = V.reshape(m, p, rank).astype(float)
    W3 = W.reshape(p, n, rank).astype(float)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=atol)

def _verify_sampled(U, V, W, sample_indices=None, atol=1e-10):
    """Verify only a subset of tensor entries for speed."""
    n, m, p = 4, 4, 8
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank).astype(float)
    V3 = V.reshape(m, p, rank).astype(float)
    W3 = W.reshape(p, n, rank).astype(float)
    
    if sample_indices is None:
        # Use all entries
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=atol)
    else:
        for (i, j, k) in sample_indices:
            val = np.einsum('r,r,r->', U3[i, j, :], V3[j, k, :], W3[k, i, :])
            if abs(val - 1.0) > atol:
                return False
        return True

def _try_reduce_fast(U, V, W, deadline, atol=1e-10):
    """Try to remove redundant rank-1 columns using fast verification."""
    n, m, p = 4, 4, 8
    rank = U.shape[1]
    
    # Precompute all tensor entry values for the full decomposition
    U3 = U.reshape(n, m, rank).astype(float)
    V3 = V.reshape(m, p, rank).astype(float)
    W3 = W.reshape(p, n, rank).astype(float)
    
    # For each (i,j,k), compute the contribution of each rank column
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # We can precompute for each (i,j,k) the vector of contributions per rank
    
    # To be efficient, precompute per-rank contributions
    # contrib[i,j,k,r] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # But that's 4*4*8*98 = 12544 values, manageable
    
    # Actually, let's just do the full verification but cache intermediate results
    # For speed, we'll check a random subset of tensor entries first
    
    # Create a list of all (i,j,k) tuples
    all_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Try removing each column, starting from the end
    keep = list(range(rank))
    
    for idx in range(len(keep) - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
        
        col_to_remove = keep[idx]
        new_keep = keep[:idx] + keep[idx+1:]
        
        if len(new_keep) == 0:
            continue
        
        U_t = U[:, new_keep]
        V_t = V[:, new_keep]
        W_t = W[:, new_keep]
        
        # Quick check: verify a sample of tensor entries
        # Use all entries for safety, but this is the expensive part
        if _verify_fast(U_t, V_t, W_t, atol=atol):
            keep = new_keep
    
    if len(keep) == rank:
        return U, V, W
    return U[:, keep], V[:, keep], W[:, keep]

def _schoolbook():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build rank-98 decomposition
    U, V, W = _build_4x4x8()
    
    # Verify the base construction
    if not _verify_fast(U, V, W, atol=1e-10):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 15.0:
        U, V, W = _try_reduce_fast(U, V, W, deadline, atol=1e-10)
    
    # Final safety check
    if not _verify_fast(U, V, W, atol=1e-10):
        U, V, W = _schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
