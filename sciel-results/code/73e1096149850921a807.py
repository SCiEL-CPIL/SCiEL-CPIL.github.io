"""Matrix multiplication 4x4 x 4x8 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a rank-1 tensor contribution.
    """
    terms = []
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using recursive Strassen.
    Returns U (16,49), V (16,49), W (16,49) with integer entries.
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    strassen_terms = _strassen_2x2_terms()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for meta in strassen_terms:
        for inner in strassen_terms:
            # U factor: A[bi*2+ii, bj*2+jj]
            for (bi, bj), cu in meta['U'].items():
                for (ii, jj), cu_i in inner['U'].items():
                    row = bi * 2 + ii
                    c = bj * 2 + jj
                    U[row * 4 + c][col] += cu * cu_i
            # V factor: B[bj*2+jj, bk*2+kk]
            for (bj, bk), cv in meta['V'].items():
                for (jj, kk), cv_i in inner['V'].items():
                    row = bj * 2 + jj
                    c = bk * 2 + kk
                    V[row * 4 + c][col] += cv * cv_i
            # W factor: C[bi*2+ii, bk*2+kk] -> W[k*4+i]
            for (bi, bk), cw in meta['W'].items():
                for (ii, kk), cw_i in inner['W'].items():
                    i = bi * 2 + ii
                    k = bk * 2 + kk
                    W[k * 4 + i][col] += cw * cw_i
            col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Tensor: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # For matmul: T[i,j,k] should be 1 for all valid (i,j,k) in the standard tensor
    # Actually, the matmul tensor M[i,j,k] = 1 means C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta for the structure
    
    # Better check: for random A, B, verify C = A@B
    rng = np.random.RandomState(42)
    A = rng.randint(-5, 6, size=(n, m)).astype(float)
    B = rng.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(len(U[0])):
        # u_r is a vector in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
        u = U_mat[:, r].reshape(n, m)
        v = V_mat[:, r].reshape(m, p)
        w = W_mat[:, r].reshape(p, n)
        # Contribution: (A^T * u) * (v^T * B) ... no
        # Actually: scalar_r = sum_ij A[i,j]*U[i,j,r] * sum_jk B[j,k]*V[j,k,r]
        # Then C[i,k] += scalar_r * W[k,i,r]
        a_proj = np.sum(A * u)
        b_proj = np.sum(B * v)
        C_computed += a_proj * b_proj * w.T
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def construct():
    """Construct rank-minimal decomposition for 4x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strategy: Split p=8 into two halves of 4, use Strassen for each 4x4x4
    # This gives rank 49*2 = 98
    U_444, V_444, W_444 = _build_4x4x4_strassen()
    
    # Convert to numpy arrays for easier manipulation
    U_444_np = np.array(U_444, dtype=float)
    V_444_np = np.array(V_444, dtype=float)
    W_444_np = np.array(W_444, dtype=float)
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First half: C_left = A @ B_left
    # U: same as U_444
    U[:, :49] = U_444_np
    
    # V: j*8+k where k in 0..3 (left half)
    # V_444 is (16, 49) with index j*4+k
    # We need to map j*4+k -> j*8+k
    for r in range(49):
        for row in range(16):
            j = row // 4
            k = row % 4
            V[j * 8 + k, r] = V_444_np[row, r]
            
    # W: k*4+i where k in 0..3 (left half of output)
    # W_444 is (16, 49) with index k*4+i
    # We need to map k*4+i -> k*4+i (since k in 0..3, index is same)
    W[:16, :49] = W_444_np
    
    # Second half: C_right = A @ B_right
    # U: same as U_444
    U[:, 49:] = U_444_np
    
    # V: j*8+k where k in 4..7 (right half)
    for r in range(49):
        for row in range(16):
            j = row // 4
            k = row % 4
            V[j * 8 + (k + 4), 49 + r] = V_444_np[row, r]
            
    # W: k*4+i where k in 4..7 (right half of output)
    # W_444 index k*4+i. For k in 4..7, we shift k by 4.
    # New index (k+4)*4 + i = k*4 + i + 16
    W[16:, 49:] = W_444_np
    
    # Convert back to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if time.time() < deadline - 10.0:
        if not _verify(U_list, V_list, W_list, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U_fb = np.zeros((n * m, rank))
            V_fb = np.zeros((m * p, rank))
            W_fb = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col] = 1
                        V_fb[j * p + k, col] = 1
                        W_fb[k * n + i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
