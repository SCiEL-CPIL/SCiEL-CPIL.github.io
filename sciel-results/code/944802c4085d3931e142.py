import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 145.0  # Safety margin before 150s kill

def _strassen_2x2x2_embed(n, m, p, i0, j0, k0):
    """
    Returns U, V, W for a 2x2x2 block at offset (i0, j0, k0) in full n x m x p space.
    Uses Strassen's algorithm (rank 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    # M1 = (A00+A11)(B00+B11)
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    # M2 = (A10+A11)B00
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    # M3 = A00(B01-B11)
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    # M4 = A11(B10-B00)
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    # M5 = (A00+A01)B11
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _schoolbook_embed(n, m, p, i0, j0, k0, bi, bj, bk):
    """
    Returns U, V, W for a schoolbook bi x bj x bk block at offset (i0, j0, k0).
    """
    rank = bi * bj * bk
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(bi):
        for j in range(bj):
            for k in range(bk):
                U[(i0 + i) * m + (j0 + j), col] = 1
                V[(j0 + j) * p + (k0 + k), col] = 1
                W[(k0 + k) * n + (i0 + i), col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using vectorized numpy for speed."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We need to check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Let's reshape to 3D tensors for efficient checking
    # U3[i,j,c], V3[j,k,c], W3[k,i,c]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a bit tricky to vectorize directly without huge intermediate arrays.
    # n=4, m=5, p=5, rank~92. 4*5*5*92 = 9200 ops per element? No, it's a dot product.
    # Let's just use the loop version but with numpy dot products if possible, or keep simple loops.
    # Given small dimensions, simple loops are fine and safe.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k]) * 0 # placeholder
                # Actually, the term is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is not a simple dot product of two vectors because W depends on k,i.
                # It's a triple sum.
                # sum_c (U[i,j,c] * W[k,i,c]) * V[j,k,c]
                # Let A_c = U[i,j,c] * W[k,i,c]
                # Then sum_c A_c * V[j,k,c] = dot(A, V[j,k,:])
                
                # Optimization: Precompute U[i*m+j, :] * W[k*n+i, :] element-wise
                # Then dot with V[j*p+k, :]
                
                prod = U[i * m + j] * W[k * n + i]
                s = np.dot(prod, V[j * p + k])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strategy: Partition 4x5x5 into 2x2x2 blocks where possible, use Strassen for 2x2x2.
    # Dimensions: i (rows of A, cols of C) = 4, j (cols of A, rows of B) = 5, k (cols of B, cols of C) = 5.
    #
    # Partition i into {0,1}, {2,3} (size 2 each)
    # Partition j into {0,1}, {2,3}, {4} (sizes 2, 2, 1)
    # Partition k into {0,1}, {2,3}, {4} (sizes 2, 2, 1)
    #
    # Total blocks: 2 * 3 * 3 = 18 blocks.
    #
    # Block types:
    # 1. 2x2x2 blocks (i-size 2, j-size 2, k-size 2): Use Strassen (rank 7)
    #    Count: 2 (i) * 2 (j in {0,1},{2,3}) * 2 (k in {0,1},{2,3}) = 8 blocks.
    #    Rank: 8 * 7 = 56.
    #
    # 2. 2x2x1 blocks (i-size 2, j-size 2, k-size 1): Use Schoolbook (rank 4)
    #    Count: 2 (i) * 2 (j in {0,1},{2,3}) * 1 (k={4}) = 4 blocks.
    #    Rank: 4 * 4 = 16.
    #
    # 3. 2x1x2 blocks (i-size 2, j-size 1, k-size 2): Use Schoolbook (rank 4)
    #    Count: 2 (i) * 1 (j={4}) * 2 (k in {0,1},{2,3}) = 4 blocks.
    #    Rank: 4 * 4 = 16.
    #
    # 4. 2x1x1 blocks (i-size 2, j-size 1, k-size 1): Use Schoolbook (rank 2)
    #    Count: 2 (i) * 1 (j={4}) * 1 (k={4}) = 2 blocks.
    #    Rank: 2 * 2 = 4.
    #
    # Total Rank = 56 + 16 + 16 + 4 = 92.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Define block boundaries
    i_splits = [0, 2, 4]
    j_splits = [0, 2, 4, 5]
    k_splits = [0, 2, 4, 5]
    
    for i_idx in range(len(i_splits) - 1):
        i0 = i_splits[i_idx]
        i1 = i_splits[i_idx + 1]
        bi = i1 - i0
        
        for j_idx in range(len(j_splits) - 1):
            j0 = j_splits[j_idx]
            j1 = j_splits[j_idx + 1]
            bj = j1 - j0
            
            for k_idx in range(len(k_splits) - 1):
                k0 = k_splits[k_idx]
                k1 = k_splits[k_idx + 1]
                bk = k1 - k0
                
                if bi == 2 and bj == 2 and bk == 2:
                    U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, i0, j0, k0)
                else:
                    U_s, V_s, W_s = _schoolbook_embed(n, m, p, i0, j0, k0, bi, bj, bk)
                
                U_parts.append(U_s)
                V_parts.append(V_s)
                W_parts.append(W_s)
                
                if time.time() > DEADLINE:
                    # Fallback to schoolbook if we run out of time (unlikely for this small size)
                    U = np.zeros((n * m, n * m * p))
                    V = np.zeros((m * p, n * m * p))
                    W = np.zeros((p * n, n * m * p))
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    rank = U_combined.shape[1]
    
    # Verify
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Fallback: pure schoolbook
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
