"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
        return False
    # T[i,j,k] = 1 for all i,j,k (matmul tensor: every A[i,j]*B[j,k] contributes to C[i,k])
    # Check: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # Use einsum for speed: reshape to (n,m,p)
    # Build the full tensor product
    # T_decomposed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
      U indexed by (i,j) flattened as i*2+j
      V indexed by (j,k) flattened as j*2+k
      W indexed by (k,i) flattened as k*2+i
    T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k in {0,1}
    """
    # Strassen 7 terms for 2x2 matmul:
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M4 - M6
    # C[1,0] = M2 + M5 - M7
    # C[1,1] = M1 - M3 + M6
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # A is 2x2: A[i,j], B is 2x2: B[j,k], C is 2x2: C[i,k]
    # U[i*m+j, c] corresponds to coefficient of A[i,j] in term c
    # V[j*p+k, c] corresponds to coefficient of B[j,k] in term c
    # W[k*n+i, c] corresponds to coefficient of C[i,k] in term c
    
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Term c: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # M1 = (A00+A11)*(B00+B11)
    #   A part: A[0,0] coeff 1, A[1,1] coeff 1
    #   B part: B[0,0] coeff 1, B[1,1] coeff 1
    #   C part: C[0,0] +1, C[1,1] +1
    U[0*2+0, 0] = 1  # A[0,0]
    U[1*2+1, 0] = 1  # A[1,1]
    V[0*2+0, 0] = 1  # B[0,0]
    V[1*2+1, 0] = 1  # B[1,1]
    W[0*2+0, 0] = 1  # C[0,0]
    W[1*2+1, 0] = 1  # C[1,1]
    
    # M2 = (A10+A11)*B00
    #   A: A[1,0] coeff 1, A[1,1] coeff 1
    #   B: B[0,0] coeff 1
    #   C: C[1,0] +1
    U[1*2+0, 1] = 1  # A[1,0]
    U[1*2+1, 1] = 1  # A[1,1]
    V[0*2+0, 1] = 1  # B[0,0]
    W[0*2+1, 1] = 1  # C[1,0]
    
    # M3 = A00*(B01-B11)
    #   A: A[0,0] coeff 1
    #   B: B[0,1] coeff 1, B[1,1] coeff -1
    #   C: C[0,1] +1
    U[0*2+0, 2] = 1  # A[0,0]
    V[0*2+1, 2] = 1  # B[0,1]
    V[1*2+1, 2] = -1  # B[1,1]
    W[1*2+0, 2] = 1  # C[0,1]
    
    # M4 = A11*(B10-B00)
    #   A: A[1,1] coeff 1
    #   B: B[1,0] coeff 1, B[0,0] coeff -1
    #   C: C[0,0] -1, C[0,1] -1
    U[1*2+1, 3] = 1  # A[1,1]
    V[1*2+0, 3] = 1  # B[1,0]
    V[0*2+0, 3] = -1  # B[0,0]
    W[0*2+0, 3] = -1  # C[0,0]
    W[1*2+0, 3] = -1  # C[0,1]
    
    # M5 = (A00+A01)*B11
    #   A: A[0,0] coeff 1, A[0,1] coeff 1
    #   B: B[1,1] coeff 1
    #   C: C[0,0] +1, C[1,0] +1
    U[0*2+0, 4] = 1  # A[0,0]
    U[0*2+1, 4] = 1  # A[0,1]
    V[1*2+1, 4] = 1  # B[1,1]
    W[0*2+0, 4] = 1  # C[0,0]
    W[0*2+1, 4] = 1  # C[1,0]
    
    # M6 = (A10-A00)*(B00+B01)
    #   A: A[1,0] coeff 1, A[0,0] coeff -1
    #   B: B[0,0] coeff 1, B[0,1] coeff 1
    #   C: C[0,1] -1, C[1,1] +1
    U[1*2+0, 5] = 1  # A[1,0]
    U[0*2+0, 5] = -1  # A[0,0]
    V[0*2+0, 5] = 1  # B[0,0]
    V[0*2+1, 5] = 1  # B[0,1]
    W[1*2+0, 5] = -1  # C[0,1]
    W[1*2+1, 5] = 1  # C[1,1]
    
    # M7 = (A01-A11)*(B10+B11)
    #   A: A[0,1] coeff 1, A[1,1] coeff -1
    #   B: B[1,0] coeff 1, B[1,1] coeff 1
    #   C: C[1,0] -1, C[0,0] +1
    U[0*2+1, 6] = 1  # A[0,1]
    U[1*2+1, 6] = -1  # A[1,1]
    V[1*2+0, 6] = 1  # B[1,0]
    V[1*2+1, 6] = 1  # B[1,1]
    W[0*2+1, 6] = -1  # C[1,0]
    W[0*2+0, 6] = 1  # C[0,0]
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_tiled(n, m, p):
    """Build rank-56 decomposition by tiling 2x2 Strassen blocks.
    For 2x4 x 4x8: split into 8 blocks of 2x2 x 2x2.
    Block (ii, jj) where ii in {0,1}, jj in {0,1,2,3}:
      A rows: 2*ii .. 2*ii+1, cols: 2*jj .. 2*jj+1
      B rows: 2*jj .. 2*jj+1, cols: 2*jj .. 2*jj+1  -- wait, B is 4x8
      Actually B is m x p = 4 x 8. 
      
    Let me reconsider the tiling.
    A is n x m = 2 x 4
    B is m x p = 4 x 8
    C is n x p = 2 x 8
    
    We can think of this as: C[i,k] = sum_j A[i,j] * B[j,k]
    
    For Strassen tiling, we need to partition the indices.
    Since n=2, we can't split i into two 2-blocks (only 1 block of size 2).
    m=4 can be split into two blocks of size 2.
    p=8 can be split into four blocks of size 2.
    
    So we have 1 * 2 * 4 = 8 sub-problems, each of type (2x2) * (2x2) -> (2x2).
    Each uses rank 7, total rank = 8 * 7 = 56.
    
    Block (ii, jj, kk) where:
      ii = 0 (only one block for n=2)
      jj in {0, 1} (m=4 split into two 2-blocks)
      kk in {0, 1, 2, 3} (p=8 split into four 2-blocks)
      
    For block (ii, jj, kk):
      A indices: i in {2*ii, 2*ii+1} = {0, 1}, j in {2*jj, 2*jj+1}
      B indices: j in {2*jj, 2*jj+1}, k in {2*kk, 2*kk+1}
      C indices: i in {2*ii, 2*ii+1} = {0, 1}, k in {2*kk, 2*kk+1}
    """
    n, m, p = 2, 4, 8
    U_sub, V_sub, W_sub = _build_strassen_2x2x2()
    # U_sub: 4x7, indexed by (i_local, j_local) -> i_local*2 + j_local
    # V_sub: 4x7, indexed by (j_local, k_local) -> j_local*2 + k_local
    # W_sub: 4x7, indexed by (k_local, i_local) -> k_local*2 + i_local
    
    n_blocks_i = n // 2  # = 1
    n_blocks_j = m // 2  # = 2
    n_blocks_k = p // 2  # = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k  # = 8
    rank = total_blocks * 7  # = 56
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    for ii in range(n_blocks_i):
        for jj in range(n_blocks_j):
            for kk in range(n_blocks_k):
                # Local indices
                for li in range(2):  # local i within block
                    for lj in range(2):  # local j within block
                        global_i = 2*ii + li
                        global_j = 2*jj + lj
                        for c in range(7):
                            U[global_i * m + global_j, col_offset + c] = U_sub[li*2 + lj, c]
                
                for lj in range(2):
                    for lk in range(2):
                        global_j = 2*jj + lj
                        global_k = 2*kk + lk
                        for c in range(7):
                            V[global_j * p + global_k, col_offset + c] = V_sub[lj*2 + lk, c]
                
                for lk in range(2):
                    for li in range(2):
                        global_k = 2*kk + lk
                        global_i = 2*ii + li
                        for c in range(7):
                            W[global_k * n + global_i, col_offset + c] = W_sub[lk*2 + li, c]
                
                col_offset += 7
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the rank-56 Strassen-tiled decomposition
    U, V, W = _build_strassen_tiled(n, m, p)
    
    # Verify
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if Strassen verification fails
    if time.time() > deadline:
        U_s, V_s, W_s = _build_schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    if _verify_decomposition(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Should not reach here
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
