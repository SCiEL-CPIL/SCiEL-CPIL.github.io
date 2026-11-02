"""
Strassen-based block decomposition for 3x5x7 matrix multiplication.

We split the dimensions into blocks of size 2 and 1:
  n=3 -> [2, 1]
  m=5 -> [2, 2, 1]
  p=7 -> [2, 2, 2, 1]

For each combination of block sizes (ni, mj, pk), we use:
  - Strassen's 2x2x2 algorithm (rank 7) when ni=mj=pk=2
  - Schoolbook (rank ni*mj*pk) otherwise

The total rank is the sum over all block combinations.
"""
import numpy as np

def strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = sum_l A[i,l] * B[l,k]
    can be decomposed as:
    T = sum_{r=0}^{6} U_r * V_r * W_r
    
    where U_r is 4x1 (flattened 2x2), V_r is 4x1, W_r is 4x1
    
    Strassen's 7 products:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We need to express this as U * V * W where:
    - U[i*2+j, r] gives the coefficient for A[i,j] in product r
    - V[j*2+k, r] gives the coefficient for B[j,k] in product r  
    - W[k*2+i, r] gives the coefficient for C[i,k] in product r
    
    Note: The trace convention is W[k*n+i, r], so for 2x2, W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Indexing: A[i,j] -> index i*2+j
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[j,k] -> index j*2+k
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[i,k] -> index k*2+i (trace convention: W[k*n+i])
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = -1  # C[1,1]
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1  # A[0,0]
    V[2, 2] = 1  # B[1,0]
    V[3, 2] = -1  # B[1,1]
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1  # A[1,1]
    V[1, 3] = 1  # B[0,1]
    V[0, 3] = -1  # B[0,0]
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1  # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1  # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1  # A[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def schoolbook_decomp(ni, mj, pk):
    """
    Return the schoolbook decomposition for ni x mj x pk matrix multiplication.
    """
    rank = ni * mj * pk
    U = np.zeros((ni * mj, rank), dtype=int)
    V = np.zeros((mj * pk, rank), dtype=int)
    W = np.zeros((pk * ni, rank), dtype=int)
    
    idx = 0
    for i in range(ni):
        for j in range(mj):
            for k in range(pk):
                U[i * mj + j, idx] = 1
                V[j * pk + k, idx] = 1
                W[k * ni + i, idx] = 1
                idx += 1
    
    return U, V, W

def construct():
    n, m, p = 3, 5, 7
    
    # Split dimensions into blocks
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2, 1]
    
    # Compute offsets for each block
    n_offsets = []
    offset = 0
    for b in n_blocks:
        n_offsets.append(offset)
        offset += b
    
    m_offsets = []
    offset = 0
    for b in m_blocks:
        m_offsets.append(offset)
        offset += b
    
    p_offsets = []
    offset = 0
    for b in p_blocks:
        p_offsets.append(offset)
        offset += b
    
    # Get Strassen decomposition for 2x2x2
    strassen_U, strassen_V, strassen_W = strassen_2x2x2()
    
    # Build the full decomposition
    all_U = []
    all_V = []
    all_W = []
    
    for ni_idx, ni in enumerate(n_blocks):
        for mj_idx, mj in enumerate(m_blocks):
            for pk_idx, pk in enumerate(p_blocks):
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    U_block, V_block, W_block = strassen_U.copy(), strassen_V.copy(), strassen_W.copy()
                else:
                    # Use schoolbook
                    U_block, V_block, W_block = schoolbook_decomp(ni, mj, pk)
                
                # Pad U to full n*m size
                U_full = np.zeros((n * m, U_block.shape[1]), dtype=int)
                n_off = n_offsets[ni_idx]
                m_off = m_offsets[mj_idx]
                for i in range(ni):
                    for j in range(mj):
                        U_full[(n_off + i) * m + (m_off + j), :] = U_block[i * mj + j, :]
                
                # Pad V to full m*p size
                V_full = np.zeros((m * p, V_block.shape[1]), dtype=int)
                m_off = m_offsets[mj_idx]
                p_off = p_offsets[pk_idx]
                for j in range(mj):
                    for k in range(pk):
                        V_full[(m_off + j) * p + (p_off + k), :] = V_block[j * pk + k, :]
                
                # Pad W to full p*n size
                W_full = np.zeros((p * n, W_block.shape[1]), dtype=int)
                p_off = p_offsets[pk_idx]
                n_off = n_offsets[ni_idx]
                for k in range(pk):
                    for i in range(ni):
                        W_full[(p_off + k) * n + (n_off + i), :] = W_block[k * ni + i, :]
                
                all_U.append(U_full)
                all_V.append(V_full)
                all_W.append(W_full)
    
    # Concatenate all blocks
    U = np.hstack(all_U)
    V = np.hstack(all_V)
    W = np.hstack(all_W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
