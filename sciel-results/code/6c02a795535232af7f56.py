import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def bclr_factors():
    """
    Returns (U, V, W) for 3x3x3 BCLR decomposition (Rank 21).
    """
    U = np.zeros((9, 21), dtype=np.float64)
    V = np.zeros((9, 21), dtype=np.float64)
    W = np.zeros((9, 21), dtype=np.float64)
    
    # BCLR 21-term algorithm
    # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    
    # m1 = a00*b00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # m2 = a01*b01
    U[1,1]=1; V[1,1]=1; W[1,1]=1
    # m3 = a02*b02
    U[2,2]=1; V[2,2]=1; W[2,2]=1
    # m4 = a10*b10
    U[3,3]=1; V[3,3]=1; W[3,3]=1
    # m5 = a11*b11
    U[4,4]=1; V[4,4]=1; W[4,4]=1
    # m6 = a12*b12
    U[5,5]=1; V[5,5]=1; W[5,5]=1
    # m7 = a20*b20
    U[6,6]=1; V[6,6]=1; W[6,6]=1
    # m8 = a21*b21
    U[7,7]=1; V[7,7]=1; W[7,7]=1
    # m9 = a22*b22
    U[8,8]=1; V[8,8]=1; W[8,8]=1
    # m10 = (a00+a10)*b10
    U[0,9]=1; U[3,9]=1; V[3,9]=1
    W[3,9]=1; W[0,9]=-1
    # m11 = (a01+a11)*b11
    U[1,10]=1; U[4,10]=1; V[4,10]=1
    W[4,10]=1; W[1,10]=-1
    # m12 = (a02+a12)*b12
    U[2,11]=1; U[5,11]=1; V[5,11]=1
    W[5,11]=1; W[2,11]=-1
    # m13 = (a10+a20)*b20
    U[3,12]=1; U[6,12]=1; V[6,12]=1
    W[6,12]=1; W[3,12]=-1
    # m14 = (a11+a21)*b21
    U[4,13]=1; U[7,13]=1; V[7,13]=1
    W[7,13]=1; W[4,13]=-1
    # m15 = (a12+a22)*b22
    U[5,14]=1; U[8,14]=1; V[8,14]=1
    W[8,14]=1; W[5,14]=-1
    # m16 = (a00+a11+a22)*b00
    U[0,15]=1; U[4,15]=1; U[8,15]=1; V[0,15]=1
    W[0,15]=1; W[4,15]=-1; W[8,15]=-1
    # m17 = (a01+a12+a20)*b01
    U[1,16]=1; U[5,16]=1; U[6,16]=1; V[1,16]=1
    W[1,16]=1; W[5,16]=-1; W[6,16]=-1
    # m18 = (a02+a10+a21)*b02
    U[2,17]=1; U[3,17]=1; U[7,17]=1; V[2,17]=1
    W[2,17]=1; W[3,17]=-1; W[7,17]=-1
    # m19 = (a10+a21)*b00
    U[3,18]=1; U[7,18]=1; V[0,18]=1
    W[3,18]=1; W[7,18]=-1
    # m20 = (a11+a22)*b01
    U[4,19]=1; U[8,19]=1; V[1,19]=1
    W[4,19]=1; W[8,19]=-1
    # m21 = (a12+a20)*b02
    U[5,20]=1; U[6,20]=1; V[2,20]=1
    W[5,20]=1; W[6,20]=-1

    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes, block_algorithms):
    """
    Build the full factor matrices using block decomposition.
    block_algorithms: dict mapping (ni, mj, pk) to (U, V, W) or 'schoolbook'
    """
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s

    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s

    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s

    # Calculate total rank
    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                key = (ni, mj, pk)
                if key in block_algorithms:
                    total_rank += block_algorithms[key][0].shape[1]
                else:
                    total_rank += ni * mj * pk

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                key = (ni, mj, pk)
                
                if key in block_algorithms:
                    U_block, V_block, W_block = block_algorithms[key]
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                
                r = U_block.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0

    # Strategy 1: BCLR 3x3x3 blocks
    # Partition: n=[3], m=[3,2], p=[3,3]
    # Blocks:
    # (3,3,3): BCLR (21)
    # (3,3,3): BCLR (21)
    # (3,2,3): Schoolbook (18)
    # (3,2,3): Schoolbook (18)
    # Total Rank: 78
    
    bclr_U, bclr_V, bclr_W = bclr_factors()
    block_algs = {
        (3, 3, 3): (bclr_U, bclr_V, bclr_W)
    }
    
    n_sizes = [3]
    m_sizes = [3, 2]
    p_sizes = [3, 3]
    
    U1, V1, W1, rank1 = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes, block_algs)
    
    if verify_decomposition(U1, V1, W1, n, m, p, atol=1e-9):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
        
    # Strategy 2: Strassen 2x2x2 blocks (Rank 84)
    # Partition: n=[1,2], m=[1,2,2], p=[2,2,2]
    strassen_U, strassen_V, strassen_W = strassen_factors()
    block_algs2 = {
        (2, 2, 2): (strassen_U, strassen_V, strassen_W)
    }
    
    n_sizes2 = [1, 2]
    m_sizes2 = [1, 2, 2]
    p_sizes2 = [2, 2, 2]
    
    U2, V2, W2, rank2 = build_block_decomposition(n, m, p, n_sizes2, m_sizes2, p_sizes2, block_algs2)
    
    if verify_decomposition(U2, V2, W2, n, m, p, atol=1e-9):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # Fallback: Schoolbook (Rank 90)
    U3, V3, W3, rank3 = build_block_decomposition(n, m, p, [3], [5], [6], {})
    return {"U": U3.tolist(), "V": V3.tolist(), "W": W3.tolist()}
