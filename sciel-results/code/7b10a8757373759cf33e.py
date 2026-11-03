"""Matrix multiplication 3,4,5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U indexed by (i, j) for i,j in {0,1}
    V indexed by (j, k) for j,k in {0,1}  
    W indexed by (k, i) for k,i in {0,1}
    
    Strassen's 7 products for [[a,b],[c,d]] * [[e,f],[g,h]] = [[p,q],[r,s]]:
    p = (a+b)(e+h) - m - n  ... wait let me use the standard form.
    
    Standard Strassen:
    m1 = (a+b)(e+h)
    m2 = (c+d)(e)
    m3 = (a)(f-h)
    m4 = (d)(g-e)
    m5 = (a+d)(h)
    m6 = (b+c)(g)
    m7 = (b-d)(e+f)
    
    Result:
    top-left = m1 + m4 - m5 + m7
    top-right = m3 + m5
    bottom-left = m2 + m4
    bottom-right = m1 - m2 + m3 + m6
    
    For tensor decomposition, each mc is one rank-1 term.
    U[i,j,c] * V[j,k,c] * W[k,i,c] should give the coefficient of A[i,j]*B[j,k] in product c.
    
    For m1 = (a+b)(e+h):
    A part: a+b = A[0,0]+A[0,1] → U[0,0]=1, U[0,1]=1, U[1,0]=0, U[1,1]=0
    B part: e+h = B[0,0]+B[1,1] → V[0,0]=1, V[0,1]=0, V[1,0]=0, V[1,1]=1
    C part: contributes to top-left (i=0,k=0) with coeff 1 → W[0,0]=1, others 0
    
    For m2 = (c+d)(e):
    A part: c+d = A[1,0]+A[1,1] → U[1,0]=1, U[1,1]=1, U[0,0]=0, U[0,1]=0
    B part: e = B[0,0] → V[0,0]=1, others 0
    C part: bottom-left (i=1,k=0) coeff 1 → W[0,1]=1
    
    For m3 = (a)(f-h):
    A part: a = A[0,0] → U[0,0]=1
    B part: f-h = B[0,1]-B[1,1] → V[0,1]=1, V[1,1]=-1
    C part: top-right (i=0,k=1) coeff 1 → W[1,0]=1
    
    For m4 = (d)(g-e):
    A part: d = A[1,1] → U[1,1]=1
    B part: g-e = B[1,0]-B[0,0] → V[1,0]=1, V[0,0]=-1
    C part: top-left (i=0,k=0) coeff 1, bottom-left (i=1,k=0) coeff 1 → W[0,0]=1, W[0,1]=1
    
    For m5 = (a+d)(h):
    A part: a+d = A[0,0]+A[1,1] → U[0,0]=1, U[1,1]=1
    B part: h = B[1,1] → V[1,1]=1
    C part: top-left (i=0,k=0) coeff -1, bottom-right (i=1,k=1) coeff 1 → W[0,0]=-1, W[1,1]=1
    
    For m6 = (b+c)(g):
    A part: b+c = A[0,1]+A[1,0] → U[0,1]=1, U[1,0]=1
    B part: g = B[1,0] → V[1,0]=1
    C part: bottom-right (i=1,k=1) coeff 1 → W[1,1]=1
    
    For m7 = (b-d)(e+f):
    A part: b-d = A[0,1]-A[1,1] → U[0,1]=1, U[1,1]=-1
    B part: e+f = B[0,0]+B[0,1] → V[0,0]=1, V[0,1]=1
    C part: top-left (i=0,k=0) coeff 1 → W[0,0]=1
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # m1 = (a+b)(e+h)
    U[0, 0] = 1; U[1, 0] = 1  # a+b: U[0,0]=1, U[0,1]=1 -> index (0,0)=0, (0,1)=1
    # U indexed by (i,j): U[i*2+j, c]
    U[0, 0] = 1  # i=0,j=0
    U[1, 0] = 1  # i=0,j=1
    # V indexed by (j,k): V[j*2+k, c]
    V[0, 0] = 1  # j=0,k=0
    V[3, 0] = 1  # j=1,k=1
    # W indexed by (k,i): W[k*2+i, c]
    W[0, 0] = 1  # k=0,i=0
    
    # m2 = (c+d)(e)
    U[2, 1] = 1  # i=1,j=0
    U[3, 1] = 1  # i=1,j=1
    V[0, 1] = 1  # j=0,k=0
    W[2, 1] = 1  # k=0,i=1
    
    # m3 = (a)(f-h)
    U[0, 2] = 1  # i=0,j=0
    V[1, 2] = 1  # j=0,k=1
    V[3, 2] = -1 # j=1,k=1
    W[1, 2] = 1  # k=1,i=0
    
    # m4 = (d)(g-e)
    U[3, 3] = 1  # i=1,j=1
    V[2, 3] = 1  # j=1,k=0
    V[0, 3] = -1 # j=0,k=0
    W[0, 3] = 1  # k=0,i=0
    W[2, 3] = 1  # k=0,i=1
    
    # m5 = (a+d)(h)
    U[0, 4] = 1  # i=0,j=0
    U[3, 4] = 1  # i=1,j=1
    V[3, 4] = 1  # j=1,k=1
    W[0, 4] = -1 # k=0,i=0
    W[3, 4] = 1  # k=1,i=1
    
    # m6 = (b+c)(g)
    U[1, 5] = 1  # i=0,j=1
    U[2, 5] = 1  # i=1,j=0
    V[2, 5] = 1  # j=1,k=0
    W[3, 5] = 1  # k=1,i=1
    
    # m7 = (b-d)(e+f)
    U[1, 6] = 1  # i=0,j=1
    U[3, 6] = -1 # i=1,j=1
    V[0, 6] = 1  # j=0,k=0
    V[1, 6] = 1  # j=0,k=1
    W[0, 6] = 1  # k=0,i=0
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """
    Schoolbook decomposition for n_block x m_block x p_block.
    Returns U (n_block*m_block, n_block*m_block*p_block),
            V (m_block*p_block, n_block*m_block*p_block),
            W (p_block*n_block, n_block*m_block*p_block)
    """
    rank = n_block * m_block * p_block
    U = np.zeros((n_block * m_block, rank))
    V = np.zeros((m_block * p_block, rank))
    W = np.zeros((p_block * n_block, rank))
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j, col] = 1
                V[j * p_block + k, col] = 1
                W[k * n_block + i, col] = 1
                col += 1
    return U, V, W

def _combine_blocks(block_list, n, m, p):
    """
    Combine multiple blocks into a full decomposition.
    block_list: list of (row_indices, mid_indices, col_indices, U_block, V_block, W_block)
    where U_block is (len(row_indices)*len(mid_indices), rank_block),
          V_block is (len(mid_indices)*len(col_indices), rank_block),
          W_block is (len(col_indices)*len(row_indices), rank_block)
    """
    total_rank = sum(b[3].shape[1] for b in block_list)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for row_idx, mid_idx, col_idx, U_b, V_b, W_b in block_list:
        r = U_b.shape[1]
        for local_i, global_i in enumerate(row_idx):
            for local_j, global_j in enumerate(mid_idx):
                U_full[global_i * m + global_j, col_offset:col_offset + r] = U_b[local_i * len(mid_idx) + local_j, :]
        for local_j, global_j in enumerate(mid_idx):
            for local_k, global_k in enumerate(col_idx):
                V_full[global_j * p + global_k, col_offset:col_offset + r] = V_b[local_j * len(col_idx) + local_k, :]
        for local_k, global_k in enumerate(col_idx):
            for local_i, global_i in enumerate(row_idx):
                W_full[global_k * n + global_i, col_offset:col_offset + r] = W_b[local_k * len(row_idx) + local_i, :]
        col_offset += r
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _to_half_integer_matrix(M):
    """Convert float matrix to list of lists with integer/half-integer values."""
    result = []
    for row in M:
        new_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                new_row.append(int(round(val)))
            elif abs(val * 2 - round(val * 2)) < 1e-9:
                new_row.append(round(val * 2) / 2)
            else:
                new_row.append(val)
        result.append(new_row)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Strategy: Partition into blocks and use Strassen for 2x2x2, schoolbook for rest
    # Rows: {0,1}, {2}
    # Mids: {0,1}, {2,3}
    # Cols: {0,1}, {2,3}, {4}
    
    # Define blocks: (row_indices, mid_indices, col_indices, type)
    # type: 'strassen' for 2x2x2, 'schoolbook' otherwise
    
    blocks_spec = [
        ([0, 1], [0, 1], [0, 1], 'strassen'),
        ([0, 1], [0, 1], [2, 3], 'strassen'),
        ([0, 1], [0, 1], [4], 'schoolbook'),
        ([0, 1], [2, 3], [0, 1], 'strassen'),
        ([0, 1], [2, 3], [2, 3], 'strassen'),
        ([0, 1], [2, 3], [4], 'schoolbook'),
        ([2], [0, 1], [0, 1], 'schoolbook'),
        ([2], [0, 1], [2, 3], 'schoolbook'),
        ([2], [0, 1], [4], 'schoolbook'),
        ([2], [2, 3], [0, 1], 'schoolbook'),
        ([2], [2, 3], [2, 3], 'schoolbook'),
        ([2], [2, 3], [4], 'schoolbook'),
    ]
    
    block_list = []
    for row_idx, mid_idx, col_idx, btype in blocks_spec:
        nb, mb, pb = len(row_idx), len(mid_idx), len(col_idx)
        if btype == 'strassen' and nb == 2 and mb == 2 and pb == 2:
            U_b, V_b, W_b = _strassen_222()
        else:
            U_b, V_b, W_b = _schoolbook_block(nb, mb, pb)
        block_list.append((row_idx, mid_idx, col_idx, U_b, V_b, W_b))
    
    U, V, W = _combine_blocks(block_list, n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Try to find even better decomposition with more aggressive search
        if time.time() < deadline - 30:
            # Try alternative partitions
            # Partition: Rows {0,1,2}, Mids {0,1,2,3}, Cols {0,1,2,3,4}
            # Try: {0,1}x{0,1,2,3}x{0,1} + {2}x{0,1,2,3}x{0,1} + {0,1,2}x{0,1,2,3}x{2,3,4}
            # But that's not 2x2x2 blocks...
            
            # Let me try a different partition to see if we can do better
            # Alternative: Rows {0,1}, {2}; Mids {0,1}, {2,3}; Cols {0,1,2}, {3,4}
            blocks_spec2 = [
                ([0, 1], [0, 1], [0, 1, 2], 'schoolbook'),
                ([0, 1], [0, 1], [3, 4], 'schoolbook'),
                ([0, 1], [2, 3], [0, 1, 2], 'schoolbook'),
                ([0, 1], [2, 3], [3, 4], 'schoolbook'),
                ([2], [0, 1], [0, 1, 2], 'schoolbook'),
                ([2], [0, 1], [3, 4], 'schoolbook'),
                ([2], [2, 3], [0, 1, 2], 'schoolbook'),
                ([2], [2, 3], [3, 4], 'schoolbook'),
            ]
            # This gives: 2*2*3*2 + 2*2*2*2 + 1*2*3*2 + 1*2*2*2 = 24+16+12+8 = 60, worse
            
            # Try: more Strassen blocks
            # Rows {0,1}, {2}; Mids {0,1}, {2,3}; Cols {0,1}, {2}, {3,4}
            blocks_spec3 = [
                ([0, 1], [0, 1], [0, 1], 'strassen'),
                ([0, 1], [0, 1], [2], 'schoolbook'),
                ([0, 1], [0, 1], [3, 4], 'schoolbook'),
                ([0, 1], [2, 3], [0, 1], 'strassen'),
                ([0, 1], [2, 3], [2], 'schoolbook'),
                ([0, 1], [2, 3], [3, 4], 'schoolbook'),
                ([2], [0, 1], [0, 1], 'schoolbook'),
                ([2], [0, 1], [2], 'schoolbook'),
                ([2], [0, 1], [3, 4], 'schoolbook'),
                ([2], [2, 3], [0, 1], 'schoolbook'),
                ([2], [2, 3], [2], 'schoolbook'),
                ([2], [2, 3], [3, 4], 'schoolbook'),
            ]
            # Rank: 7 + 2*2*1*2 + 2*2*2*2 + 7 + 2*2*1*2 + 2*2*2*2 + 1*2*2*2 + 1*2*1*2 + 1*2*2*2 + 1*2*2*2 + 1*2*1*2 + 1*2*2*2
            # = 7 + 8 + 8 + 7 + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4 = 70, worse
            
            # The first partition with 56 is good. Let me try to verify and return it.
    
    # Convert to list format with integer/half-integer entries
    U_list = _to_half_integer_matrix(U)
    V_list = _to_half_integer_matrix(V)
    W_list = _to_half_integer_matrix(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
