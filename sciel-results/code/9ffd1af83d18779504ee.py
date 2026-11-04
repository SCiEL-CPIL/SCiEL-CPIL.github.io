"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of n x m by m x p.
    U has shape (n*m, n*m*p), V has shape (m*p, n*m*p), W has shape (p*n, n*m*p).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where c = (i,j,k)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W

def _strassen_2x2x2():
    """Return (U, V, W) for Strassen decomposition of 2x2 by 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    
    Strassen for C = A*B where A, B, C are 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # We need: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k
    # (since the matrix mult tensor T[i,j,k] = 1 for all valid indices)
    # 
    # Actually, let me re-derive. The decomposition satisfies:
    # (A*B)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    #
    # For this to equal sum_j A[i,j]*B[j,k], we need the tensor identity:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # (because then sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    #  = sum_i,j,k W[k*n+i,c]... hmm, let me think more carefully.
    #
    # Actually the correct identity is:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # We want: (A*B)[i,k] = sum_c W[k*n+i, c] * P_c * Q_c
    # where P_c = sum_j U[i*m+j, c] * A[i,j]  -- wait, this doesn't work because U depends on i and j.
    #
    # Let me reconsider. The standard tensor decomposition of matrix multiplication:
    # The multiplication tensor T has entries T_{i,j,k} = 1 (for all i in [n], j in [m], k in [p]).
    # A rank-r decomposition gives T = sum_{c=1}^{r} U_c ⊗ V_c ⊗ W_c
    # where (U_c)_{i,j} = U[i*m+j, c], (V_c)_{j,k} = V[j*p+k, c], (W_c)_{k,i} = W[k*n+i, c]
    # So: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k.
    #
    # Then: sum_j A[i,j] B[j,k] = sum_{i,j,k} T_{i,j,k} A[i,j] B[j,k]
    # = sum_{i,j,k} sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
    # = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k])
    # Hmm, this isn't quite right either because the sums over j are coupled.
    #
    # Let me just use the direct verification: the decomposition is valid if
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    
    # For Strassen 2x2x2, the 7 bilinear forms are:
    # Let me define the U, V, W entries directly.
    #
    # The 7 terms M1..M7 correspond to the 7 rank-1 factors.
    # M1 = (A00+A11)(B00+B11): involves A[0,0],A[1,1] and B[0,0],B[1,1]
    #   U contribution: U[0,0]=1, U[3,0]=1 (for A00 and A11)
    #   V contribution: V[0,0]=1, V[3,0]=1 (for B00 and B11)
    #   W contribution: contributes to C00 and C11 with coefficient +1 and -1... 
    #   Wait, W[0,0]=1 (for C00, k=0,i=0) and W[3,0]=-1? No...
    #
    # Let me be very explicit. For the 2x2x2 case:
    # i,j,k each in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    #
    # We need sum_c U[i*2+j,c] V[j*2+k,c] W[k*2+i,c] = 1 for all i,j,k in {0,1}.
    #
    # Strassen gives us 7 rank-1 terms. Let me write them out:
    #
    # Term 1 (M1): (A00+A11)*(B00+B11)
    #   U: U[0,1]=1 (i=0,j=0), U[3,1]=1 (i=1,j=1)
    #   V: V[0,1]=1 (j=0,k=0), V[3,1]=1 (j=1,k=1)
    #   W: contributes to C00 (k=0,i=0) with +1, C11 (k=1,i=1) with +1
    #   But C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    #   So W[0,1]=1 (C00), W[3,1]=1 (C11)
    #
    # Term 2 (M2): (A10+A11)*B00
    #   U: U[2,2]=1 (i=1,j=0), U[3,2]=1 (i=1,j=1)
    #   V: V[0,2]=1 (j=0,k=0)
    #   W: C10 (k=0,i=1) gets +1, so W[2,2]=1
    #   Wait, C10 = M2 + M4. So W[2,2]=1 for C10.
    #
    # Term 3 (M3): A00*(B01-B11)
    #   U: U[0,3]=1 (i=0,j=0)
    #   V: V[1,3]=1 (j=0,k=1), V[3,3]=-1 (j=1,k=1)
    #   W: C01 (k=1,i=0) gets +1, C11 (k=1,i=1) gets +1
    #   W[1,3]=1 (C01), W[3,3]=1 (C11)
    #
    # Term 4 (M4): A11*(B10-B00)
    #   U: U[3,4]=1 (i=1,j=1)
    #   V: V[2,4]=1 (j=1,k=0), V[0,4]=-1 (j=0,k=0)
    #   W: C10 (k=0,i=1) gets +1, C00 (k=0,i=0) gets +1
    #   W[2,4]=1 (C10), W[0,4]=1 (C00)
    #
    # Term 5 (M5): (A00+A01)*B11
    #   U: U[0,5]=1 (i=0,j=0), U[1,5]=1 (i=0,j=1)
    #   V: V[3,5]=1 (j=1,k=1)
    #   W: C00 (k=0,i=0) gets -1, so W[0,5]=-1
    #
    # Term 6 (M6): (A10-A00)*(B00+B01)
    #   U: U[2,6]=1 (i=1,j=0), U[0,6]=-1 (i=0,j=0)
    #   V: V[0,6]=1 (j=0,k=0), V[1,6]=1 (j=0,k=1)
    #   W: C11 (k=1,i=1) gets +1, so W[3,6]=1
    #
    # Term 7 (M7): (A01-A11)*(B10+B11)
    #   U: U[1,7]=1 (i=0,j=1), U[3,7]=-1 (i=1,j=1)
    #   V: V[2,7]=1 (j=1,k=0), V[3,7]=1 (j=1,k=1)
    #   W: C00 (k=0,i=0) gets +1, so W[0,7]=1
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _combine_blocks(blocks):
    """Combine multiple block decompositions into one global decomposition.
    
    Each block is a dict with:
      'n', 'm', 'p': block dimensions
      'U', 'V', 'W': local factor matrices
      'row_offset_U': offset for U rows (i index)
      'col_offset_U': offset for U cols (j index)  
      'row_offset_V': offset for V rows (j index)
      'col_offset_V': offset for V cols (k index)
      'row_offset_W': offset for W rows (k index)
      'col_offset_W': offset for W cols (i index)
    
    Returns (U_global, V_global, W_global) as numpy arrays.
    """
    total_rank = sum(b['U'].shape[1] for b in blocks)
    n_total = max(b['n'] for b in blocks)
    m_total = max(b['m'] for b in blocks)
    p_total = max(b['p'] for b in blocks)
    
    # Actually, we need to figure out the global dimensions.
    # For 3x5x5, n=3, m=5, p=5.
    # The blocks partition the indices.
    # Let's just build it from the specific structure.
    return None  # Not used in this approach

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Strategy: Split the 3x5x5 problem into blocks where we can apply Strassen
    # for 2x2x2 subproblems.
    #
    # Split n=3 into {0, 1, 2}
    # Split m=5 into {0, 1, 2, 3, 4}  
    # Split p=5 into {0, 1, 2, 3, 4}
    #
    # We can group indices to form blocks. A natural approach:
    # For the n=3 dimension, we can't easily use Strassen (needs even or specific sizes).
    # 
    # Alternative approach: Use a block decomposition where we split into:
    # Block A: n={0,1}, m={0,1}, p={0,1} -> 2x2x2, use Strassen (rank 7)
    # Block B: n={0,1}, m={2,3,4}, p={0,1} -> 2x3x2
    # Block C: n={0,1}, m={0,1}, p={2,3,4} -> 2x2x3
    # Block D: n={0,1}, m={2,3,4}, p={2,3,4} -> 2x3x3
    # Block E: n={2}, m={0,1}, p={0,1} -> 1x2x2
    # Block F: n={2}, m={2,3,4}, p={0,1} -> 1x3x2
    # Block G: n={2}, m={0,1}, p={2,3,4} -> 1x2x3
    # Block H: n={2}, m={2,3,4}, p={2,3,4} -> 1x3x3
    #
    # For non-2x2x2 blocks, we can further decompose or use schoolbook.
    # 2x3x2: split m=3 into {0,1} and {2}. 
    #   2x2x2 (Strassen, rank 7) + 2x1x2 (schoolbook, rank 4) = 11
    # 2x2x3: split p=3 into {0,1} and {2}.
    #   2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = 11
    # 2x3x3: split m=3 into {0,1},{2} and p=3 into {0,1},{2}.
    #   2x2x2 (Strassen, rank 7) + 2x1x2 (schoolbook, rank 4) + 2x2x1 (schoolbook, rank 4) + 2x1x1 (schoolbook, rank 2) = 17
    #   Alternatively: 2x3x3 -> split into 2x2x3 + 2x1x3
    #   2x2x3 -> 2x2x2 + 2x2x1 = 7 + 4 = 11
    #   2x1x3 -> schoolbook = 6
    #   Total: 17
    # 1x2x2: schoolbook = 4
    # 1x3x2: schoolbook = 6
    # 1x2x3: schoolbook = 6
    # 1x3x3: schoolbook = 9
    #
    # Total: 7 + 11 + 11 + 17 + 4 + 6 + 6 + 9 = 71
    #
    # Let me try a different split to get better:
    # Actually, for 1xN x M or N x 1 x M cases, schoolbook is already minimal.
    #
    # Let me reconsider. The known optimal for 3x5x5 might be around 69-72.
    # Let me just implement the block approach carefully.
    
    # I'll build the decomposition by directly constructing U, V, W for each block
    # and placing them at the correct global positions.
    
    # Global dimensions: n=3, m=5, p=5
    # U: (3*5, rank) = (15, rank)
    # V: (5*5, rank) = (25, rank)
    # W: (5*3, rank) = (15, rank)
    
    # Define blocks as (n_start, n_end, m_start, m_end, p_start, p_end)
    # where n_end, m_end, p_end are exclusive.
    
    def get_block_decomp(n_b, m_b, p_b, n_off, m_off, p_off):
        """Get decomposition for an n_b x m_b by m_b x p_b block,
        placed at offsets (n_off, m_off, p_off) in the global 3x5x5."""
        # Returns (U_local, V_local, W_local, n_b, m_b, p_b, n_off, m_off, p_off)
        # where U_local has shape (n_b*m_b, rank_b), etc.
        
        if n_b == 2 and m_b == 2 and p_b == 2:
            U_l, V_l, W_l = _strassen_2x2x2()
            return U_l, V_l, W_l
        else:
            # Try to further decompose
            if n_b == 2 and m_b == 3 and p_b == 2:
                # Split m: {0,1} and {2}
                # 2x2x2 (Strassen) + 2x1x2 (schoolbook)
                U1, V1, W1 = _strassen_2x2x2()  # 2x2x2, rank 7
                U2, V2, W2 = _schoolbook(2, 1, 2)  # 2x1x2, rank 4
                # Combine: U has 4 rows total (n_b*m_b = 6)
                # U1 corresponds to rows 0,1,2,3 (i=0,1; j=0,1)
                # U2 corresponds to rows 4,5 (i=0,1; j=2)
                rank_total = U1.shape[1] + U2.shape[1]
                U_c = np.zeros((n_b * m_b, rank_total))
                V_c = np.zeros((m_b * p_b, rank_total))
                W_c = np.zeros((p_b * n_b, rank_total))
                # U1: shape (4, 7) -> rows 0-3
                U_c[0:4, 0:7] = U1
                # U2: shape (2, 4) -> rows 4-5
                U_c[4:6, 7:11] = U2
                # V1: shape (4, 7) -> rows 0-3 (j=0,1; k=0,1)
                V_c[0:4, 0:7] = V1
                # V2: shape (2, 4) -> rows 4-5 (j=2; k=0,1)
                V_c[4:6, 7:11] = V2
                # W1: shape (4, 7) -> rows 0-3 (k=0,1; i=0,1)
                W_c[0:4, 0:7] = W1
                # W2: shape (2, 4) -> rows 0-3 (k=0,1; i=0,1)
                # Wait, W2 has shape (p_b*n_b, rank) = (2*2, 4) = (4, 4)
                # But we need to map it to the right rows.
                # For the 2x1x2 subblock (j=2), the W indices are (k, i) with k in {0,1}, i in {0,1}
                # So W2 rows correspond to k*n_b+i = k*2+i, which are rows 0-3 of the full W.
                # But the full W for 2x3x2 has shape (p_b*n_b, rank) = (2*2, rank) = (4, rank).
                # So W2 maps to rows 0-3.
                W_c[0:4, 7:11] = W2
                return U_c, V_c, W_c
            
            elif n_b == 2 and m_b == 2 and p_b == 3:
                # Split p: {0,1} and {2}
                # 2x2x2 (Strassen) + 2x2x1 (schoolbook)
                U1, V1, W1 = _strassen_2x2x2()  # 2x2x2, rank 7
                U2, V2, W2 = _schoolbook(2, 2, 1)  # 2x2x1, rank 4
                rank_total = U1.shape[1] + U2.shape[1]
                U_c = np.zeros((n_b * m_b, rank_total))
                V_c = np.zeros((m_b * p_b, rank_total))
                W_c = np.zeros((p_b * n_b, rank_total))
                # U: 4 rows
                U_c[0:4, 0:7] = U1
                U_c[0:4, 7:11] = U2
                # V1: 4 rows (j=0,1; k=0,1)
                V_c[0:4, 0:7] = V1
                # V2: 4 rows (j=0,1; k=2)
                V_c[4:8, 7:11] = V2
                # W1: 4 rows (k=0,1; i=0,1)
                W_c[0:4, 0:7] = W1
                # W2: 4 rows (k=2; i=0,1)
                # W2 has shape (1*2, 4) = (2, 4)
                W_c[4:6, 7:11] = W2
                return U_c, V_c, W_c
            
            elif n_b == 2 and m_b == 3 and p_b == 3:
                # Split into 2x2x2 + 2x1x2 + 2x2x1 + 2x1x1
                U1, V1, W1 = _strassen_2x2x2()  # rank 7
                U2, V2, W2 = _schoolbook(2, 1, 2)  # rank 4
                U3, V3, W3 = _schoolbook(2, 2, 1)  # rank 4
                U4, V4, W4 = _schoolbook(2, 1, 1)  # rank 2
                rank_total = 7 + 4 + 4 + 2
                U_c = np.zeros((n_b * m_b, rank_total))
                V_c = np.zeros((m_b * p_b, rank_total))
                W_c = np.zeros((p_b * n_b, rank_total))
                # U: 6 rows
                U_c[0:4, 0:7] = U1   # j=0,1
                U_c[4:6, 7:11] = U2  # j=2, k=0,1
                U_c[0:4, 11:15] = U3 # j=0,1, k=2
                U_c[4:6, 15:17] = U4 # j=2, k=2
                # V: 9 rows
                V_c[0:4, 0:7] = V1   # j=0,1; k=0,1
                V_c[4:6, 7:11] = V2  # j=2; k=0,1
                V_c[0:4, 11:15] = V3 # j=0,1; k=2
                V_c[4:6, 15:17] = V4 # j=2; k=2
                # W: 6 rows
                W_c[0:4, 0:7] = W1   # k=0,1; i=0,1
                W_c[0:4, 7:11] = W2  # k=0,1; i=0,1
                W_c[4:6, 11:15] = W3 # k=2; i=0,1
                W_c[4:6, 15:17] = W4 # k=2; i=0,1
                return U_c, V_c, W_c
            
            else:
                # Schoolbook
                return _schoolbook(n_b, m_b, p_b)
    
    # Define the 8 blocks
    # Block A: n={0,1}, m={0,1}, p={0,1} -> 2x2x2
    # Block B: n={0,1}, m={2,3,4}, p={0,1} -> 2x3x2
    # Block C: n={0,1}, m={0,1}, p={2,3,4} -> 2x2x3
    # Block D: n={0,1}, m={2,3,4}, p={2,3,4} -> 2x3x3
    # Block E: n={2}, m={0,1}, p={0,1} -> 1x2x2
    # Block F: n={2}, m={2,3,4}, p={0,1} -> 1x3x2
    # Block G: n={2}, m={0,1}, p={2,3,4} -> 1x2x3
    # Block H: n={2}, m={2,3,4}, p={2,3,4} -> 1x3x3
    
    blocks = [
        # (n_off, n_size, m_off, m_size, p_off, p_size)
        (0, 2, 0, 2, 0, 2),   # A: 2x2x2
        (0, 2, 2, 3, 0, 2),   # B: 2x3x2
        (0, 2, 0, 2, 2, 3),   # C: 2x2x3
        (0, 2, 2, 3, 2, 3),   # D: 2x3x3
        (2, 1, 0, 2, 0, 2),   # E: 1x2x2
        (2, 1, 2, 3, 0, 2),   # F: 1x3x2
        (2, 1, 0, 2, 2, 3),   # G: 1x2x3
        (2, 1, 2, 3, 2, 3),   # H: 1x3x3
    ]
    
    # Get decompositions for each block
    block_decomps = []
    for (n_off, n_sz, m_off, m_sz, p_off, p_sz) in blocks:
        U_l, V_l, W_l = get_block_decomp(n_sz, m_sz, p_sz, n_off, m_off, p_off)
        block_decomps.append((n_off, n_sz, m_off, m_sz, p_off, p_sz, U_l, V_l, W_l))
    
    total_rank = sum(b[6].shape[1] for b in block_decomps)
    
    # Build global U, V, W
    U_global = np.zeros((n * m, total_rank))
    V_global = np.zeros((m * p, total_rank))
    W_global = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for (n_off, n_sz, m_off, m_sz, p_off, p_sz, U_l, V_l, W_l) in block_decomps:
        rank_b = U_l.shape[1]
        # U: local row = (i_local * m_sz + j_local), global row = ((n_off + i_local) * m + (m_off + j_local))
        for i_local in range(n_sz):
            for j_local in range(m_sz):
                local_row = i_local * m_sz + j_local
                global_row = (n_off + i_local) * m + (m_off + j_local)
                U_global[global_row, col_offset:col_offset + rank_b] = U_l[local_row, :]
        # V: local row = (j_local * p_sz + k_local), global row = ((m_off + j_local) * p + (p_off + k_local))
        for j_local in range(m_sz):
            for k_local in range(p_sz):
                local_row = j_local * p_sz + k_local
                global_row = (m_off + j_local) * p + (p_off + k_local)
                V_global[global_row, col_offset:col_offset + rank_b] = V_l[local_row, :]
        # W: local row = (k_local * n_sz + i_local), global row = ((p_off + k_local) * n + (n_off + i_local))
        for k_local in range(p_sz):
            for i_local in range(n_sz):
                local_row = k_local * n_sz + i_local
                global_row = (p_off + k_local) * n + (n_off + i_local)
                W_global[global_row, col_offset:col_offset + rank_b] = W_l[local_row, :]
        col_offset += rank_b
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    is_valid = verify(U_global, V_global, W_global, n, m, p)
    
    if not is_valid:
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
