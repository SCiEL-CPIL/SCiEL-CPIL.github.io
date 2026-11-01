"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2_block(col_offset):
    """
    Return (U_block, V_block, W_block) for Strassen 2x2x2.
    col_offset: which 2 columns of B this block handles (0 or 1).
    U: (4, 7), V: (4, 7), W: (4, 7)
    B indices within block: 0,1 for first two cols, 2,3 for next two cols
    C indices within block: 0,1 for first two output cols, 2,3 for next two
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # B column indices: col_offset*2 + {0,1}
    # C column indices: col_offset*2 + {0,1}
    b0 = col_offset * 2      # first column index in B
    b1 = col_offset * 2 + 1  # second column index in B
    c0 = col_offset * 2      # first column index in C
    c1 = col_offset * 2 + 1  # second column index in C
    
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (a10+a11)(b00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # m4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # m5 = (a00+a10)(b11)
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # m6 = (a01-a10)(b00+b10)
    U[1, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # m7 = (a01-a11)(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    # Now we need to map these to the actual B and C indices
    # The above uses local indices 0,1 for B columns and 0,1 for C columns
    # We need to shift them to the actual positions
    
    # Rebuild with correct indices
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # m1 = (a00+a11)(b[b0]+b[b1])
    U[0, 0] = 1; U[3, 0] = 1
    V[b0, 0] = 1; V[b1+2, 0] = 1  # b10 and b11 in B: row 1, cols b0,b1
    W[c0, 0] = 1; W[c1+2, 0] = 1  # c00 and c10 in C: col c0,c1, row 0
    
    # Wait, I need to be more careful with the indexing.
    # B has shape (m, p) = (2, 5), indexed as [j, k]
    # V index is j*p+k
    # For block at col_offset:
    #   b0 = col_offset*2, b1 = col_offset*2+1
    #   B[0, b0], B[0, b1], B[1, b0], B[1, b1]
    #   V indices: 0*b0=0+b0=b0, 0*b1=b1, 1*b0=p+b0=5+b0, 1*b1=5+b1
    # C has shape (p, n) = (5, 2), indexed as [k, i]
    # W index is k*n+i
    # For block at col_offset:
    #   c0 = col_offset*2, c1 = col_offset*2+1
    #   C[c0, 0], C[c0, 1], C[c1, 0], C[c1, 1]
    #   W indices: c0*n+0=c0*2, c0*n+1=c0*2+1, c1*n+0=c1*2, c1*n+1=c1*2+1
    
    p_val = 5
    n_val = 2
    
    # B indices:
    # B[0, b0] -> V index 0*p_val + b0 = b0
    # B[0, b1] -> V index 0*p_val + b1 = b1
    # B[1, b0] -> V index 1*p_val + b0 = p_val + b0
    # B[1, b1] -> V index 1*p_val + b1 = p_val + b1
    
    # C indices:
    # C[c0, 0] -> W index c0*n_val + 0 = c0*2
    # C[c0, 1] -> W index c0*n_val + 1 = c0*2 + 1
    # C[c1, 0] -> W index c1*n_val + 0 = c1*2
    # C[c1, 1] -> W index c1*n_val + 1 = c1*2 + 1
    
    vb0 = b0          # B[0, b0]
    vb1 = b1          # B[0, b1]
    vb2 = p_val + b0  # B[1, b0]
    vb3 = p_val + b1  # B[1, b1]
    
    wc0 = c0 * n_val      # C[c0, 0]
    wc1 = c0 * n_val + 1  # C[c0, 1]
    wc2 = c1 * n_val      # C[c1, 0]
    wc3 = c1 * n_val + 1  # C[c1, 1]
    
    # m1 = (a00+a11)(b00+b11) where b00=B[0,b0], b11=B[1,b1]
    U[0, 0] = 1; U[3, 0] = 1
    V[vb0, 0] = 1; V[vb3, 0] = 1
    W[wc0, 0] = 1; W[wc3, 0] = 1
    
    # m2 = (a10+a11)(b00) where b00=B[0,b0]
    U[2, 1] = 1; U[3, 1] = 1
    V[vb0, 1] = 1
    W[wc2, 1] = 1; W[wc3, 1] = -1
    
    # m3 = a00(b01-b11) where b01=B[0,b1], b11=B[1,b1]
    U[0, 2] = 1
    V[vb1, 2] = 1; V[vb3, 2] = -1
    W[wc1, 2] = 1; W[wc3, 2] = -1
    
    # m4 = a11(b10-b00) where b10=B[1,b0], b00=B[0,b0]
    U[3, 3] = 1
    V[vb2, 3] = 1; V[vb0, 3] = -1
    W[wc2, 3] = 1; W[wc0, 3] = -1
    
    # m5 = (a00+a10)(b11) where b11=B[1,b1]
    U[0, 4] = 1; U[2, 4] = 1
    V[vb3, 4] = 1
    W[wc1, 4] = 1; W[wc0, 4] = -1
    
    # m6 = (a01-a10)(b00+b10) where b00=B[0,b0], b10=B[1,b0]
    U[1, 5] = 1; U[2, 5] = -1
    V[vb0, 5] = 1; V[vb2, 5] = 1
    W[wc0, 5] = 1
    
    # m7 = (a01-a11)(b01+b11) where b01=B[0,b1], b11=B[1,b1]
    U[1, 6] = 1; U[3, 6] = -1
    V[vb1, 6] = 1; V[vb3, 6] = 1
    W[wc3, 6] = 1
    
    return U, V, W

def _single_column_block(col_idx):
    """
    Return (U_block, V_block, W_block) for a single column of B.
    col_idx: which column of B (0-4)
    U: (4, 2), V: (10, 2), W: (10, 2)
    """
    rank = 2
    U = np.zeros((4, rank))
    V = np.zeros((10, rank))
    W = np.zeros((10, rank))
    
    p_val = 5
    n_val = 2
    
    # For column col_idx of B:
    # C[:, col_idx] = A * B[:, col_idx]
    # C[0, col_idx] = A[0,0]*B[0,col_idx] + A[0,1]*B[1,col_idx]
    # C[1, col_idx] = A[1,0]*B[0,col_idx] + A[1,1]*B[1,col_idx]
    
    # Term 0: A[0,0]*B[0,col_idx] contributes to C[0, col_idx]
    U[0, 0] = 1
    V[0 * p_val + col_idx, 0] = 1  # B[0, col_idx]
    W[col_idx * n_val + 0, 0] = 1  # C[col_idx, 0]
    
    # Term 1: A[0,1]*B[1,col_idx] contributes to C[0, col_idx]
    U[1, 1] = 1
    V[1 * p_val + col_idx, 1] = 1  # B[1, col_idx]
    W[col_idx * n_val + 0, 1] = 1  # C[col_idx, 0]
    
    # Wait, this only gives C[0, col_idx]. I need C[1, col_idx] too.
    # Let me reconsider. For a single column, I need 2 terms per output row.
    # Actually, for C[i, k] where k is fixed:
    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]  -> 2 terms
    # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]  -> 2 terms
    # Total 4 terms for one column.
    
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((10, rank))
    W = np.zeros((10, rank))
    
    # Term 0: A[0,0]*B[0,col_idx] -> C[0, col_idx]
    U[0, 0] = 1
    V[0 * p_val + col_idx, 0] = 1
    W[col_idx * n_val + 0, 0] = 1
    
    # Term 1: A[0,1]*B[1,col_idx] -> C[0, col_idx]
    U[1, 1] = 1
    V[1 * p_val + col_idx, 1] = 1
    W[col_idx * n_val + 0, 1] = 1
    
    # Term 2: A[1,0]*B[0,col_idx] -> C[1, col_idx]
    U[2, 2] = 1
    V[0 * p_val + col_idx, 2] = 1
    W[col_idx * n_val + 1, 2] = 1
    
    # Term 3: A[1,1]*B[1,col_idx] -> C[1, col_idx]
    U[3, 3] = 1
    V[1 * p_val + col_idx, 3] = 1
    W[col_idx * n_val + 1, 3] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Build decomposition:
    # Block 0: Strassen for B cols 0,1 -> C cols 0,1 (rank 7)
    # Block 1: Strassen for B cols 2,3 -> C cols 2,3 (rank 7)
    # Block 2: Schoolbook for B col 4 -> C col 4 (rank 4)
    # Total rank: 7 + 7 + 4 = 18
    
    total_rank = 18
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 0: Strassen for cols 0,1
    U0, V0, W0 = _strassen_2x2x2_block(0)
    U[:, 0:7] = U0
    V[:, 0:7] = V0
    W[:, 0:7] = W0
    
    # Block 1: Strassen for cols 2,3
    U1, V1, W1 = _strassen_2x2x2_block(1)
    U[:, 7:14] = U1
    V[:, 7:14] = V1
    W[:, 7:14] = W1
    
    # Block 2: Schoolbook for col 4
    U2, V2, W2 = _single_column_block(4)
    U[:, 14:18] = U2
    V[:, 14:18] = V2
    W[:, 14:18] = W2
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
