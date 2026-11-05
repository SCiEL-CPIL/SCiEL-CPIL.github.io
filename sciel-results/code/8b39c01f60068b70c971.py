"""Rewritten program for matrix multiplication 4x5x6 tensor rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7) with W[k*2+i, c].
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11-A22)*(B22-B11)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B22)
    # M4 = A22*(B22-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U: rows are (i,j) for i in 0..1, j in 0..1; columns are the 7 M's
    # V: rows are (j,k) for j in 0..1, k in 0..1; columns are the 7 M's
    # W: rows are (k,i) for k in 0..1, i in 0..1; columns are the 7 M's
    # Convention: W[k*n+i, c] where n=2
    
    # A indices: A[i,j], B[j,k]
    # M1: A11*B22, -A11*B11, -A22*B22, A22*B11
    # M2: A21*B11, A22*B11
    # M3: A11*B11, -A11*B22
    # M4: A22*B22, -A22*B11
    # M5: A11*B22, A12*B22
    # M6: -A11*B11, -A11*B12, A21*B11, A21*B12
    # M7: A12*B21, A12*B22, -A22*B21, -A22*B22
    
    # U[c] has entries at positions (i*m+j) = (i*2+j)
    # V[c] has entries at positions (j*p+k) = (j*2+k)
    # W[c] has entries at positions (k*n+i) = (k*2+i)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11-A22)*(B22-B11)
    # A part: A11=1, A22=-1
    # B part: B22=1, B11=-1
    # U: row 0 (i=0,j=0)=1, row 3 (i=1,j=1)=-1
    # V: row 3 (j=1,k=1)=1, row 0 (j=0,k=0)=-1
    # C = M1: C11 += 1, C22 += 1 (since (A11-A22)*(B22-B11) contributes to C11 and C22)
    # Actually let me be more careful.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # M1 = (A[0,0]-A[1,1])*(B[1,1]-B[0,0])
    # This contributes to C[0,0] with +1, C[0,1] with 0, C[1,0] with 0, C[1,1] with +1
    # Wait: (A00-A11)*(B11-B00) = A00*B11 - A00*B00 - A11*B11 + A11*B00
    # C[0,0] gets A00*B00, C[0,1] gets A00*B01, C[1,0] gets A10*B00, C[1,1] gets A11*B11
    # So M1 contributes: +A00*B11 to C[0,1]... no wait.
    # Let me just use the standard Strassen formulas directly.
    
    # Standard Strassen for 2x2:
    # M1 = (A00-A11)*(B11-B00)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B00-B11)
    # M4 = A11*(B11-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each M, determine U, V, W entries:
    # U[row_i_j, col] = coefficient of A[i,j] in M
    # V[row_j_k, col] = coefficient of B[j,k] in M
    # W[row_k_i, col] = coefficient such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = C[i,k]
    
    # M1: A part: A00=1, A11=-1; B part: B11=1, B00=-1
    # U[0,0]=1 (A00), U[3,0]=-1 (A11)
    # V[3,0]=1 (B11), V[0,0]=-1 (B00)
    # W: C00 gets +1, C11 gets +1 (from M1 in C00=M1+M4-M5+M7, C11=M1-M2+M3+M6)
    # W[0*2+0, 0]=1 (k=0,i=0), W[1*2+1, 0]=1 (k=1,i=1)
    U[0,0]=1; U[3,0]=-1
    V[3,0]=1; V[0,0]=-1
    W[0,0]=1; W[3,0]=1
    
    # M2: (A10+A11)*B00
    # U[2,1]=1 (A10), U[3,1]=1 (A11)
    # V[0,1]=1 (B00)
    # C10 gets +1 (from M2 in C10=M2+M4)
    # W[0*2+1, 1]=1 (k=0,i=1)
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1
    
    # M3: A00*(B00-B11)
    # U[0,2]=1 (A00)
    # V[0,2]=1 (B00), V[3,2]=-1 (B11)
    # C01 gets +1 (from M3 in C01=M3+M5), C11 gets +1 (from M3 in C11=M1-M2+M3+M6)
    # W[1*2+0, 2]=1 (k=1,i=0), W[1*2+1, 2]=1 (k=1,i=1)
    U[0,2]=1
    V[0,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # M4: A11*(B11-B00)
    # U[3,3]=1 (A11)
    # V[3,3]=1 (B11), V[0,3]=-1 (B00)
    # C00 gets +1, C10 gets +1
    # W[0*2+0, 3]=1 (k=0,i=0), W[0*2+1, 3]=1 (k=0,i=1)
    U[3,3]=1
    V[3,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    
    # M5: (A00+A01)*B11
    # U[0,4]=1 (A00), U[1,4]=1 (A01)
    # V[3,4]=1 (B11)
    # C00 gets -1, C01 gets +1
    # W[0*2+0, 4]=-1 (k=0,i=0), W[1*2+0, 4]=1 (k=1,i=0)
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    
    # M6: (A10-A00)*(B00+B01)
    # U[2,5]=1 (A10), U[0,5]=-1 (A00)
    # V[0,5]=1 (B00), V[1,5]=1 (B01)
    # C11 gets +1
    # W[1*2+1, 5]=1 (k=1,i=1)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M7: (A01-A11)*(B10+B11)
    # U[1,6]=1 (A01), U[3,6]=-1 (A11)
    # V[2,6]=1 (B10), V[3,6]=1 (B11)
    # C00 gets +1
    # W[0*2+0, 6]=1 (k=0,i=0)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """Return schoolbook decomposition for n_block x m_block x p_block."""
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

def _combine_blocks(blocks, n, m, p):
    """Combine a list of (U, V, W, row_offsets, col_offsets) into full matrices.
    Each block operates on a sub-matrix.
    blocks: list of (U, V, W, i_start, j_start, k_start, i_len, j_len, k_len)
    """
    total_rank = sum(b[0].shape[1] for b in blocks)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for (U_b, V_b, W_b, i_start, j_start, k_start, i_len, j_len, k_len) in blocks:
        r = U_b.shape[1]
        # U_b has shape (i_len * j_len, r), maps to rows [i_start*m+j_start : (i_start+i_len)*m + (j_start+j_len)]
        # Actually, U rows are indexed by (i*m + j) in the block's local coords
        # Global row = (i_start + i_local)*m + (j_start + j_local)
        for i_local in range(i_len):
            for j_local in range(j_len):
                global_row = (i_start + i_local) * m + (j_start + j_local)
                local_row = i_local * j_len + j_local
                U_full[global_row, col_offset:col_offset+r] = U_b[local_row, :]
        
        # V_b has shape (j_len * k_len, r)
        for j_local in range(j_len):
            for k_local in range(k_len):
                global_row = (j_start + j_local) * p + (k_start + k_local)
                local_row = j_local * k_len + k_local
                V_full[global_row, col_offset:col_offset+r] = V_b[local_row, :]
        
        # W_b has shape (k_len * i_len, r), indexed by (k*i_len + i) in local
        # Global row = (k_start + k_local)*n + (i_start + i_local)
        for k_local in range(k_len):
            for i_local in range(i_len):
                global_row = (k_start + k_local) * n + (i_start + i_local)
                local_row = k_local * i_len + i_local
                W_full[global_row, col_offset:col_offset+r] = W_b[local_row, :]
        
        col_offset += r
    
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 110  # safety margin
    
    # Strategy: Split into blocks and use Strassen where beneficial.
    # 4x5x6: Split rows 4=2+2, cols 5=2+3 or 5=1+4, etc.
    
    # Try: Split into 2x2x2 + 2x3x6 blocks, or other combinations.
    # Better: 4=2+2, 5=2+3, 6=3+3
    # Block1: 2x2x2 (Strassen, rank 7)
    # Block2: 2x2x3 (schoolbook 12 or try to improve)
    # Block3: 2x3x2 (schoolbook 12)
    # Block4: 2x3x3 (schoolbook 18)
    # Total: 7+12+12+18 = 49
    
    # Or: 4=2+2, 5=1+4, 6=3+3
    # Block1: 2x1x3 (schoolbook 6)
    # Block2: 2x4x3 (schoolbook 24)
    # Block3: 2x1x3 (schoolbook 6)
    # Block4: 2x4x3 (schoolbook 24)
    # Total: 60, worse.
    
    # Let's try: 4=2+2, 5=2+3, 6=3+3
    # Use Strassen for 2x2x2, schoolbook for others.
    
    blocks = []
    
    # Block (0,0): 2x2x2 at i_start=0, j_start=0, k_start=0
    U1, V1, W1 = _strassen_2x2x2()
    blocks.append((U1, V1, W1, 0, 0, 0, 2, 2, 2))
    
    # Block (0,1): 2x2x3 at i_start=0, j_start=2, k_start=2
    U2, V2, W2 = _schoolbook_block(2, 2, 3)
    blocks.append((U2, V2, W2, 0, 2, 2, 2, 2, 3))
    
    # Block (1,0): 2x3x2 at i_start=2, j_start=0, k_start=0
    U3, V3, W3 = _schoolbook_block(2, 3, 2)
    blocks.append((U3, V3, W3, 2, 0, 0, 2, 3, 2))
    
    # Block (1,1): 2x3x3 at i_start=2, j_start=2, k_start=2
    U4, V4, W4 = _schoolbook_block(2, 3, 3)
    blocks.append((U4, V4, W4, 2, 2, 2, 2, 3, 3))
    
    U_full, V_full, W_full = _combine_blocks(blocks, n, m, p)
    rank1 = U_full.shape[1]
    
    # Alternative: 4=2+2, 5=3+2, 6=3+3
    # Block (0,0): 2x3x3 (schoolbook 18)
    # Block (0,1): 2x2x3 (schoolbook 12)
    # Block (1,0): 2x3x3 (schoolbook 18)
    # Block (1,1): 2x2x3 (schoolbook 12)
    # Total: 60, worse.
    
    # Alternative: 4=1+3, 5=2+3, 6=3+3
    # Block (0,0): 1x2x3 (schoolbook 6)
    # Block (0,1): 1x3x3 (schoolbook 9)
    # Block (1,0): 3x2x3 (schoolbook 18)
    # Block (1,1): 3x3x3 (schoolbook 27)
    # Total: 60, worse.
    
    # Try: 4=2+2, 5=2+3, 6=2+4
    # Block (0,0): 2x2x2 (Strassen 7)
    # Block (0,1): 2x2x4 (schoolbook 16)
    # Block (1,0): 2x3x2 (schoolbook 12)
    # Block (1,1): 2x3x4 (schoolbook 24)
    # Total: 59
    
    blocks2 = []
    U1, V1, W1 = _strassen_2x2x2()
    blocks2.append((U1, V1, W1, 0, 0, 0, 2, 2, 2))
    U2, V2, W2 = _schoolbook_block(2, 2, 4)
    blocks2.append((U2, V2, W2, 0, 2, 2, 2, 2, 4))
    U3, V3, W3 = _schoolbook_block(2, 3, 2)
    blocks2.append((U3, V3, W3, 2, 0, 0, 2, 3, 2))
    U4, V4, W4 = _schoolbook_block(2, 3, 4)
    blocks2.append((U4, V4, W4, 2, 2, 2, 2, 3, 4))
    U_full2, V_full2, W_full2 = _combine_blocks(blocks2, n, m, p)
    rank2 = U_full2.shape[1]
    
    # Try: 4=2+2, 5=3+2, 6=2+4
    # Block (0,0): 2x3x2 (schoolbook 12)
    # Block (0,1): 2x2x2 (Strassen 7)
    # Block (1,0): 2x3x2 (schoolbook 12)
    # Block (1,1): 2x2x2 (Strassen 7)
    # Total: 38!
    
    blocks3 = []
    U1, V1, W1 = _schoolbook_block(2, 3, 2)
    blocks3.append((U1, V1, W1, 0, 0, 0, 2, 3, 2))
    U2, V2, W2 = _strassen_2x2x2()
    blocks3.append((U2, V2, W2, 0, 3, 2, 2, 2, 2))
    U3, V3, W3 = _schoolbook_block(2, 3, 2)
    blocks3.append((U3, V3, W3, 2, 0, 0, 2, 3, 2))
    U4, V4, W4 = _strassen_2x2x2()
    blocks3.append((U4, V4, W4, 2, 3, 2, 2, 2, 2))
    U_full3, V_full3, W_full3 = _combine_blocks(blocks3, n, m, p)
    rank3 = U_full3.shape[1]
    
    # Try: 4=2+2, 5=2+3, 6=2+4 with Strassen on 2x2x2 blocks
    # Already did above (rank2=59)
    
    # Try: 4=2+2, 5=2+3, 6=4+2
    # Block (0,0): 2x2x4 (schoolbook 16)
    # Block (0,1): 2x3x4 (schoolbook 24)
    # Block (1,0): 2x2x2 (Strassen 7)
    # Block (1,1): 2x3x2 (schoolbook 12)
    # Total: 59
    
    # Best so far is rank3=38
    # Let's verify: 2*3*2 + 7 + 2*3*2 + 7 = 12+7+12+7 = 38. Yes!
    
    # Can we do better? Try 4=2+2, 5=3+2, 6=4+2
    # Block (0,0): 2x3x4 (24)
    # Block (0,1): 2x2x4 (16)
    # Block (1,0): 2x3x2 (12)
    # Block (1,1): 2x2x2 (7)
    # Total: 59
    
    # Try 4=2+2, 5=4+1, 6=3+3
    # Block (0,0): 2x4x3 (24)
    # Block (0,1): 2x1x3 (6)
    # Block (1,0): 2x4x3 (24)
    # Block (1,1): 2x1x3 (6)
    # Total: 60
    
    # Try 4=3+1, 5=3+2, 6=3+3
    # Block (0,0): 3x3x3 (27)
    # Block (0,1): 3x2x3 (18)
    # Block (1,0): 1x3x3 (9)
    # Block (1,1): 1x2x3 (6)
    # Total: 60
    
    # The 38-rank solution is good. Let's also try if we can use Strassen-like
    # decompositions for 2x3x2 blocks.
    # 2x3x2: Is there a rank < 12 decomposition?
    # The tensor is M_{2,3,2}. Known lower bound is around 10-11.
    # Let's try a simple approach: treat 2x3x2 as 2x2x2 + 2x1x2
    # 2x2x2: Strassen 7
    # 2x1x2: schoolbook 4
    # Total: 11
    
    # Let's build a 2x3x2 decomposition using Strassen(2x2x2) + schoolbook(2x1x2)
    def _decomp_2x3x2():
        """Decompose 2x3x2 using Strassen for 2x2x2 part and schoolbook for 2x1x2 part."""
        # Split the 3 columns into 2+1
        # Part 1: 2x2x2 (Strassen)
        U_s, V_s, W_s = _strassen_2x2x2()
        # Part 2: 2x1x2 (schoolbook)
        U_e, V_e, W_e = _schoolbook_block(2, 1, 2)
        
        # Combine: total rank = 7 + 4 = 11
        U = np.zeros((2*3, 11))
        V = np.zeros((3*2, 11))
        W = np.zeros((2*2, 11))
        
        # Part 1: j=0,1 (first 2 of 3 columns)
        U[:, :7] = U_s  # U_s has shape (2*2, 7) = (4, 7), but we need (6, 11)
        # Actually U_s rows are (i*2+j) for i in 0..1, j in 0..1
        # We need U rows (i*3+j) for i in 0..1, j in 0..2
        # For j=0,1: row i*3+j
        for i in range(2):
            for j in range(2):
                U[i*3+j, :7] = U_s[i*2+j, :]
        
        # V_s rows are (j*2+k) for j in 0..1, k in 0..1
        # We need V rows (j*2+k) for j in 0..2, k in 0..1
        # For j=0,1: row j*2+k
        for j in range(2):
            for k in range(2):
                V[j*2+k, :7] = V_s[j*2+k, :]
        
        # W_s rows are (k*2+i) for k in 0..1, i in 0..1
        # We need W rows (k*2+i) for k in 0..1, i in 0..1
        # Same indexing!
        W[:, :7] = W_s
        
        # Part 2: j=2 (last 1 of 3 columns)
        # U_e has shape (2*1, 4) = (2, 4), rows (i*1+j) = i
        # We need U rows (i*3+2) for i in 0..1
        for i in range(2):
            U[i*3+2, 7:11] = U_e[i, :]
        
        # V_e has shape (1*2, 4) = (2, 4), rows (j*2+k) = k
        # We need V rows (2*2+k) = 4+k for k in 0..1
        for k in range(2):
            V[4+k, 7:11] = V_e[k, :]
        
        # W_e has shape (2*2, 4) = (4, 4), rows (k*2+i)
        # We need W rows (k*2+i) for k in 0..1, i in 0..1
        W[:, 7:11] = W_e
        
        return U, V, W
    
    # Now rebuild with improved 2x3x2 blocks
    blocks4 = []
    U1, V1, W1 = _decomp_2x3x2()
    blocks4.append((U1, V1, W1, 0, 0, 0, 2, 3, 2))
    U2, V2, W2 = _strassen_2x2x2()
    blocks4.append((U2, V2, W2, 0, 3, 2, 2, 2, 2))
    U3, V3, W3 = _decomp_2x3x2()
    blocks4.append((U3, V3, W3, 2, 0, 0, 2, 3, 2))
    U4, V4, W4 = _strassen_2x2x2()
    blocks4.append((U4, V4, W4, 2, 3, 2, 2, 2, 2))
    U_full4, V_full4, W_full4 = _combine_blocks(blocks4, n, m, p)
    rank4 = U_full4.shape[1]
    # Should be 11+7+11+7 = 36
    
    # Choose the best
    candidates = [
        (rank1, U_full, V_full, W_full),
        (rank2, U_full2, V_full2, W_full2),
        (rank3, U_full3, V_full3, W_full3),
        (rank4, U_full4, V_full4, W_full4),
    ]
    candidates.sort(key=lambda x: x[0])
    best_rank, best_U, best_V, best_W = candidates[0]
    
    # Convert to lists with integer/half-integer entries
    U_list = [[(int(v) if v == int(v) else v) for v in row] for row in best_U.tolist()]
    V_list = [[(int(v) if v == int(v) else v) for v in row] for row in best_V.tolist()]
    W_list = [[(int(v) if v == int(v) else v) for v in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
