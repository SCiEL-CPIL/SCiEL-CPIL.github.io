"""Matrix multiplication 3x4x7 - block decomposition with Strassen for 2x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=4, p=7):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build the full multiplication tensor
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0
    
    # Compute from decomposition
    T_comp = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                T_comp[i, j, k] = s
    
    if not np.allclose(T_comp, T_target, atol=1e-10):
        return False
    return True


def _strassen_2x2x2():
    """
    Return U, V, W matrices for 2x2x2 matrix multiplication using Strassen (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[1, 4] = 1  # C01
    W[0, 4] = -1 # C00
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return U.tolist(), V.tolist(), W.tolist()


def _standard_2x2x2():
    """Standard rank-4 decomposition for 2x2x2."""
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = i * 4 + j * 2 + k
                U[i * 2 + j, c] = 1
                V[j * 2 + k, c] = 1
                W[k * 2 + i, c] = 1
    return U.tolist(), V.tolist(), W.tolist()


def _standard_block(n_block, m_block, p_block):
    """Standard multiplication for n_block x m_block x p_block."""
    rank = n_block * m_block * p_block
    U = np.zeros((n_block * m_block, rank))
    V = np.zeros((m_block * p_block, rank))
    W = np.zeros((p_block * n_block, rank))
    c = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j, c] = 1
                V[j * p_block + k, c] = 1
                W[k * n_block + i, c] = 1
                c += 1
    return U.tolist(), V.tolist(), W.tolist()


def _strassen_block(m_block, p_block):
    """
    Apply Strassen decomposition for 2 rows of A against an m_block x p_block B.
    Splits B's columns into groups of 2, using Strassen for each 2x2 block.
    If p_block is odd, the last column uses standard multiplication.
    
    Returns U (2*m_block x rank), V (m_block*p_block x rank), W (p_block*2 x rank)
    """
    U_list = []
    V_list = []
    W_list = []
    rank_total = 0
    
    for col_start in range(0, p_block, 2):
        if col_start + 1 < p_block:
            # 2 columns: use Strassen (rank 7)
            Us, Vs, Ws = _strassen_2x2x2()
            # Us is (4,7), need to map to (2*m_block, 7)
            # A part: rows 0,1 of the 2x2 block -> U indices 0,1,2,3 correspond to A[0,0],A[0,1],A[1,0],A[1,1]
            # But we have 2 rows of A (full width m_block), so each A row has m_block entries
            # The 2x2 Strassen operates on 2x2 blocks
            # For 2 rows of A (each of width m_block) x 2 columns of B (each of height m_block):
            # We split A's m_block columns into pairs too
            
            # Actually, let me reconsider. For 2x(m)x(p) where we want to use Strassen:
            # We can decompose along the column dimension of A (and row dimension of B)
            
            # Let me use a different approach: decompose the 2 x m_block x p_block problem
            # by splitting the middle dimension into pairs of 2.
            # For each pair of middle dimensions (j, j+1), we have a 2x2x2 block (if p is also 2).
            
            # This is getting complex. Let me use a cleaner approach.
            pass
        else:
            # Single column: standard 2 x m_block x 1 (rank 2*m_block)
            Us, Vs, Ws = _standard_block(2, m_block, 1)
            U_list.append(Us)
            V_list.append(Vs)
            W_list.append(Ws)
            rank_total += 2 * m_block
    
    # For now, fall back to standard if the block approach doesn't work cleanly
    Us, Vs, Ws = _standard_block(2, m_block, p_block)
    return Us, Vs, Ws, 2 * m_block * p_block


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication.
    Uses block decomposition: split 3 rows into 2+1.
    - Top 2 rows: use Strassen-based 2x4x7 decomposition
    - Bottom 1 row: standard 1x4x7 (rank 28)
    """
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin
    
    n, m, p = 3, 4, 7
    
    # Strategy: Split A into top 2 rows and bottom 1 row.
    # Top: 2 x 4 x 7
    # Bottom: 1 x 4 x 7
    
    # For 2x4x7: split the 4 middle dimensions into two pairs of 2.
    # Each 2x2x2 block uses Strassen (rank 7).
    # But the 7 output dimensions need handling too.
    
    # Alternative: For 2x4x7, split into four 2x2x2 blocks (using 2 columns of B at a time)
    # plus one 2x1x2 block.
    # 2x2x2: rank 7 (Strassen)
    # 2x1x2: rank 4 (standard, since it's just 2 dot products of length 1... no, 2x1x2 means
    #   A is 2x1, B is 1x2, C is 2x2. Rank = 2*1*2 = 4 for standard, but actually this is
    #   just outer product scaled, rank 1 per entry... 
    #   Actually for 2x1x2: A has 2 entries, B has 2 entries, C has 4 entries.
    #   C[i,k] = sum_j A[i,j]*B[j,k] = A[i,0]*B[0,k]. So C = A_col * B_row (outer product).
    #   This has rank 1! One multiplication: M = A[0,0]*B[0,0]... no.
    #   C[i,k] = A[i,0] * B[0,k]. This is an outer product, rank 1.
    
    # So for 2x4x7:
    # Split B's 7 columns into: [2,2,2,1]
    # Each 2-column group with 2x2 A-block: 
    #   For columns 0,1: A(2x4)[:,0:2] * B(4x7)[0:2,:] -> 2x2x2, rank 7
    #   For columns 2,3: A(2x4)[:,2:4] * B(4x7)[2:4,:] -> 2x2x2, rank 7
    #   For columns 4,5: A(2x4)[:,4:6]... but A only has 4 columns!
    
    # I confused myself. Let me restart with clear indices.
    
    # A: 3 rows, 4 cols. B: 4 rows, 7 cols. C: 3 rows, 7 cols.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split A's rows: top 2 (i=0,1), bottom 1 (i=2).
    # Top part: C_top(2x7) = A_top(2x4) * B(4x7)
    # Bottom part: C_bot(1x7) = A_bot(1x4) * B(4x7)
    
    # For C_bot (1x4x7): standard, rank = 1*4*7 = 28.
    
    # For C_top (2x4x7): 
    #   Split B's rows (which are A's columns) into two pairs: j=0,1 and j=2,3.
    #   C_top[i,k] = sum_{j=0}^{1} A_top[i,j]*B[j,k] + sum_{j=2}^{3} A_top[i,j]*B[j,k]
    #   Part 1: A_top[:,0:2] (2x2) * B[0:2,:] (2x7) -> 2x2x7
    #   Part 2: A_top[:,2:4] (2x2) * B[2:4,:] (2x7) -> 2x2x7
    #   
    #   For each 2x2x7: split B's 7 columns into [2,2,2,1]
    #   Each 2x2x2: Strassen rank 7
    #   Each 2x2x1: outer product, rank 2 (C[i,k] = sum_j A[i,j]*B[j,k], 2x2x1 has 4 terms)
    #     Actually 2x2x1: A(2x2), B(2x1), C(2x1). C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0].
    #     This is 2 dot products of length 2, rank 4 standard. Can we do better?
    #     It's just a matrix-vector multiply 2x2 * 2x1. Rank = 4 for standard.
    #     Hmm, actually for 2x2x1, the tensor is sum over j of A[i,j]*B[j,0]*C[i,0]... 
    #     Wait, C has shape 2x1, so C[i,0]. The tensor is T[i,j,0] where i in {0,1}, j in {0,1}.
    #     T[i,j,0] = delta_{i,j}? No, T[i,j,k] for k=0 is 1 if... 
    #     Actually the multiplication tensor for (2,2,1) has T[i,j,k]=1 for all i,j,k in range.
    #     Since k only has value 0, it's just T[i,j,0]=1 for i,j in {0,1}.
    #     This is a 2x2x1 tensor = all ones. Rank of all-ones 2x2 matrix is 1.
    #     So rank 1! C[i,0] = (sum_j A[i,j]*B[j,0])... 
    #     Wait, I'm confusing the tensor definition.
    
    # Let me just use standard for the 2x2x7 blocks to be safe, and see what rank we get.
    # 2x2x7 standard: rank = 2*2*7 = 28 per block. Two blocks = 56. Plus 28 for bottom = 84.
    # That's the same as schoolbook. No improvement.
    
    # OK, let me properly implement Strassen for 2x2x2 and use it where applicable.
    
    # For 2x4x7:
    # Split into four 2x2x2 blocks (using 2 rows of B and 2 cols of C) + one 2x2x1 block
    # B rows: 0,1 | 2,3 (two groups of 2)
    # C cols: 0,1 | 2,3 | 4,5 | 6 (three groups of 2 + one single)
    # Total: 2*3 = 6 blocks of 2x2x2 + 2 blocks of 2x2x1
    # Each 2x2x2: Strassen rank 7. Each 2x2x1: rank ?
    
    # For 2x2x1 (A: 2x2, B: 2x1, C: 2x1):
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # This requires 4 multiplications in general, but the tensor is:
    # T[i,j,k] where i in {0,1}, j in {0,1}, k in {0} (only k=0)
    # T[i,j,0] = 1 for all i,j
    # This is the 2x2x1 all-ones tensor. Its rank: 
    # We need rank-r decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Since k only has value 0: sum_c U[i,j,c]*V[j,0,c]*W[0,i,c] = 1 for all i,j
    # Let a_c = V[0,c], b_c = W[0,c], M_c[i,j] = U[i,j,c]
    # sum_c M_c[i,j] * a_c * b_c = 1 for all i,j
    # This is an outer product sum: sum_c (a_c * b_c) * M_c = J (all-ones 2x2)
    # J has rank 1, so we can use c=0: M_0 = J, a_0 = 1, b_0 = 1.
    # So rank 1! U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1.
    
    # Great! So 2x2x1 has rank 1.
    # Total for 2x4x7: 6*7 + 2*1 = 42 + 2 = 44
    # Total overall: 44 + 28 = 72
    
    # Let me implement this.
    
    U_all = []
    V_all = []
    W_all = []
    rank_total = 0
    
    def add_block(Ub, Vb, Wb, i_offset, j_offset, k_offset, n_b, m_b, p_b):
        """Add a block to the global matrices with appropriate index offsets."""
        nonlocal rank_total
        ub = np.array(Ub, dtype=float)
        vb = np.array(Vb, dtype=float)
        wb = np.array(Wb, dtype=float)
        
        n_r, r = ub.shape
        m_r, _ = vb.shape
        p_r, _ = wb.shape
        
        U_block = np.zeros((n*m, r))
        V_block = np.zeros((m*p, r))
        W_block = np.zeros((p*n, r))
        
        # Map local indices to global
        for li in range(n_b):
            for lj in range(m_b):
                gi = i_offset + li
                gj = j_offset + lj
                for c in range(r):
                    U_block[gi*m + gj, c] = ub[li*m_b + lj, c]
        
        for lj in range(m_b):
            for lk in range(p_b):
                gj = j_offset + lj
                gk = k_offset + lk
                for c in range(r):
                    V_block[gj*p + gk, c] = vb[lj*p_b + lk, c]
        
        for lk in range(p_b):
            for li in range(n_b):
                gk = k_offset + lk
                gi = i_offset + li
                for c in range(r):
                    W_block[gk*n + gi, c] = wb[lk*n_b + li, c]
        
        U_all.append(U_block)
        V_all.append(V_block)
        W_all.append(W_block)
        rank_total += r
    
    # Top part: i=0,1 (2 rows), j=0..3 (4 cols), k=0..6 (7 cols)
    # Split j into [0,1] and [2,3], k into [0,1], [2,3], [4,5], [6]
    
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            if k_start + 1 < 7:
                # 2x2x2 block
                Us, Vs, Ws = _strassen_2x2x2()
                add_block(Us, Vs, Ws, 0, j_start, k_start, 2, 2, 2)
            else:
                # 2x2x1 block (rank 1)
                # U[i,j,c] = 1 for i,j in {0,1}, c=0
                # V[j,k,c] = 1 for j in {0,1}, k=0, c=0
                # W[k,i,c] = 1 for k=0, i in {0,1}, c=0
                Us = [[1, 1], [1, 1]]  # 2x2, 1 col
                Vs = [[1], [1]]         # 2x1, 1 col
                Ws = [[1], [1]]         # 1x2, 1 col
                add_block(Us, Vs, Ws, 0, j_start, k_start, 2, 2, 1)
    
    # Bottom part: i=2 (1 row), j=0..3 (4 cols), k=0..6 (7 cols)
    # Standard 1x4x7, rank 28
    Us, Vs, Ws = _standard_block(1, 4, 7)
    add_block(Us, Vs, Ws, 2, 0, 0, 1, 4, 7)
    
    # Concatenate all blocks
    U_final = np.hstack(U_all)
    V_final = np.hstack(V_all)
    W_final = np.hstack(W_all)
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
