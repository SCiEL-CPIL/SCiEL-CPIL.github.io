"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _build_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c] for n x m, m x p."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Reconstruct T from the decomposition
    T_recon = np.zeros_like(T, dtype=float)
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        T_recon += np.einsum('a,b,c->abc', u, v, w)
    
    if not np.allclose(T_recon, T, atol=tol):
        return False
    return True


def _strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for matrix multiplication.
    
    For multiplying an n x m matrix by an m x p matrix, we can use
    Strassen's algorithm when dimensions allow. The key idea is to
    partition the matrices and use 7 multiplications instead of 8
    for 2x2 blocks.
    
    For n=2, m=4, p=5, we can partition as:
    - A (2x4) into two 2x2 blocks: A1, A2
    - B (4x5) into blocks: top 2x5, bottom 2x5
    - We need to handle the non-power-of-2 dimensions carefully.
    
    A simpler approach: use the known rank-36 decomposition for 
    2x4 by 4x5 if it exists, or construct one via Strassen on 
    appropriate sub-blocks.
    
    Let's use a block Strassen approach:
    Partition A (2x4) as [A_left | A_right] where A_left, A_right are 2x2
    Partition B (4x5) as [B_top; B_bottom] where B_top, B_bottom are 2x5
    
    C = A_left @ B_top + A_right @ B_bottom
    
    We can apply Strassen to each 2x2 @ 2x5 multiplication, but 
    that doesn't directly help since the second factor isn't square.
    
    Alternative: pad to 2x4 @ 4x4 and then handle the 5th column separately.
    
    Actually, let's think about this differently. The known result is that
    for 2x4 @ 4x5, a rank-36 decomposition exists using Strassen-like
    techniques. Let me construct it explicitly.
    
    Strategy: 
    1. Split A (2x4) into A1 (2x2) and A2 (2x2)
    2. Split B (4x5) into B1 (2x5) and B2 (2x5)
    3. C = A1 @ B1 + A2 @ B2
    
    For each 2x2 @ 2x5 product, we can use a rank-7 Strassen-like 
    decomposition (since 2x2 @ 2x2 has rank 7, and 2x2 @ 2x5 can be
    handled similarly by splitting the 5 columns).
    
    Actually, let me use a more direct approach. The key insight from the
    inspiration is that a block Strassen-based rank 36 decomposition exists.
    
    Let me construct it by treating the problem as two 2x2 @ 2x5 multiplications
    and using rank-7 decompositions for each, but with sharing.
    
    For 2x2 @ 2x2, Strassen uses 7 multiplications:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 2x2 @ 2x5, we can split the 5 columns into 2+2+1 and handle each
    2x2 @ 2x2 block with Strassen (rank 7) and the 2x2 @ 2x1 with schoolbook (rank 2).
    
    But this gives 7+7+2 = 16 per 2x2@2x5, and we have two such products,
    giving 32 total. Plus we need to combine them, but the combination
    is linear (addition), so no extra rank needed.
    
    Wait, that gives rank 32, which is better than 36. Let me verify this.
    
    Actually, let me be more careful. Let me just implement the known
    rank-36 decomposition directly.
    """
    # For now, let's return None to indicate we need to construct it
    # We'll implement a specific construction below
    return None


def _construct_rank36(n, m, p):
    """
    Construct a rank-36 decomposition for 2x4 @ 4x5 matrix multiplication.
    
    This uses a Strassen-based approach:
    - Split A (2x4) into A_left (2x2) and A_right (2x2)
    - Split B (4x5) into B_top (2x5) and B_bottom (2x5)
    - C = A_left @ B_top + A_right @ B_bottom
    
    For each 2x2 @ 2x5, use a rank-18 decomposition (3x Strassen 2x2@2x2 
    plus some adjustments), but with sharing between the two products.
    
    Actually, let me use a cleaner approach:
    
    Partition:
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x5
    
    C = A1 @ B1 + A2 @ B2
    
    For A1 @ B1 (2x2 @ 2x5):
    Split B1 = [B1a | B1b | B1c] where B1a, B1b are 2x2 and B1c is 2x1
    A1 @ B1 = A1 @ B1a + A1 @ B1b + A1 @ B1c
    Use Strassen for A1 @ B1a (rank 7), A1 @ B1b (rank 7), schoolbook for A1 @ B1c (rank 2)
    Total: 16
    
    Similarly for A2 @ B2: 16
    
    Total: 32
    
    But we can do better by sharing. Let me think...
    
    Actually, the known rank-36 might come from a different partition.
    Let me just implement the straightforward 32-rank version and verify.
    """
    rank = 32
    nm, mp, pn = n * m, m * p, p * n
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # A is 2x4, indices: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # a-index = i * 4 + j
    # B is 4x5, indices: B[j, k] where j in {0,1,2,3}, k in {0,1,2,3,4}
    # b-index = j * 5 + k
    # C is 2x5, indices: C[i, k] where i in {0,1}, k in {0,1,2,3,4}
    # c-index = k * 2 + i
    
    # Split A into A1 (cols 0-1) and A2 (cols 2-3)
    # Split B into B1 (rows 0-1) and B2 (rows 2-3)
    # C = A1 @ B1 + A2 @ B2
    
    # For A1 @ B1 (2x2 @ 2x5):
    # Split B1 into B1a (cols 0-1), B1b (cols 2-3), B1c (col 4)
    # A1 @ B1a: Strassen (rank 7)
    # A1 @ B1b: Strassen (rank 7)  
    # A1 @ B1c: schoolbook (rank 2)
    
    # For A2 @ B2 (2x2 @ 2x5):
    # Split B2 into B2a (cols 0-1), B2b (cols 2-3), B2c (col 4)
    # A2 @ B2a: Strassen (rank 7)
    # A2 @ B2b: Strassen (rank 7)
    # A2 @ B2c: schoolbook (rank 2)
    
    # Total: 7 + 7 + 2 + 7 + 7 + 2 = 32
    
    def add_strassen_2x2_2x2(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset, start_col):
        """Add Strassen decomposition for a 2x2 @ 2x2 sub-problem."""
        # A sub-block: rows A_row_offset..A_row_offset+1, cols A_col_offset..A_col_offset+1
        # B sub-block: rows B_row_offset..B_row_offset+1, cols B_col_offset..B_col_offset+1
        # C sub-block: rows C_row_offset..C_row_offset+1, cols C_col_offset..C_col_offset+1
        
        # Strassen 7 multiplications:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices in global:
        # A11: (A_row_offset, A_col_offset)
        # A12: (A_row_offset, A_col_offset+1)
        # A21: (A_row_offset+1, A_col_offset)
        # A22: (A_row_offset+1, A_col_offset+1)
        
        # B indices in global:
        # B11: (B_row_offset, B_col_offset)
        # B12: (B_row_offset, B_col_offset+1)
        # B21: (B_row_offset+1, B_col_offset)
        # B22: (B_row_offset+1, B_col_offset+1)
        
        # C indices in global:
        # C11: (C_row_offset, C_col_offset)
        # C12: (C_row_offset, C_col_offset+1)
        # C21: (C_row_offset+1, C_col_offset)
        # C22: (C_row_offset+1, C_col_offset+1)
        
        def a_idx(i, j):
            return i * m + j
        
        def b_idx(i, j):
            return i * p + j
        
        def c_idx(i, j):
            return j * n + i
        
        A11 = a_idx(A_row_offset, A_col_offset)
        A12 = a_idx(A_row_offset, A_col_offset + 1)
        A21 = a_idx(A_row_offset + 1, A_col_offset)
        A22 = a_idx(A_row_offset + 1, A_col_offset + 1)
        
        B11 = b_idx(B_row_offset, B_col_offset)
        B12 = b_idx(B_row_offset, B_col_offset + 1)
        B21 = b_idx(B_row_offset + 1, B_col_offset)
        B22 = b_idx(B_row_offset + 1, B_col_offset + 1)
        
        C11 = c_idx(C_row_offset, C_col_offset)
        C12 = c_idx(C_row_offset, C_col_offset + 1)
        C21 = c_idx(C_row_offset + 1, C_col_offset)
        C22 = c_idx(C_row_offset + 1, C_col_offset + 1)
        
        # M1 = (A11 + A22)(B11 + B22)
        U[A11][start_col] = 1
        U[A22][start_col] = 1
        V[B11][start_col] = 1
        V[B22][start_col] = 1
        W[C11][start_col] = 1
        W[C12][start_col] = 0  # M1 contributes to C11 and C22
        W[C21][start_col] = 0
        W[C22][start_col] = 1
        col = start_col + 1
        
        # M2 = (A21 + A22)B11
        U[A21][col] = 1
        U[A22][col] = 1
        V[B11][col] = 1
        W[C21][col] = 1
        col += 1
        
        # M3 = A11(B12 - B22)
        U[A11][col] = 1
        V[B12][col] = 1
        V[B22][col] = -1
        W[C12][col] = 1
        col += 1
        
        # M4 = A22(B21 - B11)
        U[A22][col] = 1
        V[B21][col] = 1
        V[B11][col] = -1
        W[C11][col] = 1
        W[C21][col] = 1
        col += 1
        
        # M5 = (A11 + A12)B22
        U[A11][col] = 1
        U[A12][col] = 1
        V[B22][col] = 1
        W[C11][col] = -1
        W[C12][col] = 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[A21][col] = 1
        U[A11][col] = -1
        V[B11][col] = 1
        V[B12][col] = 1
        W[C22][col] = 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[A12][col] = 1
        U[A22][col] = -1
        V[B21][col] = 1
        V[B22][col] = 1
        W[C11][col] = 1
        W[C22][col] = 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_2x1(A_row_offset, A_col_offset, B_row_offset, B_col, C_row_offset, C_col, start_col):
        """Add schoolbook decomposition for a 2x2 @ 2x1 sub-problem."""
        # A sub-block: rows A_row_offset..A_row_offset+1, cols A_col_offset..A_col_offset+1
        # B sub-block: rows B_row_offset..B_row_offset+1, col B_col
        # C sub-block: rows C_row_offset..C_row_offset+1, col C_col
        
        def a_idx(i, j):
            return i * m + j
        
        def b_idx(i, j):
            return i * p + j
        
        def c_idx(i, j):
            return j * n + i
        
        # C[i, C_col] = sum_{j} A[i, A_col_offset+j] * B[B_row_offset+j, B_col]
        # For i in {0,1}, j in {0,1}
        for i in range(2):
            for j in range(2):
                U[a_idx(A_row_offset + i, A_col_offset + j)][start_col] = 1
                V[b_idx(B_row_offset + j, B_col)][start_col] = 1
                W[c_idx(C_row_offset + i, C_col)][start_col] = 1
                start_col += 1
        
        return start_col
    
    # A1 @ B1: A1 is rows 0-1, cols 0-1 of A; B1 is rows 0-1, cols 0-4 of B
    # Split B1 into cols 0-1, 2-3, 4
    col = add_strassen_2x2_2x2(0, 0, 0, 0, 0, 0, col)  # A1 @ B1[0:2, 0:2]
    col = add_strassen_2x2_2x2(0, 0, 0, 2, 0, 2, col)  # A1 @ B1[0:2, 2:4]
    col = add_schoolbook_2x2_2x1(0, 0, 0, 4, 0, 4, col)  # A1 @ B1[0:2, 4]
    
    # A2 @ B2: A2 is rows 0-1, cols 2-3 of A; B2 is rows 2-3, cols 0-4 of B
    # Split B2 into cols 0-1, 2-3, 4
    col = add_strassen_2x2_2x2(0, 2, 2, 0, 0, 0, col)  # A2 @ B2[2:4, 0:2]
    col = add_strassen_2x2_2x2(0, 2, 2, 2, 0, 2, col)  # A2 @ B2[2:4, 2:4]
    col = add_schoolbook_2x2_2x1(0, 2, 2, 4, 0, 4, col)  # A2 @ B2[2:4, 4]
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}, actual_rank


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try the Strassen-based decomposition first
    try:
        result, rank = _construct_rank36(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
