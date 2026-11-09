"""Matrix multiplication 4x4 by 4x5 rank-minimal exact decomposition."""
DIMENSIONS = [4, 4, 5]

import time
import random
import itertools
from fractions import Fraction

import numpy as np


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """
    Generate Strassen decomposition for 2x2x2.
    Rank 7.
    
    The 7 products are:
    M1 = (A00+A11)(B01+B11)
    M2 = (A10+A11)(B00)
    M3 = (A00)(B10-B00)
    M4 = (A11)(B01-B10)
    M5 = (A00+A01)(B11)
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    Output:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]  # 2x2 = 4 entries
    V = [[0] * rank for _ in range(4)]  # 2x2 = 4 entries
    W = [[0] * rank for _ in range(4)]  # 2x2 = 4 entries
    
    # M1 = (A00+A11)(B01+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[1][0] = 1  # B01
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10+A11)(B00)
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = (A00)(B10-B00)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[0][2] = -1 # B00
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = (A11)(B01-B10)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[2][3] = -1 # B10
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00+A01)(B11)
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    W[3][6] = 1  # C11
    
    return U, V, W


def _make_4x4x4_strassen():
    """
    Generate Strassen decomposition for 4x4x4 using recursive 2x2 blocking.
    Rank 49 = 7^2.
    
    We divide the 4x4 matrices into 2x2 blocks of 2x2 submatrices.
    The 4x4x4 multiplication can be seen as 4 separate 2x2x2 multiplications
    (for each pair of block rows/cols), but with sharing via Strassen.
    
    Actually, the correct approach: treat 4x4 as 2x2 blocks where each block is 2x2.
    The 4x4x4 tensor decomposes into 4 independent 2x2x2 tensors? No.
    
    Let me think again. A 4x4 matrix A has 16 entries. We can view it as a 2x2 
    matrix of 2x2 blocks. Same for B. The product C is also 2x2 blocks of 2x2.
    
    C_block[i][j] = sum_k A_block[i][k] * B_block[k][j]
    
    Each block multiplication is 2x2 * 2x2 = 2x2, which can be done with Strassen (rank 7).
    But we need to be careful about how the entries map.
    
    Actually, the standard recursive Strassen for 4x4: 
    - Split 4x4 into four 2x2 quadrants
    - Apply Strassen at the block level (7 block multiplications)
    - Each block multiplication is itself a 2x2x2 problem, which can be done with rank 7
    - Total: 7 * 7 = 49
    
    Let me implement this properly.
    """
    n = 4
    rank = 49
    U = [[0] * rank for _ in range(n * n)]  # 16 entries
    V = [[0] * rank for _ in range(n * n)]  # 16 entries
    W = [[0] * rank for _ in range(n * n)]  # 16 entries
    
    # Get the 2x2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2()
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    
    # Strassen at block level:
    # M1 = (A00+A11)(B01+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block-level Strassen product, we need to multiply two 2x2 matrices.
    # Each such multiplication uses 7 rank-1 terms.
    
    # Let's define the block indices:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3),
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    def add_block_product(block_A, block_B, block_C, sign, base_rank):
        """
        Add a block-level product: (sum of A blocks) * (sum of B blocks) -> C block
        Each block is 2x2, so we use the 2x2x2 Strassen decomposition.
        
        block_A: list of (block_idx, coefficient) for A
        block_B: list of (block_idx, coefficient) for B  
        block_C: (block_idx, coefficient) for C
        sign: overall sign
        base_rank: starting rank index
        """
        # For each of the 7 Strassen terms in 2x2x2:
        for t in range(7):
            r = base_rank + t
            
            # U contributions from A blocks
            for a_idx, a_coeff in block_A:
                # a_idx is 0=A00, 1=A01, 2=A10, 3=A11
                row_block = a_idx // 2
                col_block = a_idx % 2
                for dr in range(2):
                    for dc in range(2):
                        global_row = row_block * 2 + dr
                        global_col = col_block * 2 + dc
                        local_idx = dr * 2 + dc  # index in 2x2
                        U[global_row * 4 + global_col][r] += sign * a_coeff * U2[local_idx][t]
            
            # V contributions from B blocks
            for b_idx, b_coeff in block_B:
                row_block = b_idx // 2
                col_block = b_idx % 2
                for dr in range(2):
                    for dc in range(2):
                        global_row = row_block * 2 + dr
                        global_col = col_block * 2 + dc
                        local_idx = dr * 2 + dc
                        V[global_row * 4 + global_col][r] += b_coeff * V2[local_idx][t]
            
            # W contributions to C block
            c_idx, c_coeff = block_C
            row_block = c_idx // 2
            col_block = c_idx % 2
            for dr in range(2):
                for dc in range(2):
                    global_row = row_block * 2 + dr
                    global_col = col_block * 2 + dc
                    local_idx = dr * 2 + dc
                    W[global_row * 4 + global_col][r] += sign * c_coeff * W2[local_idx][t]
        
        return base_rank + 7
    
    # M1 = (A00+A11)(B01+B11)
    # A blocks: A00 (idx 0) with coeff 1, A11 (idx 3) with coeff 1
    # B blocks: B01 (idx 1) with coeff 1, B11 (idx 3) with coeff 1
    # C: C00 (idx 0) with coeff 1, C11 (idx 3) with coeff 1
    r = 0
    r = add_block_product([(0, 1), (3, 1)], [(1, 1), (3, 1)], (0, 1), 1, r)
    r = add_block_product([(0, 1), (3, 1)], [(1, 1), (3, 1)], (3, 1), 1, r)
    
    # M2 = (A10+A11)B00
    # A blocks: A10 (idx 2) with coeff 1, A11 (idx 3) with coeff 1
    # B blocks: B00 (idx 0) with coeff 1
    # C: C10 (idx 2) with coeff 1, C11 (idx 3) with coeff -1
    r = add_block_product([(2, 1), (3, 1)], [(0, 1)], (2, 1), 1, r)
    r = add_block_product([(2, 1), (3, 1)], [(0, 1)], (3, -1), 1, r)
    
    # M3 = A00(B10-B00)
    # A blocks: A00 (idx 0) with coeff 1
    # B blocks: B10 (idx 2) with coeff 1, B00 (idx 0) with coeff -1
    # C: C01 (idx 1) with coeff 1, C11 (idx 3) with coeff 1
    r = add_block_product([(0, 1)], [(2, 1), (0, -1)], (1, 1), 1, r)
    r = add_block_product([(0, 1)], [(2, 1), (0, -1)], (3, 1), 1, r)
    
    # M4 = A11(B01-B10)
    # A blocks: A11 (idx 3) with coeff 1
    # B blocks: B01 (idx 1) with coeff 1, B10 (idx 2) with coeff -1
    # C: C00 (idx 0) with coeff 1, C10 (idx 2) with coeff 1
    r = add_block_product([(3, 1)], [(1, 1), (2, -1)], (0, 1), 1, r)
    r = add_block_product([(3, 1)], [(1, 1), (2, -1)], (2, 1), 1, r)
    
    # M5 = (A00+A01)B11
    # A blocks: A00 (idx 0) with coeff 1, A01 (idx 1) with coeff 1
    # B blocks: B11 (idx 3) with coeff 1
    # C: C00 (idx 0) with coeff -1, C01 (idx 1) with coeff 1
    r = add_block_product([(0, 1), (1, 1)], [(3, 1)], (0, -1), 1, r)
    r = add_block_product([(0, 1), (1, 1)], [(3, 1)], (1, 1), 1, r)
    
    # M6 = (A10-A00)(B00+B01)
    # A blocks: A10 (idx 2) with coeff 1, A00 (idx 0) with coeff -1
    # B blocks: B00 (idx 0) with coeff 1, B01 (idx 1) with coeff 1
    # C: C11 (idx 3) with coeff 1
    r = add_block_product([(2, 1), (0, -1)], [(0, 1), (1, 1)], (3, 1), 1, r)
    
    # M7 = (A01-A11)(B10+B11)
    # A blocks: A01 (idx 1) with coeff 1, A11 (idx 3) with coeff -1
    # B blocks: B10 (idx 2) with coeff 1, B11 (idx 3) with coeff 1
    # C: C00 (idx 0) with coeff 1, C11 (idx 3) with coeff 1
    r = add_block_product([(1, 1), (3, -1)], [(2, 1), (3, 1)], (0, 1), 1, r)
    r = add_block_product([(1, 1), (3, -1)], [(2, 1), (3, 1)], (3, 1), 1, r)
    
    return U, V, W, r


def _make_4x4x5_strassen():
    """
    Generate decomposition for 4x4 * 4x5.
    
    Strategy: 
    - Use 4x4x4 Strassen (rank 49) for the first 4 columns
    - Add schoolbook (rank 16) for the 5th column
    - Total rank: 65
    
    But we can potentially do better by finding linear dependencies.
    For now, let's start with this and verify.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Get 4x4x4 Strassen
    U444, V444, W444, rank_444 = _make_4x4x4_strassen()
    
    # Add schoolbook for the 5th column
    # The 5th column of B corresponds to k=4
    # We need: C[i][4] = sum_j A[i][j] * B[j][4] for i=0..3
    # This is 4*1 = 4 rank-1 terms (one for each i,j pair... wait, 4*4=16 terms)
    
    # Actually, for the 5th column:
    # C[i][4] = sum_{j=0}^{3} A[i][j] * B[j][4]
    # Each term A[i][j]*B[j][4] contributes to C[i][4]
    # So we need 16 rank-1 terms for the 5th column
    
    total_rank = rank_444 + 16
    U = [[0] * total_rank for _ in range(n * m)]  # 16 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 20 rows
    
    # Copy 4x4x4 Strassen (for columns 0-3 of output)
    for i in range(16):
        for r in range(rank_444):
            U[i][r] = U444[i][r]
    
    for i in range(16):  # First 16 entries of B (columns 0-3)
        for r in range(rank_444):
            V[i][r] = V444[i][r]
    
    for i in range(16):  # First 16 entries of C (columns 0-3)
        for r in range(rank_444):
            W[i][r] = W444[i][r]
    
    # Add schoolbook for 5th column (k=4)
    # B[j][4] is at index j*5 + 4 in V (since p=5, row j, col 4)
    # C[i][4] is at index 4*4 + i = 16 + i in W (since p=5, row 4, col i)
    
    c = rank_444
    for i in range(n):  # 0..3
        for j in range(m):  # 0..3
            U[i * m + j][c] = 1  # A[i][j]
            V[j * p + 4][c] = 1  # B[j][4]
            W[4 * n + i][c] = 1  # C[i][4] -> index 4*4+i = 16+i
            c += 1
    
    return {"U": U, "V": V, "W": W}


def _verify(decomp, verbose=False):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    rank = U.shape[1]
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using decomposition
    # C[i,k] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] ... wait, let me check the convention
    
    # The convention is: W[k*n+i, r]
    # So C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                # Sum over j
                s = 0.0
                for j in range(m):
                    s += U[i * m + j, r] * V[j * p + k, r]
                C_computed[i, k] += s * W[k * n + i, r]
    
    max_err = np.max(np.abs(C_computed - C_expected))
    if verbose:
        print(f"Verification: max error = {max_err:.2e}")
        if max_err > 1e-9:
            print("MISMATCH!")
            print("Expected:", C_expected)
            print("Computed:", C_computed)
    
    return max_err < 1e-9


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition
    try:
        decomp = _make_4x4x5_strassen()
        if _verify(decomp, verbose=False):
            return decomp
    except Exception as e:
        print(f"Strassen failed: {e}")
    
    # Fallback to schoolbook
    return _schoolbook()
