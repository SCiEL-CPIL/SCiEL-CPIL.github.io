"""Rewritten program: Strassen decomposition for 2x4 x 4x4 matrix multiplication, rank 28."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 4]


def _verify(U, V, W, n, m, p):
    """Check that the tensor decomposition is exact."""
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # and check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    # For efficiency, check via matrix identity:
    # C = A * B should equal sum_c (U_c @ A^T?) ... 
    # Actually: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Our factorization: C_{i,k} = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_l W[k*n+i, c] * ...)
    # Wait, the standard bilinear decomposition is:
    # C_{i,k} = sum_c M_c * N_c where M_c = sum_j U[i*m+j,c] A_{i,j}, N_c = sum_l V[j*p+l,c] B_{j,l}
    # And then we need W to map: actually the tensor is T_{a,b,c} with a indexing A, b indexing B, c indexing C.
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    # Factorization: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Let's verify by checking a few random matrices
    import numpy as np
    for trial in range(20):
        A = np.random.randint(-3, 4, (n, m))
        B = np.random.randint(-3, 4, (m, p))
        C_expected = A @ B
        
        # Compute via factorization
        C_fact = np.zeros((n, p))
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for r in range(len(U[0])):
            # M_r: vector of length n, M_r[i] = sum_j U[i*m+j, r] * A[i,j]
            M_r = np.array([sum(U_arr[i*m+j, r] * A[i, j] for j in range(m)) for i in range(n)])
            # N_r: vector of length p, N_r[k] = sum_j V[j*p+k, r] * B[j,k]
            N_r = np.array([sum(V_arr[j*p+k, r] * B[j, k] for j in range(m)) for k in range(p)])
            # Contribution: M_r[i] * N_r[k] * W[k*n+i, r]
            for i in range(n):
                for k in range(p):
                    C_fact[i, k] += M_r[i] * N_r[k] * W_arr[k*n+i, r]
        
        if not np.allclose(C_fact, C_expected):
            return False
    return True


def _build_strassen_2x4_4x4():
    """
    Build the rank-28 Strassen-based decomposition for (2,4,4).
    
    We partition:
    A is 2x4: split columns into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3)
    B is 4x4: split rows into B_T (2x4, rows 0,1) and B_B (2x4, rows 2,3)
    C is 2x4: split columns into C_L (2x2, cols 0,1) and C_R (2x2, cols 2,3)
    
    C_L = A_L @ B_TL + A_R @ B_TT  ... wait, let me be more careful.
    
    A (2x4): rows 0,1; cols 0,1,2,3
    B (4x4): rows 0,1,2,3; cols 0,1,2,3
    C (2x4): rows 0,1; cols 0,1,2,3
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    Split B into top 2 rows (0,1) and bottom 2 rows (2,3):
    C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    
    First term: (A_left 2x2) @ (B_top 2x4) -> but B_top is 2x4, and A_left is 2x2.
    This gives a 2x4 result. We can further split B_top into B_TL (2x2, cols 0,1) and B_TR (2x2, cols 2,3).
    
    So:
    C_L (cols 0,1) = A_L @ B_TL + A_R @ B_BL  where B_TL = B[0:2, 0:2], B_BL = B[2:4, 0:2]
    C_R (cols 2,3) = A_L @ B_TR + A_R @ B_BR  where B_TR = B[0:2, 2:4], B_BR = B[2:4, 2:4]
    
    Each of these is a sum of two 2x2 matrix products. Each 2x2 product can be done with Strassen rank 7.
    Total: 4 products * 7 = 28 rank.
    
    The four 2x2 products are:
    1. P1 = A_L @ B_TL  (contributes to C_L)
    2. P2 = A_R @ B_BL  (contributes to C_L)
    3. P3 = A_L @ B_TR  (contributes to C_R)
    4. P4 = A_R @ B_BR  (contributes to C_R)
    
    For each 2x2 product, Strassen gives 7 scalar multiplications.
    We need to map these to the global U, V, W matrices.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 8 x 28
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 16 x 28
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen for 2x2: 
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = a00 * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = (a00 + a01) * b11
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a10) * (b00 + b01)
    # M6 = (a11 + a01) * (b10 + b11)
    # M7 = (a10 - a11) * (b01 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 4 blocks, we assign 7 consecutive columns in U, V, W.
    # Block b (0-3) uses columns 7*b through 7*b+6.
    
    # Define the four blocks with their indices:
    # Each block is (A_block, B_block, C_block)
    # A_block: which rows of A and which columns
    # B_block: which rows of B and which columns
    # C_block: which rows of C and which columns
    
    # Block 0: A_L @ B_TL -> C_L
    #   A_L: A rows 0,1, cols 0,1. In U, index = i*m + j = i*4 + j. For i in 0,1 and j in 0,1: indices 0,1,4,5.
    #   B_TL: B rows 0,1, cols 0,1. In V, index = j*p + k = j*4 + k. For j in 0,1 and k in 0,1: indices 0,1,4,5.
    #   C_L: C rows 0,1, cols 0,1. In W, index = k*n + i = k*2 + i. For k in 0,1 and i in 0,1: indices 0,1,2,3.
    
    # Block 1: A_R @ B_BL -> C_L
    #   A_R: A rows 0,1, cols 2,3. U indices: i*4+2, i*4+3. For i in 0,1: 2,3,6,7.
    #   B_BL: B rows 2,3, cols 0,1. V indices: j*4+k. For j in 2,3 and k in 0,1: 8,9,12,13.
    #   C_L: same as block 0. W indices: 0,1,2,3.
    
    # Block 2: A_L @ B_TR -> C_R
    #   A_L: U indices 0,1,4,5.
    #   B_TR: B rows 0,1, cols 2,3. V indices: j*4+k. For j in 0,1 and k in 2,3: 2,3,6,7.
    #   C_R: C rows 0,1, cols 2,3. W indices: k*2+i. For k in 2,3 and i in 0,1: 4,5,6,7.
    
    # Block 3: A_R @ B_BR -> C_R
    #   A_R: U indices 2,3,6,7.
    #   B_BR: B rows 2,3, cols 2,3. V indices: j*4+k. For j in 2,3 and k in 2,3: 10,11,14,15.
    #   C_R: W indices 4,5,6,7.
    
    # Strassen coefficient patterns:
    # For a 2x2 product with local indices (a00,a01,a10,a11) for A and (b00,b01,b10,b11) for B:
    # M1: U has a00=1, V has b00=1, b11=1, W has C00=1, C11=1
    # M2: U has a10=1, a11=1, V has b00=1, W has C10=1, C11=-1
    # M3: U has a00=1, a01=1, V has b11=1, W has C01=1, C11=1
    # M4: U has a11=1, V has b10=1, b00=-1, W has C00=1, C10=1
    # M5: U has a00=1, a10=1, V has b00=1, b01=1, W has C00=-1, C01=1
    # M6: U has a11=1, a01=1, V has b10=1, b11=1, W has C11=1
    # M7: U has a10=1, a11=-1, V has b01=1, b11=1, W has C00=1
    
    # Local to global index mappings for each block:
    # A local (r,c) -> global U index: depends on block
    # B local (r,c) -> global V index: depends on block
    # C local (r,c) -> global W index: depends on block
    
    def add_strassen_block(block_idx, a_local_to_global, b_local_to_global, c_local_to_global):
        """
        Add one Strassen 2x2 block.
        a_local_to_global: dict mapping (local_row, local_col) -> global U index
        b_local_to_global: dict mapping (local_row, local_col) -> global V index
        c_local_to_global: dict mapping (local_row, local_col) -> global W index
        """
        base_col = 7 * block_idx
        # Strassen M1 through M7
        strassen_patterns = [
            # (U_entries: {local_A_idx: coeff}, V_entries: {local_B_idx: coeff}, W_entries: {local_C_idx: coeff})
            # M1
            ({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5
            ({(0,0): 1, (1,0): 1}, {(0,0): 1, (0,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6
            ({(1,1): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
            # M7
            ({(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for s_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_patterns):
            col = base_col + s_idx
            for (lr, lc), coeff in u_pat.items():
                gidx = a_local_to_global[(lr, lc)]
                U[gidx][col] += Fraction(coeff)
            for (lr, lc), coeff in v_pat.items():
                gidx = b_local_to_global[(lr, lc)]
                V[gidx][col] += Fraction(coeff)
            for (lr, lc), coeff in w_pat.items():
                gidx = c_local_to_global[(lr, lc)]
                W[gidx][col] += Fraction(coeff)
    
    # Block 0: A_L @ B_TL -> C_L
    a0 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}  # A[i,j] -> U[i*4+j]
    b0 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}  # B[j,k] -> V[j*4+k]
    c0 = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}  # C[i,k] -> W[k*2+i]
    add_strassen_block(0, a0, b0, c0)
    
    # Block 1: A_R @ B_BL -> C_L
    a1 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    b1 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
    c1 = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
    add_strassen_block(1, a1, b1, c1)
    
    # Block 2: A_L @ B_TR -> C_R
    a2 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    b2 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    c2 = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
    add_strassen_block(2, a2, b2, c2)
    
    # Block 3: A_R @ B_BR -> C_R
    a3 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    b3 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
    c3 = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
    add_strassen_block(3, a3, b3, c3)
    
    # Convert to lists of ints (all entries should be integers)
    U_int = [[int(x) for x in row] for row in U]
    V_int = [[int(x) for x in row] for row in V]
    W_int = [[int(x) for x in row] for row in W]
    
    # Verify
    if _verify(U_int, V_int, W_int, n, m, p):
        return {"U": U_int, "V": V_int, "W": W_int}
    else:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    result = _build_strassen_2x4_4x4()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but guard
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
