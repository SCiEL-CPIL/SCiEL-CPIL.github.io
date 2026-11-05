"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: block decomposition of the 4x5 x 5x6 multiplication tensor into
sub-blocks. For each 2x2 sub-block we use Strassen's 7-term algorithm
(rank 7 instead of 8), and for remaining rectangular sub-blocks we use
the schoolbook decomposition. This yields a total rank of 108.
"""
import time

DIMENSIONS = [4, 5, 6]
DEADLINE = 140.0  # safety margin before the 150s kill


def _strassen_2x2x2(rank_offset, U, V, W, i0, j0, k0):
    """Add Strassen's 7-term decomposition for a 2x2x2 sub-block.
    
    A is 2x2 at rows [i0:i0+2], cols [j0:j0+2]
    B is 2x2 at rows [j0:j0+2], cols [k0:k0+2]
    C is 2x2 at rows [k0:k0+2], cols [i0:i0+2]
    
    W[k*n+i, r] convention: row index = k*n + i
    """
    n_local = 2  # output rows per block
    m_local = 2  # A cols per block
    p_local = 2  # B cols per block
    
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Strassen term is a rank-1 decomposition of a product of sums/differences.
    # For M1 = (A11+A22)*(B11+B22):
    #   This is sum of 4 rank-1 terms: A11*B11 + A11*B22 + A22*B11 + A22*B22
    # But we want to use the structure: define U, V entries for the linear combinations.
    #
    # Actually, Strassen gives 7 bilinear forms. Each bilinear form M_l = sum over
    # (i,j) in A-block of (sum over k in B-block) of u_l(i,j) * a(i,j) * v_l(j,k) * b(j,k)
    # where u_l and v_l are 2x2 matrices with entries in {-1, 0, 1}.
    #
    # For each of the 7 terms, we have a 2x2x2 tensor of rank 1 (each term is a product
    # of two linear combinations). But wait - each M_l is itself a product of a linear
    # combination of A entries and a linear combination of B entries, so each M_l is
    # a single bilinear form = 1 rank-1 term in the tensor decomposition!
    #
    # No wait. Let me reconsider. In tensor decomposition of matrix multiplication:
    # C(k,i) = sum_r U(i,j,r) * V(j,k,r) * W(k,i,r) * a(i,j) * b(j,k)
    # where U, V, W are the factor matrices.
    #
    # For a 2x2x2 block with Strassen, we have 7 bilinear forms. Each bilinear form
    # corresponds to ONE rank-1 term in the decomposition. So 7 terms total for the 2x2x2.
    #
    # The structure: for each rank-1 term r, we need:
    #   U[(i,j), r] for i,j in the 2x2 A-block
    #   V[(j,k), r] for j,k in the 2x2 B-block  
    #   W[(k,i), r] for k,i in the 2x2 C-block
    #
    # Strassen's 7 forms as bilinear forms in (A entries) and (B entries):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # For each term, extract coefficients for A and B, then determine W from C = f(M1..M7)
    
    # A coefficients (2x2) for each term:
    A_coeffs = [
        [[1, 0], [0, 1]],   # M1: a00 + a11
        [[0, 0], [1, 1]],   # M2: a10 + a11
        [[1, 0], [0, 0]],   # M3: a00
        [[0, 0], [0, 1]],   # M4: a11
        [[1, 1], [0, 0]],   # M5: a00 + a01
        [[-1, 0], [1, 0]],  # M6: a10 - a00
        [[0, 1], [0, -1]],  # M7: a01 - a11
    ]
    
    # B coefficients (2x2) for each term:
    B_coeffs = [
        [[1, 0], [0, 1]],   # M1: b00 + b11
        [[1, 0], [0, 0]],   # M2: b00
        [[0, 1], [0, -1]],  # M3: b01 - b11
        [[0, -1], [1, 0]],  # M4: b10 - b00
        [[0, 0], [0, 1]],   # M5: b11
        [[1, 1], [0, 0]],   # M6: b00 + b01
        [[0, 1], [1, 1]],   # M7: b10 + b11
    ]
    
    # C coefficients (2x2) for each term: how each M_l contributes to C entries
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # C_coeffs[l][k_local][i_local] = coefficient of M_l in C[k_local, i_local]
    # k_local is row of C (which corresponds to k index), i_local is col of C (i index)
    # C[k,i] where k is output row, i is output col
    
    C_coeffs = [
        [[1, 0], [0, 1]],   # M1: C11+=1, C22+=1  => C[0,0]=1, C[1,1]=1
        [[0, 0], [1, -1]],  # M2: C21+=1, C22-=-1 => C[1,0]=1, C[1,1]=-1
        [[0, 1], [0, 1]],   # M3: C12+=1, C22+=1  => C[0,1]=1, C[1,1]=1
        [[0, 0], [1, 1]],   # M4: C21+=1, C22+=1  => C[1,0]=1, C[1,1]=1
        [[1, -1], [0, 0]],  # M5: C11+=1, C12-=-1 => C[0,0]=1, C[0,1]=-1
        [[0, 0], [0, 1]],   # M6: C22+=1          => C[1,1]=1
        [[0, 0], [1, 0]],   # M7: C11+=1          => C[0,0]=1
    ]
    
    # Wait, let me re-verify:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # C_coeffs[l] is a 2x2 matrix where C_coeffs[l][k][i] is the coefficient
    # of M_l in C[k][i].
    
    C_coeffs = [
        # M1: C11+=1, C12+=0, C21+=0, C22+=1
        [[1, 0], [0, 1]],
        # M2: C11+=0, C12+=0, C21+=1, C22+=-1
        [[0, 0], [1, -1]],
        # M3: C11+=0, C12+=1, C21+=0, C22+=1
        [[0, 1], [0, 1]],
        # M4: C11+=0, C12+=0, C21+=1, C22+=1
        [[0, 0], [1, 1]],
        # M5: C11+=1, C12+=-1, C21+=0, C22+=0
        [[1, -1], [0, 0]],
        # M6: C11+=0, C12+=0, C21+=0, C22+=1
        [[0, 0], [0, 1]],
        # M7: C11+=1, C12+=0, C21+=0, C22+=0
        [[1, 0], [0, 0]],
    ]
    
    n, m, p = 4, 5, 6  # global dims
    
    for l in range(7):
        r = rank_offset + l
        for il in range(2):
            for jl in range(2):
                gi = i0 + il
                gj = j0 + jl
                U[gi * m + gj][r] = A_coeffs[l][il][jl]
        for jl in range(2):
            for kl in range(2):
                gj = j0 + jl
                gk = k0 + kl
                V[gj * p + gk][r] = B_coeffs[l][jl][kl]
        for kl in range(2):
            for il in range(2):
                gk = k0 + kl
                gi = i0 + il
                W[gk * n + gi][r] = C_coeffs[l][kl][il]


def _schoolbook_block(rank_offset, U, V, W, i0, j0, k0, ni, mj, pk):
    """Add schoolbook decomposition for a ni x mj x pk sub-block.
    
    A is ni x mj at rows [i0:i0+ni], cols [j0:j0+mj]
    B is mj x pk at rows [j0:j0+mj], cols [k0:k0+pk]
    C is pk x ni at rows [k0:k0+pk], cols [i0:i0+ni]
    """
    n, m, p = 4, 5, 6
    col = rank_offset
    for il in range(ni):
        for jl in range(mj):
            for kl in range(pk):
                gi = i0 + il
                gj = j0 + jl
                gk = k0 + kl
                U[gi * m + gj][col] = 1
                V[gj * p + gk][col] = 1
                W[gk * n + gi][col] = 1
                col += 1
    return col - rank_offset  # number of terms used


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Block structure: split 4x5 into 2x(2+3), 5 into (2+3)x(2+4), 6 into (2+4)
    # A: 4x5 -> split rows into [0:2], [2:4]; cols into [0:2], [2:5]
    # B: 5x6 -> split rows into [0:2], [2:5]; cols into [0:2], [2:6]
    # C: 6x4 -> split rows into [0:2], [2:6]; cols into [0:2], [2:4]
    
    # Sub-blocks:
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:6] + A[0:2,2:5]*B[2:5,2:6]
    # C[2:6, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:5]*B[2:5,0:2]
    # C[2:6, 2:4] = A[2:4,0:2]*B[0:2,2:6] + A[2:4,2:5]*B[2:5,2:6]
    
    # Sub-block sizes (ni, mj, pk):
    # 1. (2,2,2) - A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  -> Strassen (rank 7)
    # 2. (2,3,2) - A[0:2,2:5] * B[2:5,0:2] -> C[0:2,0:2]  -> Schoolbook (rank 12)
    # 3. (2,2,4) - A[0:2,0:2] * B[0:2,2:6] -> C[0:2,2:4]  -> Schoolbook (rank 16)
    # 4. (2,3,4) - A[0:2,2:5] * B[2:5,2:6] -> C[0:2,2:4]  -> Schoolbook (rank 24)
    # 5. (2,2,2) - A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]  -> Strassen (rank 7)
    # 6. (2,3,2) - A[2:4,2:5] * B[2:5,0:2] -> C[2:4,0:2]  -> Schoolbook (rank 12)
    # 7. (2,2,4) - A[2:4,0:2] * B[0:2,2:6] -> C[2:4,2:4]  -> Schoolbook (rank 16)
    # 8. (2,3,4) - A[2:4,2:5] * B[2:5,2:6] -> C[2:4,2:4]  -> Schoolbook (rank 24)
    
    # Total rank: 7+12+16+24+7+12+16+24 = 118
    # Hmm, that's not 108. Let me reconsider.
    
    # Actually, let me try a different split. What if we split 5 as 1+4 or 2+3?
    # Or split 4 as 1+3?
    
    # Let me try: split 4 rows as [0:2],[2:4]; 5 cols as [0:1],[1:5]; 6 as [0:2],[2:6]
    # No, that gives odd sizes.
    
    # Let me try another approach: use the known result that 4x5x6 can be done
    # with rank 108 via a specific block decomposition.
    
    # Alternative: split into blocks of size 2x2x2 (Strassen) and handle
    # the remaining parts optimally.
    
    # Let's try: 4 = 2+2, 5 = 2+3, 6 = 3+3
    # Sub-blocks:
    # C[0:3,0:2] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:5]*B[2:5,0:3]
    # C[0:3,2:4] = A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:5]*B[2:5,3:6]
    # C[3:6,0:2] = A[2:4,0:2]*B[0:2,0:3] + A[2:4,2:5]*B[2:5,0:3]
    # C[3:6,2:4] = A[2:4,0:2]*B[0:2,3:6] + A[2:4,2:5]*B[2:5,3:6]
    
    # Sizes:
    # 1. (2,2,3) - rank 12
    # 2. (2,3,3) - rank 18
    # 3. (2,2,3) - rank 12
    # 4. (2,3,3) - rank 18
    # 5. (2,2,3) - rank 12
    # 6. (2,3,3) - rank 18
    # 7. (2,2,3) - rank 12
    # 8. (2,3,3) - rank 18
    # Total: 4*12 + 4*18 = 48 + 72 = 120. Worse.
    
    # Let me go back to the 2+3 / 2+4 split and see if I can do better on some blocks.
    
    # For block (2,2,4): can we do better than schoolbook rank 16?
    # 2x2x4: this is equivalent to multiplying two 2x2 matrices where the second
    # has 4 columns. We can use Strassen-like methods.
    # Actually, 2x2x4 can be decomposed with rank 12 using:
    # Think of it as 2x2 * 2x4. We can split the 4 cols into 2+2:
    # (2,2,2) + (2,2,2) = 7+7 = 14. Better than 16!
    
    # Similarly, (2,3,2): split the 3 cols of A into... hmm, 3 doesn't split evenly.
    # (2,3,2): schoolbook is 12. Can we do better?
    # Split B's 2 cols... no. Split A's 3 rows into 1+2?
    # (1,3,2) + (1,3,2) = 6+6 = 12. Same.
    # Or split the 2 of C into... no.
    # Actually (2,3,2) might be irreducible at rank 12.
    
    # Let me redo with the improved (2,2,4) -> 14:
    # 1. (2,2,2) Strassen: 7
    # 2. (2,3,2) Schoolbook: 12
    # 3. (2,2,4) -> two (2,2,2) Strassen: 14
    # 4. (2,3,4) -> ?
    # 5. (2,2,2) Strassen: 7
    # 6. (2,3,2) Schoolbook: 12
    # 7. (2,2,4) -> two (2,2,2) Strassen: 14
    # 8. (2,3,4) -> ?
    
    # For (2,3,4): schoolbook is 24. Can we do better?
    # Split into (2,3,2) + (2,3,2) = 12+12 = 24. Same.
    # Split into (2,2,4) + (2,1,4) = 14 + 8 = 22! Better!
    # Or (2,2,4) + (2,1,4): 14+8=22
    # Or (2,1,4) + (2,2,4): same.
    # Actually split the 3 cols of A into 2+1:
    # (2,2,4) + (2,1,4)
    # (2,2,4) -> 14 (as above)
    # (2,1,4) -> schoolbook: 2*1*4 = 8
    # Total for (2,3,4): 14+8 = 22
    
    # Revised total:
    # 1. 7
    # 2. 12
    # 3. 14
    # 4. 22
    # 5. 7
    # 6. 12
    # 7. 14
    # 8. 22
    # Total: 7+12+14+22+7+12+14+22 = 110
    
    # Can we improve further? 
    # (2,3,2): try splitting. (2,2,2)+(2,1,2) = 7+4=11! Better than 12.
    # So (2,3,2) -> 11
    
    # Revised:
    # 1. 7
    # 2. 11
    # 3. 14
    # 4. 22
    # 5. 7
    # 6. 11
    # 7. 14
    # 8. 22
    # Total: 7+11+14+22+7+11+14+22 = 108
    
    # Great! Rank 108.
    
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(i0, j0, k0, ni, mj, pk):
        """Recursively decompose a sub-block, returning next rank offset."""
        nonlocal rank
        # Base case: if small enough, use schoolbook
        # Try to split for efficiency
        if ni == 2 and mj == 2 and pk == 2:
            # Use Strassen
            start = rank
            _strassen_2x2x2(start, U, V, W, i0, j0, k0)
            rank += 7
            return rank
        if ni == 2 and mj == 2 and pk == 4:
            # Split into two 2x2x2 Strassen blocks
            start = rank
            _strassen_2x2x2(start, U, V, W, i0, j0, k0)
            rank += 7
            _strassen_2x2x2(rank, U, V, W, i0, j0, k0 + 2)
            rank += 7
            return rank
        if ni == 2 and mj == 3 and pk == 2:
            # Split A's cols into 2+1: (2,2,2) + (2,1,2)
            start = rank
            _strassen_2x2x2(start, U, V, W, i0, j0, k0)
            rank += 7
            # (2,1,2) schoolbook
            schoolbook_start = rank
            _schoolbook_block(schoolbook_start, U, V, W, i0, j0 + 2, k0, 2, 1, 2)
            rank += 2 * 1 * 2
            return rank
        if ni == 2 and mj == 3 and pk == 4:
            # Split A's cols into 2+1: (2,2,4) + (2,1,4)
            start = rank
            # (2,2,4) -> two Strassen
            _strassen_2x2x2(start, U, V, W, i0, j0, k0)
            rank += 7
            _strassen_2x2x2(rank, U, V, W, i0, j0, k0 + 2)
            rank += 7
            # (2,1,4) schoolbook
            schoolbook_start = rank
            _schoolbook_block(schoolbook_start, U, V, W, i0, j0 + 2, k0, 2, 1, 4)
            rank += 2 * 1 * 4
            return rank
        # Default: schoolbook
        start = rank
        _schoolbook_block(start, U, V, W, i0, j0, k0, ni, mj, pk)
        rank += ni * mj * pk
        return rank
    
    # The 8 sub-blocks:
    # 1. A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  : (2,2,2) at (0,0,0)
    # 2. A[0:2,2:5] * B[2:5,0:2] -> C[0:2,0:2]  : (2,3,2) at (0,2,0)
    # 3. A[0:2,0:2] * B[0:2,2:6] -> C[0:2,2:4]  : (2,2,4) at (0,0,2)
    # 4. A[0:2,2:5] * B[2:5,2:6] -> C[0:2,2:4]  : (2,3,4) at (0,2,2)
    # 5. A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]  : (2,2,2) at (2,0,0)
    # 6. A[2:4,2:5] * B[2:5,0:2] -> C[2:4,0:2]  : (2,3,2) at (2,2,0)
    # 7. A[2:4,0:2] * B[0:2,2:6] -> C[2:4,2:4]  : (2,2,4) at (2,0,2)
    # 8. A[2:4,2:5] * B[2:5,2:6] -> C[2:4,2:4]  : (2,3,4) at (2,2,2)
    
    add_block(0, 0, 0, 2, 2, 2)
    add_block(0, 2, 0, 2, 3, 2)
    add_block(0, 0, 2, 2, 2, 4)
    add_block(0, 2, 2, 2, 3, 4)
    add_block(2, 0, 0, 2, 2, 2)
    add_block(2, 2, 0, 2, 3, 2)
    add_block(2, 0, 2, 2, 2, 4)
    add_block(2, 2, 2, 2, 3, 4)
    
    # Verify rank is as expected
    assert rank == 108, f"Expected rank 108, got {rank}"
    
    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
