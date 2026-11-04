"""Strassen-based tensor decomposition for 4x4x6 matrix multiplication.

We build a rank-77 decomposition by combining:
  - a rank-49 decomposition of 4x4x4 (two Strassen 2x2x2 levels)
  - a rank-28 decomposition of 4x4x2 (one Strassen 2x2x2 level)
The output W uses the official trace convention W[k*n+i, r].
"""
import numpy as np

DIMENSIONS = [4, 4, 6]


def _strassen_222():
    """Return U, V, W (each shape (4,7)) for 2x2x2 matrix multiplication.
    Index convention: U[2*i+j, r], V[2*j+k, r], W[2*k+i, r].
    """
    # Strassen's 7 products for C = A @ B where A,B are 2x2.
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # U rows: i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
    # V rows: j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 2; j=1,k=1 -> 3
    # W rows: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # M1: (a00+a11)*(b00+b11)
    # U: rows 0 (a00), 3 (a11) -> +1
    # V: rows 0 (b00), 3 (b11) -> +1
    # W: C00=M1, C11=M1 -> W[0] (k=0,i=0)=+1, W[3] (k=1,i=1)=+1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)*b00
    # U: rows 2 (a10), 3 (a11) -> +1
    # V: row 0 (b00) -> +1
    # W: C10=M2 -> W[1] (k=0,i=1)=+1; C11=-M2 -> W[3] (k=1,i=1)=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a00*(b01-b11)
    # U: row 0 (a00) -> +1
    # V: row 1 (b01) -> +1, row 3 (b11) -> -1
    # W: C01=M3 -> W[2] (k=1,i=0)=+1; C11=M3 -> W[3] (k=1,i=1)=+1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: a11*(b10-b00)
    # U: row 3 (a11) -> +1
    # V: row 2 (b10) -> +1, row 0 (b00) -> -1
    # W: C00=M4 -> W[0] (k=0,i=0)=+1; C10=M4 -> W[1] (k=0,i=1)=+1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a00+a01)*b11
    # U: rows 0 (a00), 1 (a01) -> +1
    # V: row 3 (b11) -> +1
    # W: C00=-M5 -> W[0] (k=0,i=0)=-1; C01=M5 -> W[2] (k=1,i=0)=+1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a10-a00)*(b00+b10)
    # U: row 2 (a10) -> +1, row 0 (a00) -> -1
    # V: row 0 (b00) -> +1, row 2 (b10) -> +1
    # W: C11=M6 -> W[3] (k=1,i=1)=+1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01-a11)*(b10+b11)
    # U: row 1 (a01) -> +1, row 3 (a11) -> -1
    # V: row 2 (b10) -> +1, row 3 (b11) -> +1
    # W: C00=M7 -> W[0] (k=0,i=0)=+1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _embed_222_into_444(U2, V2, W2, i_off, j_off, k_off):
    """Embed a 2x2x2 decomposition into a 4x4x4 decomposition.
    The 2x2x2 block corresponds to A[i_off:i_off+2, j_off:j_off+2] @ B[j_off:j_off+2, k_off:k_off+2].
    """
    U = np.zeros((16, 7), dtype=np.int64)
    V = np.zeros((16, 7), dtype=np.int64)
    W = np.zeros((16, 7), dtype=np.int64)
    # U row index: i*4 + j, where i in 0..1 maps to i_off+i, j in 0..1 maps to j_off+j
    for r in range(7):
        for i in range(2):
            for j in range(2):
                idx_u = (i_off + i) * 4 + (j_off + j)
                idx_2 = i * 2 + j
                U[idx_u, r] = U2[idx_2, r]
        for j in range(2):
            for k in range(2):
                idx_v = (j_off + j) * 4 + (k_off + k)
                idx_2 = j * 2 + k
                V[idx_v, r] = V2[idx_2, r]
        for k in range(2):
            for i in range(2):
                idx_w = (k_off + k) * 4 + (i_off + i)
                idx_2 = k * 2 + i
                W[idx_w, r] = W2[idx_2, r]
    return U, V, W


def _decomp_444():
    """Build a rank-49 decomposition for 4x4x4 matrix multiplication.
    This uses Strassen at the 4x4 level by splitting into four 2x2 blocks.
    The 4x4 Strassen has 7 products, each of which is a 2x2x2 multiplication (rank 7).
    Total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_222()
    
    # 4x4 Strassen products (from the 2x2x2 formula applied to 2x2 blocks):
    # Let A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # Let B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B21)
    # M7 = (A12-A22) * (B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each product Mi, we need to build a 4x4x4 factor by summing over the 2x2x2 decompositions.
    # Each Mi is a product of (2x2 matrices) which itself can be decomposed with rank 7.
    # We'll build U, V, W of shape (16, 49) for the 4x4x4 case.
    
    rank_444 = 49
    U = np.zeros((16, rank_444), dtype=np.int64)
    V = np.zeros((16, rank_444), dtype=np.int64)
    W = np.zeros((16, rank_444), dtype=np.int64)
    
    # For each of the 7 Strassen products, we have 7 sub-products (from the 2x2x2 decomposition).
    # We'll index the columns as: product * 7 + subproduct
    
    # Define the 4x4 Strassen structure:
    # Product m (0..6), with coefficients for A blocks and B blocks.
    # A blocks: 11=(0,0), 12=(0,1), 21=(1,0), 22=(1,1)
    # B blocks: 11=(0,0), 12=(0,1), 21=(1,0), 22=(1,1)
    
    # For each product, define:
    #   a_coeffs: dict from (i_block, j_block) to coefficient for A
    #   b_coeffs: dict from (j_block, k_block) to coefficient for B
    #   w_coeffs: dict from (k_block, i_block) to coefficient for C
    
    strassen_444 = [
        # M1: (A11+A22)*(B11+B22)
        {'a': {(0,0):1, (1,1):1}, 'b': {(0,0):1, (1,1):1}, 'w': {(0,0):1, (1,1):1}},
        # M2: (A21+A22)*B11
        {'a': {(1,0):1, (1,1):1}, 'b': {(0,0):1}, 'w': {(0,1):1, (1,1):-1}},
        # M3: A11*(B12-B22)
        {'a': {(0,0):1}, 'b': {(0,1):1, (1,1):-1}, 'w': {(1,0):1, (1,1):1}},
        # M4: A22*(B21-B11)
        {'a': {(1,1):1}, 'b': {(1,0):1, (0,0):-1}, 'w': {(0,0):1, (0,1):1}},
        # M5: (A11+A12)*B22
        {'a': {(0,0):1, (0,1):1}, 'b': {(1,1):1}, 'w': {(0,0):-1, (1,0):1}},
        # M6: (A21-A11)*(B11+B21)
        {'a': {(1,0):1, (0,0):-1}, 'b': {(0,0):1, (1,0):1}, 'w': {(1,1):1}},
        # M7: (A12-A22)*(B21+B22)
        {'a': {(0,1):1, (1,1):-1}, 'b': {(1,0):1, (1,1):1}, 'w': {(0,0):1}},
    ]
    
    for m, prod in enumerate(strassen_444):
        col_base = m * 7
        for r in range(7):
            col = col_base + r
            # For this sub-product r of the 2x2x2 decomposition:
            # U2[row_u, r] gives the coefficient for the A element at (i_2, j_2)
            # V2[row_v, r] gives the coefficient for the B element at (j_2, k_2)
            # W2[row_w, r] gives the coefficient for the C element at (k_2, i_2)
            
            for (ib, jb), a_coeff in prod['a'].items():
                for (kb, ib2), w_coeff in prod['w'].items():
                    # We need to be careful: the W coefficient applies to the output block.
                    # For a given sub-product r, the contribution to U is from all A blocks with a_coeff != 0.
                    # The contribution to V is from all B blocks with b_coeff != 0.
                    # The contribution to W is from all C blocks with w_coeff != 0.
                    pass
            
            # Better approach: for each sub-product r, the U, V, W vectors are:
            # U[i*4+j, col] = sum over (ib,jb) of a_coeff[ib,jb] * U2[(i_ib)*2 + (j_jb), r]
            # where i_ib = i - ib*2, j_jb = j - jb*2, and we only sum when i_ib, j_jb in 0..1.
            
            for i in range(4):
                for j in range(4):
                    total = 0
                    for (ib, jb), a_coeff in prod['a'].items():
                        i_2 = i - ib * 2
                        j_2 = j - jb * 2
                        if 0 <= i_2 < 2 and 0 <= j_2 < 2:
                            total += a_coeff * U2[i_2 * 2 + j_2, r]
                    U[i * 4 + j, col] = total
            
            for j in range(4):
                for k in range(4):
                    total = 0
                    for (jb, kb), b_coeff in prod['b'].items():
                        j_2 = j - jb * 2
                        k_2 = k - kb * 2
                        if 0 <= j_2 < 2 and 0 <= k_2 < 2:
                            total += b_coeff * V2[j_2 * 2 + k_2, r]
                    V[j * 4 + k, col] = total
            
            for k in range(4):
                for i in range(4):
                    total = 0
                    for (kb, ib), w_coeff in prod['w'].items():
                        k_2 = k - kb * 2
                        i_2 = i - ib * 2
                        if 0 <= k_2 < 2 and 0 <= i_2 < 2:
                            total += w_coeff * W2[k_2 * 2 + i_2, r]
                    W[k * 4 + i, col] = total
    
    return U, V, W


def _decomp_442():
    """Build a rank-28 decomposition for 4x4x2 matrix multiplication.
    This uses Strassen by splitting into two 2x2x2 multiplications:
    C_top = A_top @ B_top (2x4 @ 4x2 = 2x2)
    C_bot = A_bot @ B_bot (2x4 @ 4x2 = 2x2)
    Each 2x2x2 has rank 7, total rank 14? No, that's not right.
    
    Actually, 4x4x2: A is 4x4, B is 4x2, C is 4x2.
    Split A into A_top (2x4) and A_bot (2x4).
    Split B into B_top (4x1) and B_bot (4x1)? No, B is 4x2.
    
    Let me reconsider. For 4x4x2:
    A is 4x4, B is 4x2.
    C = A @ B is 4x2.
    
    We can split this into two independent 2x2x2 multiplications?
    No, that doesn't work directly because B has 2 columns.
    
    Alternative: use Strassen on the 2x2 block structure.
    A = [[A11, A12], [A21, A22]] where each is 2x2.
    B = [[B1], [B2]] where B1, B2 are 2x2.
    C = [[C1], [C2]] where C1, C2 are 2x2.
    
    C1 = A11 @ B1 + A12 @ B2
    C2 = A21 @ B1 + A22 @ B2
    
    This requires 4 products of 2x2 matrices, each of rank 7.
    Total rank = 4 * 7 = 28.
    """
    U2, V2, W2 = _strassen_222()
    
    rank_442 = 28
    U = np.zeros((16, rank_442), dtype=np.int64)
    V = np.zeros((8, rank_442), dtype=np.int64)
    W = np.zeros((8, rank_442), dtype=np.int64)
    # Note: V has shape (m*p, rank) = (4*2, rank) = (8, rank)
    # W has shape (p*n, rank) = (2*4, rank) = (8, rank)
    
    # The 4 products are:
    # P1 = A11 @ B1
    # P2 = A12 @ B2
    # P3 = A21 @ B1
    # P4 = A22 @ B2
    # C1 = P1 + P2
    # C2 = P3 + P4
    
    products = [
        # (a_block, b_block, c_block, c_coeff)
        # a_block: (i_off, j_off), b_block: (j_off, k_off), c_block: (k_off, i_off)
        ((0, 0), (0, 0), (0, 0), 1),  # P1 -> C1
        ((0, 1), (1, 0), (0, 0), 1),  # P2 -> C1
        ((1, 0), (0, 0), (1, 0), 1),  # P3 -> C2
        ((1, 1), (1, 0), (1, 0), 1),  # P4 -> C2
    ]
    
    for p_idx, (a_block, b_block, c_block, c_coeff) in enumerate(products):
        col_base = p_idx * 7
        for r in range(7):
            col = col_base + r
            ai, aj = a_block
            bj, bk = b_block
            ck, ci = c_block
            
            for i in range(4):
                for j in range(4):
                    i_2 = i - ai * 2
                    j_2 = j - aj * 2
                    if 0 <= i_2 < 2 and 0 <= j_2 < 2:
                        U[i * 4 + j, col] = U2[i_2 * 2 + j_2, r]
            
            for j in range(4):
                for k in range(2):
                    j_2 = j - bj * 2
                    k_2 = k - bk * 2
                    if 0 <= j_2 < 2 and 0 <= k_2 < 2:
                        V[j * 2 + k, col] = V2[j_2 * 2 + k_2, r]
            
            for k in range(2):
                for i in range(4):
                    k_2 = k - ck * 2
                    i_2 = i - ci * 2
                    if 0 <= k_2 < 2 and 0 <= i_2 < 2:
                        W[k * 4 + i, col] = c_coeff * W2[k_2 * 2 + i_2, r]
    
    return U, V, W


def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 4x4x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Build 4x4x4 decomposition (rank 49)
    U_444, V_444, W_444 = _decomp_444()
    
    # Build 4x4x2 decomposition (rank 28)
    U_442, V_442, W_442 = _decomp_442()
    
    # Combine: total rank = 49 + 28 = 77
    total_rank = 49 + 28
    
    U = np.zeros((16, total_rank), dtype=np.int64)
    V = np.zeros((24, total_rank), dtype=np.int64)
    W = np.zeros((24, total_rank), dtype=np.int64)
    # U shape: (n*m, rank) = (16, 77)
    # V shape: (m*p, rank) = (4*6, 77) = (24, 77)
    # W shape: (p*n, rank) = (6*4, 77) = (24, 77)
    
    # 4x4x4 part uses columns 0..48
    # 4x4x2 part uses columns 49..76
    
    # For the 4x4x4 part:
    # V_444 has shape (16, 49) for m*p = 4*4 = 16
    # But we need V to have shape (24, 77) for m*p = 4*6 = 24
    # The 4x4x4 part only uses the first 4 columns of B (k in 0..3)
    # So V_444[j*4+k, col] for k in 0..3 maps to V[j*6+k, col] for k in 0..3
    
    for col in range(49):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, col] = U_444[i * 4 + j, col]
        for j in range(4):
            for k in range(4):
                V[j * 6 + k, col] = V_444[j * 4 + k, col]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, col] = W_444[k * 4 + i, col]
    
    # For the 4x4x2 part:
    # V_442 has shape (8, 28) for m*p = 4*2 = 8
    # But we need V to have shape (24, 77) for m*p = 4*6 = 24
    # The 4x4x2 part uses the last 2 columns of B (k in 4..5)
    # So V_442[j*2+k, col] for k in 0..1 maps to V[j*6+4+k, col] for k in 0..1
    
    for col in range(28):
        col_offset = 49 + col
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, col_offset] = U_442[i * 4 + j, col]
        for j in range(4):
            for k in range(2):
                V[j * 6 + 4 + k, col_offset] = V_442[j * 2 + k, col]
        for k in range(2):
            for i in range(4):
                W[(4 + k) * 4 + i, col_offset] = W_442[k * 4 + i, col]
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
