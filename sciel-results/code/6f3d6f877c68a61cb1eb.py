import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _strassen_decomp(k):
    """
    Generate the Strassen decomposition for [2^k, 2^k, 2^k] matrix multiplication.
    Returns a list of tuples (u_indices, v_indices, w_indices, coeff)
    where u_indices, v_indices, w_indices are lists of (row, col) tuples
    representing the non-zero positions in the rank-1 tensor factors.
    """
    if k == 0:
        return [([(0, 0)], [(0, 0)], [(0, 0)], 1)]
    
    sub_decomp = _strassen_decomp(k - 1)
    size = 2 ** (k - 1)
    
    # Define block mapping functions
    # A blocks: A11, A12, A21, A22
    # B blocks: B11, B12, B21, B22
    # C blocks: C11, C12, C21, C22
    
    def map_indices(indices, block_type, block_id):
        """
        Map sub-matrix indices to full matrix indices based on block.
        block_type: 'A', 'B', or 'C'
        block_id: 11, 12, 21, or 22
        """
        if block_id == 11:
            return [(i, j) for (i, j) in indices]
        elif block_id == 12:
            return [(i, j + size) for (i, j) in indices]
        elif block_id == 21:
            return [(i + size, j) for (i, j) in indices]
        elif block_id == 22:
            return [(i + size, j + size) for (i, j) in indices]
    
    terms = []
    
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M, we need to generate terms by expanding the linear combinations.
    # Each term in sub_decomp is a rank-1 tensor for a [size, size, size] multiplication.
    # We need to "lift" each sub-term to the appropriate blocks in the larger matrix.
    
    # Helper to add terms for a product of linear combinations
    def add_product_terms(a_blocks, b_blocks, c_blocks, coeff):
        """
        a_blocks: list of (block_id, coeff) for A
        b_blocks: list of (block_id, coeff) for B
        c_blocks: list of (block_id, coeff) for C
        """
        for (a_id, a_coeff) in a_blocks:
            for (b_id, b_coeff) in b_blocks:
                for (c_id, c_coeff) in c_blocks:
                    total_coeff = coeff * a_coeff * b_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    
                    for (u_sub, v_sub, w_sub, sub_coeff) in sub_decomp:
                        u_idx = map_indices(u_sub, 'A', a_id)
                        v_idx = map_indices(v_sub, 'B', b_id)
                        w_idx = map_indices(w_sub, 'C', c_id)
                        terms.append((u_idx, v_idx, w_idx, total_coeff * sub_coeff))
    
    # M1 = (A11 + A22) * (B11 + B22)
    # Contributes to C11 (+1), C22 (+1)
    add_product_terms([(11, 1), (22, 1)], [(11, 1), (22, 1)], [(11, 1)], 1)
    add_product_terms([(11, 1), (22, 1)], [(11, 1), (22, 1)], [(22, 1)], 1)
    
    # M2 = (A21 + A22) * B11
    # Contributes to C21 (+1), C22 (-1)
    add_product_terms([(21, 1), (22, 1)], [(11, 1)], [(21, 1)], 1)
    add_product_terms([(21, 1), (22, 1)], [(11, 1)], [(22, -1)], 1)
    
    # M3 = A11 * (B12 - B22)
    # Contributes to C12 (+1), C22 (+1)
    add_product_terms([(11, 1)], [(12, 1), (22, -1)], [(12, 1)], 1)
    add_product_terms([(11, 1)], [(12, 1), (22, -1)], [(22, 1)], 1)
    
    # M4 = A22 * (B21 - B11)
    # Contributes to C11 (+1), C21 (+1)
    add_product_terms([(22, 1)], [(21, 1), (11, -1)], [(11, 1)], 1)
    add_product_terms([(22, 1)], [(21, 1), (11, -1)], [(21, 1)], 1)
    
    # M5 = (A11 + A12) * B22
    # Contributes to C11 (-1), C12 (+1)
    add_product_terms([(11, 1), (12, 1)], [(22, 1)], [(11, -1)], 1)
    add_product_terms([(11, 1), (12, 1)], [(22, 1)], [(12, 1)], 1)
    
    # M6 = (A21 - A11) * (B11 + B12)
    # Contributes to C22 (+1)
    add_product_terms([(21, 1), (11, -1)], [(11, 1), (12, 1)], [(22, 1)], 1)
    
    # M7 = (A12 - A22) * (B21 + B22)
    # Contributes to C11 (+1)
    add_product_terms([(12, 1), (22, -1)], [(21, 1), (22, 1)], [(11, 1)], 1)
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We decompose the 4x5 * 5x5 multiplication into:
    # 1. A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]  (Rank 49 via Strassen)
    # 2. A[0:4, 0:4] * B[0:4, 4]   -> C[0:4, 4]    (Rank 16 via schoolbook)
    # 3. A[0:4, 4]   * B[4, 0:5]   -> C[0:4, 0:5]  (Rank 20 via schoolbook)
    
    # Generate Strassen decomposition for [4, 4, 4]
    strassen_terms = _strassen_decomp(2)  # k=2 for size 4
    
    # Count total rank
    rank = len(strassen_terms) + 16 + 20
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 20 x rank
    V = [[0] * rank for _ in range(m * p)]  # 25 x rank
    W = [[0] * rank for _ in range(p * n)]  # 20 x rank
    
    col = 0
    
    # Part 1: [4, 4, 4] using Strassen
    for u_indices, v_indices, w_indices, coeff in strassen_terms:
        if coeff == 0:
            continue
        
        for (i, j) in u_indices:
            # U[i*m + j, col]
            u_idx = i * 5 + j
            U[u_idx][col] += coeff
        
        for (j, k) in v_indices:
            # V[j*p + k, col]
            v_idx = j * 5 + k
            V[v_idx][col] += coeff
        
        for (k, i) in w_indices:
            # W[k*n + i, col]
            w_idx = k * 4 + i
            W[w_idx][col] += coeff
        
        col += 1
    
    # Part 2: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # Schoolbook: rank 4*4 = 16
    for i in range(4):
        for j in range(4):
            k = 4
            U[i * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    # Part 3: A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    # Schoolbook: rank 4*5 = 20
    for i in range(4):
        j = 4
        for k in range(5):
            U[i * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
