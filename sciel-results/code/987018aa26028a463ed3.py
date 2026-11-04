import time
import sys

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2_UVW():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2 -> 2x2 matrix multiplication.
    
    U: shape (4, 7) for A[0,0], A[0,1], A[1,0], A[1,1]
    V: shape (4, 7) for B[0,0], B[0,1], B[1,0], B[1,1]
    W: shape (4, 7) for C[0,0], C[0,1], C[1,0], C[1,1]
    
    C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    U = [
        [1, 1, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 1, 0, 0, -1],  # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, 1],   # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [1, 0, -1, 0, 1, 0, 1],  # b10
        [1, 0, 0, 0, 1, 0, 1],   # b11
    ]
    W = [
        [1, 0, 0, 1],   # C00 = M1 - M4 + M6
        [0, 1, 1, 0],   # C01 = M3 + M5
        [0, 1, 0, 1],   # C10 = M2 + M4
        [1, 0, -1, 1],  # C11 = M1 - M3 + M7
    ]
    return U, V, W

def _block_4x4x4_UVW():
    """
    Build a rank-7 decomposition for 4x4 x 4x4 -> 4x4 using 2x2 Strassen blocks.
    
    We use a block Strassen approach: split each 4x4 matrix into four 2x2 blocks.
    A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]] where Cij = Ai0*B0j + Ai1*B1j
    
    Using Strassen on the 2x2 block system:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M4 + M6
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M7
    
    Each Mj is a 2x2 product, which itself has rank 7 (Strassen).
    So total rank = 7 * 7 = 49? No, that's not right.
    
    Actually, we can't just multiply the ranks. Let me think again.
    
    Each block product like (A00+A11)(B00+B11) is a 2x2 x 2x2 -> 2x2 multiplication.
    If we use Strassen for each, that's 7 scalar products per block product.
    There are 7 block products, so 49 scalar products total.
    
    But we can do better by sharing. Actually, the standard way to get rank 7 for 4x4x4
    is to use a different approach. Let me use the known fact that the tensor rank
    of 4x4x4 matrix multiplication is at most 7 (from Strassen's algorithm applied
    recursively, but that gives 7^log2(4) = 49).
    
    Hmm, I'm confusing myself. Strassen's algorithm for 2x2x2 has rank 7.
    For 4x4x4, if we apply Strassen recursively, we get rank 7^2 = 49.
    The trivial rank is 64.
    
    Let me just use the trivial rank-64 decomposition for the 4x4x4 part,
    or find a better one.
    
    Actually, let me reconsider the whole approach. For 4x4x7, the trivial rank is 112.
    If I split into 4x4x4 (rank 64) + 4x4x3 (rank 48), total = 112. No improvement.
    
    I need a better approach. Let me think about what's known.
    
    The best known upper bound for 4x4x4 is 49 (recursive Strassen).
    For 4x4x3, I'm not sure of the best bound.
    
    Let me try a different strategy: use the fact that we can write the 4x4x7 tensor
    as a sum of simpler tensors.
    
    Alternative: Use a "tensor train" style decomposition where we factor the indices
    in a hierarchical way.
    
    For now, let me implement a simple improvement: split the 7 columns into 4+3,
    use recursive Strassen (rank 49) for the 4x4x4 part, and trivial (rank 48) for
    the 4x4x3 part. Total rank = 97.
    
    That's better than 112!
    """
    # For 4x4x4: use recursive Strassen, rank 49
    # For 4x4x3: use trivial, rank 48
    # Total: 97
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # This is complex, so let me use a simpler approach for now.
    
    # Actually, let me just use the trivial decomposition for both parts
    # and see if I can find any improvement.
    
    # For 4x4x4 trivial: rank 64
    # For 4x4x3 trivial: rank 48
    # Total: 112 (same as original)
    
    # Let me implement the 4x4x4 part with rank 49 using recursive Strassen.
    
    def _strassen_4x4x4_UVW():
        """
        Build rank-49 decomposition for 4x4x4 using recursive Strassen.
        """
        # Get the 2x2x2 Strassen decomposition
        U2, V2, W2 = _strassen_2x2_2x2_UVW()
        
        # For 4x4x4, we have 4x4 A, 4x4 B, 4x4 C
        # Split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        #
        # Using Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        #
        # Each Mj is a 2x2x2x2 -> 2x2 multiplication (i.e., 2x2 matrix product)
        # Each has rank 7 (Strassen)
        # Total rank = 7 * 7 = 49
        
        # Build the full 4x4x4 decomposition
        rank = 49
        U = [[0] * rank for _ in range(16)]  # 4x4 = 16 entries
        V = [[0] * rank for _ in range(16)]  # 4x4 = 16 entries
        W = [[0] * rank for _ in range(16)]  # 4x4 = 16 entries
        
        # Define the 7 Strassen combinations at the block level
        # For each block product Mj, we need to specify which A-blocks and B-blocks
        # are involved, and with what coefficients.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Block indices: A00=0, A01=1, A10=2, A11=3 (each is a 2x2 block)
        # Similarly for B and C blocks.
        
        # For each of the 7 block products, and for each of the 7 Strassen terms
        # within that product, we add contributions.
        
        # A-block to row mapping: A00 rows 0-3, A01 rows 4-7, A10 rows 8-11, A11 rows 12-15
        # B-block to col mapping: B00 cols 0-3, B01 cols 4-7, B10 cols 8-11, B11 cols 12-15
        # C-block to output mapping: C00 outputs 0-3, C01 outputs 4-7, C10 outputs 8-11, C11 outputs 12-15
        
        block_rows = {0: [0,1,2,3], 1: [4,5,6,7], 2: [8,9,10,11], 3: [12,13,14,15]}
        block_cols = {0: [0,1,2,3], 1: [4,5,6,7], 2: [8,9,10,11], 3: [12,13,14,15]}
        block_outs = {0: [0,1,2,3], 1: [4,5,6,7], 2: [8,9,10,11], 3: [12,13,14,15]}
        
        # For each block product Mj, define:
        # - Which A-blocks are involved and with what coefficients
        # - Which B-blocks are involved and with what coefficients
        # - Which C-blocks it contributes to and with what coefficients
        
        block_products = [
            # M1 = (A00+A11)(B00+B11)
            {'a_blocks': [(0, 1), (3, 1)], 'b_blocks': [(0, 1), (3, 1)], 
             'c_contrib': [(0, 1), (3, 1)]},  # C00 += M1, C11 += M1
            # M2 = (A10+A11)B00
            {'a_blocks': [(2, 1), (3, 1)], 'b_blocks': [(0, 1)],
             'c_contrib': [(2, 1)]},  # C10 += M2
            # M3 = A00(B01-B11)
            {'a_blocks': [(0, 1)], 'b_blocks': [(1, 1), (3, -1)],
             'c_contrib': [(1, 1), (3, -1)]},  # C01 += M3, C11 -= M3
            # M4 = A11(B10-B00)
            {'a_blocks': [(3, 1)], 'b_blocks': [(2, 1), (0, -1)],
             'c_contrib': [(0, -1), (2, 1)]},  # C00 -= M4, C10 += M4
            # M5 = (A00+A01)B11
            {'a_blocks': [(0, 1), (1, 1)], 'b_blocks': [(3, 1)],
             'c_contrib': [(1, 1)]},  # C01 += M5
            # M6 = (A10-A00)(B00+B01)
            {'a_blocks': [(2, 1), (0, -1)], 'b_blocks': [(0, 1), (1, 1)],
             'c_contrib': [(0, 1)]},  # C00 += M6
            # M7 = (A01-A11)(B10+B11)
            {'a_blocks': [(1, 1), (3, -1)], 'b_blocks': [(2, 1), (3, 1)],
             'c_contrib': [(3, 1)]},  # C11 += M7
        ]
        
        # For each block product and each Strassen term within it
        for bp_idx, bp in enumerate(block_products):
            for s_idx in range(7):  # 7 Strassen terms
                r = bp_idx * 7 + s_idx
                
                # Get the Strassen coefficients for this term
                u_coeff = U2[s_idx]  # length 4: coefficients for a00,a01,a10,a11
                v_coeff = V2[s_idx]  # length 4: coefficients for b00,b01,b10,b11
                w_coeff = W2[s_idx]  # length 4: coefficients for C00,C01,C10,C11
                
                # For each A-block involved
                for a_blk, a_coeff in bp['a_blocks']:
                    for local_i in range(2):
                        for local_j in range(2):
                            # Global row index for A[a_blk, local_i, local_j]
                            global_row = a_blk * 4 + local_i * 2 + local_j
                            # Strassen local index for this position in the 2x2 block
                            local_idx = local_i * 2 + local_j
                            U[global_row][r] = a_coeff * u_coeff[local_idx]
                
                # For each B-block involved
                for b_blk, b_coeff in bp['b_blocks']:
                    for local_j in range(2):
                        for local_k in range(2):
                            global_col = b_blk * 4 + local_j * 2 + local_k
                            local_idx = local_j * 2 + local_k
                            V[global_col][r] = b_coeff * v_coeff[local_idx]
                
                # For each C-block this contributes to
                for c_blk, c_coeff in bp['c_contrib']:
                    for local_i in range(2):
                        for local_k in range(2):
                            global_out = c_blk * 4 + local_i * 2 + local_k
                            local_idx = local_i * 2 + local_k
                            W[global_out][r] = c_coeff * w_coeff[local_idx]
        
        return U, V, W
    
    U444, V444, W444 = _strassen_4x4x4_UVW()
    
    # Build 4x4x3 trivial decomposition (rank 48)
    rank_443 = 4 * 4 * 3
    U443 = [[0] * rank_443 for _ in range(16)]
    V443 = [[0] * rank_443 for _ in range(12)]  # 4x3 = 12
    W443 = [[0] * rank_443 for _ in range(12)]  # 3x4 = 12
    
    t = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 3 + k
                w_idx = k * 4 + i
                U443[u_idx][t] = 1
                V443[v_idx][t] = 1
                W443[w_idx][t] = 1
                t += 1
    
    # Combine: total rank = 49 + 48 = 97
    total_rank = 49 + 48
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]  # 4x7 = 28
    W = [[0] * total_rank for _ in range(28)]  # 7x4 = 28
    
    # Copy 4x4x4 part (columns 0-3 of B, rows 0-3 of C)
    for i in range(16):
        for r in range(49):
            U[i][r] = U444[i][r]
    
    for j in range(4):
        for k in range(4):
            v_idx = j * 7 + k
            for r in range(49):
                V[v_idx][r] = V444[j * 4 + k][r]
    
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            for r in range(49):
                W[w_idx][r] = W444[k * 4 + i][r]
    
    # Copy 4x4x3 part (columns 4-6 of B, rows 4-6 of C)
    offset = 49
    for i in range(16):
        for r in range(48):
            U[i][offset + r] = U443[i][r]
    
    for j in range(4):
        for k in range(3):
            v_idx = j * 7 + (4 + k)
            for r in range(48):
                V[v_idx][offset + r] = V443[j * 3 + k][r]
    
    for k in range(3):
        for i in range(4):
            w_idx = (4 + k) * 4 + i
            for r in range(48):
                W[w_idx][offset + r] = W443[k * 4 + i][r]
    
    return U, V, W

def _trivial_decomposition(n, m, p):
    """
    Return the trivial (rank n*m*p) exact tensor decomposition for n x m x p matrix multiplication.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition (U, V, W) correctly computes matrix multiplication.
    """
    try:
        import numpy as np
        np.random.seed(12345)
        
        for trial in range(3):
            A = np.random.randint(-10, 10, (n, m), dtype=float)
            B = np.random.randint(-10, 10, (m, p), dtype=float)
            C_expected = A @ B
            
            rank = len(U[0])
            C_computed = np.zeros((n, p), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        w_val = W[k * n + i]
                        for r in range(rank):
                            s += u_row[r] * v_row[r] * w_val[r]
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    except Exception:
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 7
    
    # Try the hierarchical decomposition (rank 97)
    try:
        U, V, W = _block_4x4x4_UVW()
        
        # Verify
        if _verify_decomposition(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to trivial decomposition (rank 112)
    U, V, W = _trivial_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
