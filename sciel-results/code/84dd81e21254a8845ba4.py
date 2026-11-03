"""Matrix multiplication 3x5x6 with block Strassen optimization."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # We'll build the decomposition by blocks.
    # Split j (size 5) into j1 (size 2) and j2 (size 3)
    # Split k (size 6) into k1 (size 2) and k2 (size 4)
    # For each i in range(3):
    #   Block (i, j1, k1): 1x2x2 -> use Strassen rank 7? No, 1x2x2 is just 4 products
    #   Actually for 1x2x2, A is 1x2, B is 2x2, C is 2x1. Rank = 4 (schoolbook)
    #   For 2x2x2 blocks, we can use Strassen rank 7.
    
    # Let's partition:
    # i: [0], [1,2]  (sizes 1, 2)
    # j: [0,1], [2,3,4]  (sizes 2, 3)
    # k: [0,1], [2,3], [4,5]  (sizes 2, 2, 2)
    
    # Blocks:
    # (i=[0], j=[0,1], k=[0,1]): 1x2x2, rank=4
    # (i=[0], j=[0,1], k=[2,3]): 1x2x2, rank=4
    # (i=[0], j=[0,1], k=[4,5]): 1x2x2, rank=4
    # (i=[0], j=[2,3,4], k=[0,1]): 1x3x2, rank=6
    # (i=[0], j=[2,3,4], k=[2,3]): 1x3x2, rank=6
    # (i=[0], j=[2,3,4], k=[4,5]): 1x3x2, rank=6
    # (i=[1,2], j=[0,1], k=[0,1]): 2x2x2, rank=7 (Strassen)
    # (i=[1,2], j=[0,1], k=[2,3]): 2x2x2, rank=7
    # (i=[1,2], j=[0,1], k=[4,5]): 2x2x2, rank=7
    # (i=[1,2], j=[2,3,4], k=[0,1]): 2x3x2, rank=12 (schoolbook) or better?
    # (i=[1,2], j=[2,3,4], k=[2,3]): 2x3x2, rank=12
    # (i=[1,2], j=[2,3,4], k=[4,5]): 2x3x2, rank=12
    
    # Total: 3*4 + 3*6 + 3*7 + 3*12 = 12 + 18 + 21 + 36 = 87
    
    # Can we do better on 2x3x2? 
    # 2x3x2: A is 2x3, B is 3x2, C is 2x2. 
    # Schoolbook: 2*3*2 = 12.
    # We can split j (size 3) into [0] and [1,2].
    # (i=[1,2], j=[0], k=[0,1]): 2x1x2, rank=4
    # (i=[1,2], j=[1,2], k=[0,1]): 2x2x2, rank=7 (Strassen)
    # Total for 2x3x2: 4+7 = 11, saving 1.
    
    # Revised total: 3*4 + 3*6 + 3*7 + 3*11 = 12 + 18 + 21 + 33 = 84
    
    # Can we do better on 1x3x2?
    # 1x3x2: A is 1x3, B is 3x2, C is 2x1. Rank = 6 (schoolbook).
    # Split j into [0] and [1,2]:
    # (i=[0], j=[0], k=[0,1]): 1x1x2, rank=2
    # (i=[0], j=[1,2], k=[0,1]): 1x2x2, rank=4
    # Total: 2+4 = 6. No improvement.
    
    # So total rank = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end, block_rank):
        nonlocal col
        # For this block, we need to add block_rank rank-1 terms
        # The block is (i_end-i_start) x (j_end-j_start) x (k_end-k_start)
        n_b = i_end - i_start
        m_b = j_end - j_start
        p_b = k_end - k_start
        
        if n_b == 1 and m_b == 1 and p_b == 1:
            # 1x1x1: rank 1
            U[i_start * m + j_start][col] = 1
            V[j_start * p + k_start][col] = 1
            W[k_start * n + i_start][col] = 1
            col += 1
        elif n_b == 1 and m_b == 1 and p_b == 2:
            # 1x1x2: rank 2
            for kk in range(p_b):
                U[i_start * m + j_start][col] = 1
                V[j_start * p + k_start + kk][col] = 1
                W[(k_start + kk) * n + i_start][col] = 1
                col += 1
        elif n_b == 1 and m_b == 2 and p_b == 2:
            # 1x2x2: rank 4 (schoolbook)
            for jj in range(m_b):
                for kk in range(p_b):
                    U[i_start * m + j_start + jj][col] = 1
                    V[(j_start + jj) * p + k_start + kk][col] = 1
                    W[(k_start + kk) * n + i_start][col] = 1
                    col += 1
        elif n_b == 1 and m_b == 3 and p_b == 2:
            # 1x3x2: rank 6 (schoolbook)
            for jj in range(m_b):
                for kk in range(p_b):
                    U[i_start * m + j_start + jj][col] = 1
                    V[(j_start + jj) * p + k_start + kk][col] = 1
                    W[(k_start + kk) * n + i_start][col] = 1
                    col += 1
        elif n_b == 2 and m_b == 2 and p_b == 2:
            # 2x2x2: use Strassen, rank 7
            # Strassen for C = A*B where A is 2x2, B is 2x2, C is 2x2
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1 - M4 + M6 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M3 + M5 + M7
            
            # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
            # A[i_l, j_l] -> U[(i_start+i_l)*m + (j_start+j_l), col_offset + t]
            # B[j_l, k_l] -> V[(j_start+j_l)*p + (k_start+k_l), col_offset + t]
            # C[k_l, i_l] -> W[(k_start+k_l)*n + (i_start+i_l), col_offset + t]
            # Note: C is indexed as C[k, i] in the W matrix per the trace convention W[k*n+i]
            
            M_defs = [
                # (A_coeffs: dict of (i_l,j_l)->val, B_coeffs: dict of (j_l,k_l)->val, C_coeffs: dict of (k_l,i_l)->val)
                ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),  # M2
                ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):-1}),  # M3
                ({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):1, (0,0):-1}),  # M4
                ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),  # M5
                ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):1}),  # M6
                ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M7
            ]
            
            for A_coeffs, B_coeffs, C_coeffs in M_defs:
                for (il, jl), a_val in A_coeffs.items():
                    U[(i_start + il) * m + (j_start + jl)][col] = a_val
                for (jl, kl), b_val in B_coeffs.items():
                    V[(j_start + jl) * p + (k_start + kl)][col] = b_val
                for (kl, il), c_val in C_coeffs.items():
                    W[(k_start + kl) * n + (i_start + il)][col] = c_val
                col += 1
        elif n_b == 2 and m_b == 3 and p_b == 2:
            # 2x3x2: split into 2x1x2 (rank 4) + 2x2x2 (rank 7) = 11
            # First part: j_start only
            for kk in range(p_b):
                for il in range(n_b):
                    U[(i_start + il) * m + j_start][col] = 1
                    V[j_start * p + k_start + kk][col] = 1
                    W[(k_start + kk) * n + (i_start + il)][col] = 1
                    col += 1
            # Second part: j_start+1, j_start+2 with Strassen
            # Call the 2x2x2 Strassen code
            i_s2 = i_start
            j_s2 = j_start + 1
            k_s2 = k_start
            n_b2 = 2
            m_b2 = 2
            p_b2 = 2
            M_defs = [
                ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
                ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
                ({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):1, (0,0):-1}),
                ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
                ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):1}),
                ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ]
            for A_coeffs, B_coeffs, C_coeffs in M_defs:
                for (il, jl), a_val in A_coeffs.items():
                    U[(i_s2 + il) * m + (j_s2 + jl)][col] = a_val
                for (jl, kl), b_val in B_coeffs.items():
                    V[(j_s2 + jl) * p + (k_s2 + kl)][col] = b_val
                for (kl, il), c_val in C_coeffs.items():
                    W[(k_s2 + kl) * n + (i_s2 + il)][col] = c_val
                col += 1
        else:
            raise ValueError(f"Unknown block size: {n_b}x{m_b}x{p_b}")
    
    # i=[0], j=[0,1], k=[0,1]: 1x2x2
    add_block(0, 1, 0, 2, 0, 2, 4)
    # i=[0], j=[0,1], k=[2,3]: 1x2x2
    add_block(0, 1, 0, 2, 2, 4, 4)
    # i=[0], j=[0,1], k=[4,5]: 1x2x2
    add_block(0, 1, 0, 2, 4, 6, 4)
    # i=[0], j=[2,3,4], k=[0,1]: 1x3x2
    add_block(0, 1, 2, 5, 0, 2, 6)
    # i=[0], j=[2,3,4], k=[2,3]: 1x3x2
    add_block(0, 1, 2, 5, 2, 4, 6)
    # i=[0], j=[2,3,4], k=[4,5]: 1x3x2
    add_block(0, 1, 2, 5, 4, 6, 6)
    # i=[1,2], j=[0,1], k=[0,1]: 2x2x2 Strassen
    add_block(1, 3, 0, 2, 0, 2, 7)
    # i=[1,2], j=[0,1], k=[2,3]: 2x2x2 Strassen
    add_block(1, 3, 0, 2, 2, 4, 7)
    # i=[1,2], j=[0,1], k=[4,5]: 2x2x2 Strassen
    add_block(1, 3, 0, 2, 4, 6, 7)
    # i=[1,2], j=[2,3,4], k=[0,1]: 2x3x2
    add_block(1, 3, 2, 5, 0, 2, 11)
    # i=[1,2], j=[2,3,4], k=[2,3]: 2x3x2
    add_block(1, 3, 2, 5, 2, 4, 11)
    # i=[1,2], j=[2,3,4], k=[4,5]: 2x3x2
    add_block(1, 3, 2, 5, 4, 6, 11)
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

