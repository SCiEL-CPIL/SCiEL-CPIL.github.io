import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS
    
    # Initialize U, V, W with 0
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the decomposition block by block.
    # We need a list of (U_block, V_block, W_block) for each sub-problem
    # and the offset in the global rank dimension.
    
    blocks = []
    
    # Helper to get Strassen 2x2x2 decomposition
    # Returns lists of (u, v, w) where u, v, w are vectors of length 4 (flattened)
    # Actually, we need to map to the specific sub-matrix indices.
    
    # Let's define the Strassen coefficients for 2x2x2
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We will create a function to add a block to the global matrices.
    # block_add(i_offset, j_offset, k_offset, n_b, m_b, p_b, decomposition_type)
    
    # decomposition_type:
    # '222': 2x2x2 Strassen (Rank 7)
    # '223': 2x2x3 (Rank 9) -> 222 + 221
    # '12p': 1x2xp (Rank 2)
    # '221': 2x2x1 (Rank 2)
    
    current_rank = 0
    
    def add_block(i_off, j_off, k_off, n_b, m_b, p_b, dtype):
        nonlocal current_rank
        if time.time() > deadline:
            return
        
        if dtype == '222':
            # Rank 7
            # Local indices: i in 0..1, j in 0..1, k in 0..1
            # U indices: i*2 + j
            # V indices: j*2 + k
            # W indices: k*2 + i
            
            # P1: (A11+A22)(B11+B22)
            # u: A11=1, A22=1. v: B11=1, B22=1. w: C11=1, C22=1
            blocks.append((
                { (i_off+0)*2 + (j_off+0): 1, (i_off+1)*2 + (j_off+1): 1 },
                { (j_off+0)*2 + (k_off+0): 1, (j_off+1)*2 + (k_off+1): 1 },
                { (k_off+0)*2 + (i_off+0): 1, (k_off+1)*2 + (i_off+1): 1 }
            ))
            
            # P2: (A21+A22)B11
            blocks.append((
                { (i_off+1)*2 + (j_off+0): 1, (i_off+1)*2 + (j_off+1): 1 },
                { (j_off+0)*2 + (k_off+0): 1 },
                { (k_off+0)*2 + (i_off+1): 1 }
            ))
            
            # P3: A11(B12-B22)
            blocks.append((
                { (i_off+0)*2 + (j_off+0): 1 },
                { (j_off+0)*2 + (k_off+1): 1, (j_off+1)*2 + (k_off+1): -1 },
                { (k_off+1)*2 + (i_off+0): 1 }
            ))
            
            # P4: A22(B21-B11)
            blocks.append((
                { (i_off+1)*2 + (j_off+1): 1 },
                { (j_off+1)*2 + (k_off+0): 1, (j_off+0)*2 + (k_off+0): -1 },
                { (k_off+0)*2 + (i_off+1): 1 }
            ))
            
            # P5: (A11+A12)B22
            blocks.append((
                { (i_off+0)*2 + (j_off+0): 1, (i_off+0)*2 + (j_off+1): 1 },
                { (j_off+1)*2 + (k_off+1): 1 },
                { (k_off+1)*2 + (i_off+0): 1 }
            ))
            
            # P6: (A21-A11)(B11+B12)
            blocks.append((
                { (i_off+1)*2 + (j_off+0): 1, (i_off+0)*2 + (j_off+0): -1 },
                { (j_off+0)*2 + (k_off+0): 1, (j_off+0)*2 + (k_off+1): 1 },
                { (k_off+0)*2 + (i_off+1): 1, (k_off+1)*2 + (i_off+1): 1 }
            ))
            
            # P7: (A12-A22)(B21+B22)
            blocks.append((
                { (i_off+0)*2 + (j_off+1): 1, (i_off+1)*2 + (j_off+1): -1 },
                { (j_off+1)*2 + (k_off+0): 1, (j_off+1)*2 + (k_off+1): 1 },
                { (k_off+0)*2 + (i_off+0): 1, (k_off+1)*2 + (i_off+0): 1 }
            ))
            
            current_rank += 7
            
        elif dtype == '221':
            # 2x2x1. Rank 2.
            # C1 = A11 B11 + A12 B21
            # C2 = A21 B11 + A22 B21
            # Term 1: A11 B11 -> C1
            blocks.append((
                { (i_off+0)*2 + (j_off+0): 1 },
                { (j_off+0)*1 + (k_off+0): 1 },
                { (k_off+0)*2 + (i_off+0): 1 }
            ))
            # Term 2: A12 B21 -> C1
            blocks.append((
                { (i_off+0)*2 + (j_off+1): 1 },
                { (j_off+1)*1 + (k_off+0): 1 },
                { (k_off+0)*2 + (i_off+0): 1 }
            ))
            # Term 3: A21 B11 -> C2
            blocks.append((
                { (i_off+1)*2 + (j_off+0): 1 },
                { (j_off+0)*1 + (k_off+0): 1 },
                { (k_off+0)*2 + (i_off+1): 1 }
            ))
            # Term 4: A22 B21 -> C2
            blocks.append((
                { (i_off+1)*2 + (j_off+1): 1 },
                { (j_off+1)*1 + (k_off+0): 1 },
                { (k_off+0)*2 + (i_off+1): 1 }
            ))
            # Wait, rank of 2x2x1 is 2?
            # C1 = A11 B11 + A12 B21
            # C2 = A21 B11 + A22 B21
            # We can combine?
            # Let T1 = (A11+A21) * (B11) ? No.
            # Actually, R(2,2,1) = 2.
            # C = A * b.
            # C1 = A11 b1 + A12 b2
            # C2 = A21 b1 + A22 b2
            # We can use 2 terms:
            # Term 1: (A11, A21) dot (b1) -> contributes to C1, C2?
            # If we take u = [1, 0, 0, 0] (A11), v = [1, 0] (b1), w = [1, 1] (C1, C2)?
            # Then C1 gets A11 b1. C2 gets A11 b1. But C2 should get A21 b1.
            # So we need separate terms for each A_ij?
            # Actually, for n=2, m=2, p=1, the rank is 2.
            # Decomposition:
            # T1: A11, B11, W=[1,0] -> C1 += A11 B11
            # T2: A12, B21, W=[1,0] -> C1 += A12 B21
            # T3: A21, B11, W=[0,1] -> C2 += A21 B11
            # T4: A22, B21, W=[0,1] -> C2 += A22 B21
            # This is rank 4.
            # Can we do better?
            # C1 = A11 B11 + A12 B21
            # C2 = A21 B11 + A22 B21
            # Let X = A11 + A21, Y = B11.
            # Let Z = A12 + A22, W = B21.
            # This doesn't separate C1 and C2 easily.
            # Actually, R(2,2,1) is 2.
            # Reference: "The rank of the matrix multiplication tensor"
            # R(n,1,1) = 1. R(1,n,1) = 1. R(1,1,n) = 1.
            # R(2,2,1) = 2.
            # Terms:
            # 1. (A11+A21) * B11 * [1, 1]? No.
            # Let's just use the 4-term decomposition if 2 is hard to verify, but 4 is safe.
            # However, 4 terms for 2x2x1 is schoolbook.
            # Is it 2?
            # C1 = A11 B11 + A12 B21
            # C2 = A21 B11 + A22 B21
            # If we use 2 terms:
            # T1: u1, v1, w1
            # T2: u2, v2, w2
            # w1, w2 are vectors in R2.
            # If w1 = [1, 0], w2 = [0, 1].
            # Then C1 = <u1, A> <v1, B>. C2 = <u2, A> <v2, B>.
            # This implies C1 and C2 are independent rank-1 forms.
            # But C1 and C2 share B11 and B21.
            # C1 = A11 B11 + A12 B21
            # C2 = A21 B11 + A22 B21
            # If we set u1 = [1, 0, 0, 0] (A11), v1 = [1, 0] (B11).
            # u2 = [0, 0, 1, 0] (A21), v2 = [1, 0] (B11).
            # This doesn't cover A12, A22.
            # So rank is at least 4?
            # Actually, R(2,2,1) = 2 is incorrect?
            # R(2,2,1) is the rank of the bilinear form for 2x2 matrix times 2x1 vector.
            # It is known that R(2,2,1) = 2.
            # Wait, if R(2,2,1)=2, then C1 and C2 must be linear combinations of 2 rank-1 terms.
            # This is possible if the terms are coupled.
            # Let's use 4 terms to be safe, or find the 2-term one.
            # For now, let's use 4 terms for 2x2x1 to ensure correctness, but this increases rank.
            # My previous estimate for 2x2x3 was 7+2=9. If 2x2x1 is 4, then 2x2x3 is 7+4=11.
            # Total rank would be 11+11+7+7+4+4 = 44.
            # Still better than 60.
            
            # Let's stick to the 4-term decomposition for 2x2x1 for simplicity and correctness.
            current_rank += 4

        elif dtype == '12p':
            # 1x2xp. Rank 2.
            # C1 = A1 B11 + A2 B21
            # C2 = A1 B12 + A2 B22
            # ...
            # Term 1: A1, B1, W=[1, 0, ...]
            # Term 2: A2, B2, W=[1, 0, ...]
            # This gives Ck = A1 B1k + A2 B2k.
            # This is correct.
            for k_local in range(p_b):
                # We need separate W for each k?
                # No, the term is A1 * B1_row.
                # The contribution to C_k is A1 * B1_k.
                # So for a fixed j (0 or 1), the term is A_j * B_j_row.
                # W should have 1s in all k positions?
                # No, W is a vector of length n_b * p_b?
                # W index is k*n + i.
                # For n_b=1, i=0.
                # W[k*1 + 0] = 1 for all k?
                # Then C_k gets A_j B_jk.
                # Yes.
                # So for each j in 0..1:
                #   u: A_j = 1
                #   v: B_jk = 1 for all k
                #   w: C_k = 1 for all k
                pass # Implemented below

    # Add blocks
    # A1 (rows 0,1) * B1 (cols 0,1,2)
    # A1a (cols 0,1) * B1a (rows 0,1) -> 2x2x3
    add_block(0, 0, 0, 2, 2, 3, '223')
    # A1b (cols 2,3) * B1b (rows 2,3) -> 2x2x3
    add_block(0, 2, 0, 2, 2, 3, '223')
    
    # A1 * B2 (cols 3,4)
    # A1a (cols 0,1) * B2a (rows 0,1) -> 2x2x2
    add_block(0, 0, 2, 2, 2, 2, '222')
    # A1b (cols 2,3) * B2b (rows 2,3) -> 2x2x2
    add_block(0, 2, 2, 2, 2, 2, '222')
    
    # A2 (row 2) * B1 (cols 0,1,2)
    # A2a (cols 0,1) * B1a (rows 0,1) -> 1x2x3
    add_block(2, 0, 0, 1, 2, 3, '12p')
    # A2b (cols 2,3) * B1b (rows 2,3) -> 1x2x3
    add_block(2, 2, 0, 1, 2, 3, '12p')
    
    # A2 * B2 (cols 3,4)
    # A2c (cols 0,1) * B2c (rows 0,1) -> 1x2x2
    add_block(2, 0, 2, 1, 2, 2, '12p')
    # A2d (cols 2,3) * B2d (rows 2,3) -> 1x2x2
    add_block(2, 2, 2, 1, 2, 2, '12p')

    # Build U, V, W
    U = [[0] * current_rank for _ in range(n * m)]
    V = [[0] * current_rank for _ in range(m * p)]
    W = [[0] * current_rank for _ in range(p * n)]
    
    col = 0
    for u_dict, v_dict, w_dict in blocks:
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
        
    return {"U": U, "V": V, "W": W}

# Need to implement '223' and '12p' properly in add_block
