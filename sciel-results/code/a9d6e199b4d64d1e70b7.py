DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 2x2 block decomposition of 3x3 sub-matrices
    # Rank of 3x3 mult is 19 (Laderman)
    # Total rank = 2*2*2 * 19 = 152
    
    # Laderman factors for 3x3 matrix multiplication
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    # Note: W index is k*3+i (col*3+row)
    
    # Laderman 19-term decomposition
    # Reference: Laderman (1976)
    # The coefficients are integers.
    
    U3 = [
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a11
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a12
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a13
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a21
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a22
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # a23
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # a31
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # a32
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # a33
    ]
    
    V3 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b11
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b12
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b13
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b21
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b22
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b23
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b31
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b32
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # b33
    ]
    
    # This V3 is identity, which implies the decomposition is just scaling?
    # No, Laderman factors are not identity.
    # I must use the correct Laderman factors.
    # Since I am not 100% sure of the Laderman factors, I will use the Strassen 189 solution which is safe.
    # Strassen 2x2 rank 7.
    # 3x3 blocks of 2x2.
    # Rank 27 * 7 = 189.
    
    # Let's stick to Strassen 189 to ensure correctness.
    # The prompt says "score 1.143" was achieved by Strassen.
    # I will implement Strassen 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # U2 (2x2 A factors)
    U2 = [
        [1, 1, 0, 0, 1, 1, 0], # a11
        [1, 1, 0, 0, 0, 0, 1], # a21? No.
        # a11: M1, M2, M3, M4, M5, M6, M7
        # M1: a11+a22
        # M2: a21+a22
        # M3: a11
        # M4: a22
        # M5: a11+a12
        # M6: a21-a11
        # M7: a12-a22
        # Row a11: [1, 0, 1, 0, 1, -1, 0]
        # Row a12: [0, 0, 0, 0, 1, 0, 1]
        # Row a21: [0, 1, 0, 0, 0, 1, 0]
        # Row a22: [1, 1, 0, 1, 0, 0, -1]
    ]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V2 = [
        [1, 1, 1, -1, 0, 1, 0], # b11
        [0, 0, 1, 0, 0, 1, 1], # b12
        [0, 0, -1, 1, 1, 0, 1], # b21
        [1, 0, -1, 0, 1, 0, 1]  # b22
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, -1], # c11
        [0, 0, 1, 0, 1, 0, 0], # c12
        [0, 1, 0, 1, 0, 0, 0], # c21
        [1, -1, 1, 0, 0, 0, 1]  # c22
    ]
    
    # Fill 6x6 matrices
    # Block indices I, J, K in 0..2
    # Local indices a, b, c in 0..1
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Base rank index for this block
                base_rank = (I * 3 + J * 1 + K) * 7 # Wait, 3x3x3=27 blocks.
                # The block is determined by I (row block of A), J (col block of A / row block of B), K (col block of B)
                # Actually, the term is A[I,J] * B[J,K] -> C[I,K]
                # So the block is identified by (I, J, K).
                # There are 27 such terms.
                base_rank = (I * 9 + J * 3 + K) * 7
                
                for r in range(7):
                    rank_idx = base_rank + r
                    for a in range(2):
                        for b in range(2):
                            # A index: Row I*2+a, Col J*2+b
                            a_idx = (I * 2 + a) * 6 + (J * 2 + b)
                            U[a_idx][rank_idx] = U2[a * 2 + b][r]
                            
                            # B index: Row J*2+b, Col K*2+c? No, b is shared.
                            # B index: Row J*2+b, Col K*2+c
                            # Wait, the local indices in U2, V2, W2 are for 2x2.
                            # U2 row is a*2+b.
                            # V2 row is b*2+c.
                            # W2 row is c*2+a. (Task: k*n+i -> c*2+a)
                            
                            # We need to iterate c as well for V and W.
                            # But U is fixed for a,b.
                            # V and W depend on c.
                            # So we should loop c inside?
                            # No, U does not depend on c.
                            # V depends on b,c.
                            # W depends on c,a.
                            
                            # Let's restructure the loop.
                            pass

    # Correct loop structure
    for I in range(3):
        for J in range(3):
            for K in range(3):
                base_rank = (I * 9 + J * 3 + K) * 7
                for r in range(7):
                    rank_idx = base_rank + r
                    for a in range(2):
                        for b in range(2):
                            a_idx = (I * 2 + a) * 6 + (J * 2 + b)
                            U[a_idx][rank_idx] = U2[a * 2 + b][r]
                            
                            for c in range(2):
                                b_idx = (J * 2 + b) * 6 + (K * 2 + c)
                                V[b_idx][rank_idx] = V2[b * 2 + c][r]
                                
                                c_idx = (K * 2 + c) * 6 + (I * 2 + a)
                                W[c_idx][rank_idx] = W2[c * 2 + a][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
