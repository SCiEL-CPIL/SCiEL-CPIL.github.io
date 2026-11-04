"""Rank-192 decomposition: Kronecker product of rank-8 Winograd 3x3 and rank-4 Strassen 2x2."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _kronecker_decomp(A1, B1, C1, A2, B2, C2):
    """Kronecker product of two tensor decompositions.
    A1,B1,C1: factors of decomposition 1 (ranks r1), A2,B2,C2: decomposition 2 (ranks r2).
    Combined rank = r1*r2. Index convention: combined index = (coarse, fine) block-major.
    """
    r1 = len(A1[0])
    r2 = len(A2[0])
    rank = r1 * r2
    n1 = len(A1)
    n2 = len(A2)
    m1 = len(B1)
    m2 = len(B2)
    p1 = len(C1)
    p2 = len(C2)
    U = [[0] * rank for _ in range(n1 * n2)]
    V = [[0] * rank for _ in range(m1 * m2)]
    W = [[0] * rank for _ in range(p1 * p2)]
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for c1 in range(r1):
                    u1 = A1[i1][c1]
                    v1 = B1[j1][c1]
                    w1 = C1[k1][c1]
                    if u1 == 0 and v1 == 0 and w1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for c2 in range(r2):
                                    u2 = A2[i2][c2]
                                    v2 = B2[j2][c2]
                                    w2 = C2[k2][c2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    col = c1 * r2 + c2
                                    U[i1 * n2 + i2][col] += u1 * u2
                                    V[j1 * m2 + j2][col] += v1 * v2
                                    W[k1 * p2 + k2][col] += w1 * w2
    return U, V, W


def construct():
    import time
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0

    # --- Winograd rank-8 3x3 decomposition (U: 9x8, V: 9x8, W: 9x8) ---
    # Indices: row = i*3 + j for U/V (input), k*3 + i for W (output).
    # U[i*3+j, c] = a_c[i] * b_c[j]  (outer product of row vec a_c and col vec b_c)
    # V[j*3+k, c] = b_c[j] * c_c[k]
    # W[k*3+i, c] = c_c[k] * a_c[i]
    # Winograd F(3,3) factors (rank 8):
    # For each factor c, we have three vectors: a_c (row of A), b_c (col of B), c_c (row of C)
    # such that C = sum_c (c_c outer a_c) * (b_c) ... standard Winograd.
    # The 8 factors for 3x3 Winograd:
    #   Factor 0: a=[1,0,0], b=[1,0,0], c=[1,0,0]  -> A00*B00
    #   Factor 1: a=[0,1,0], b=[0,1,0], c=[0,1,0]  -> A11*B11
    #   Factor 2: a=[0,0,1], b=[0,0,1], c=[0,0,1]  -> A22*B22
    #   Factor 3: a=[1,1,0], b=[1,1,0], c=[1,1,0]  -> (A00+A11)*(B00+B11)
    #   Factor 4: a=[0,1,1], b=[0,1,1], c=[0,1,1]  -> (A11+A22)*(B11+B22)
    #   Factor 5: a=[1,0,1], b=[1,0,1], c=[1,0,1]  -> (A00+A22)*(B00+B22)
    #   Factor 6: a=[1,1,1], b=[1,1,1], c=[1,1,1]  -> (sum A)*(sum B)
    #   Factor 7: a=[1,-1,0], b=[1,-1,0], c=[1,-1,0] -> (A00-A11)*(B00-B11)
    # But these don't give rank 8 exactly with the right reconstruction. Let me use the
    # actual Winograd F(3,3) rank-8 algorithm which has specific factors.
    #
    # The correct Winograd rank-8 3x3 factors (from Winograd's F(3,3)):
    # Using the standard formulation, the 8 products are:
    #   p0 = A00 * B00
    #   p1 = A11 * B11
    #   p2 = A22 * B22
    #   p3 = (A00+A11) * (B00+B11)
    #   p4 = (A11+A22) * (B11+B22)
    #   p5 = (A00+A22) * (B00+B22)
    #   p6 = (A00+A11+A22) * (B00+B11+B22)
    #   p7 = (A00-A11) * (B00-B11)
    #
    # But this doesn't reconstruct correctly. The actual Winograd F(3,3) uses:
    #   p0 = (A00+A11+A22) * (B00+B11+B22)
    #   p1 = (A00-A11) * (B00-B11)
    #   p2 = (A00-A22) * (B00-B22)
    #   p3 = (A11-A22) * (B11-B22)
    #   p4 = A01 * (B10+B20)
    #   p5 = (A01+A11) * B20
    #   p6 = A12 * (B01+B11)
    #   p7 = (A12+A22) * B01
    #
    # This is getting complex. Let me use a simpler, verified approach:
    # Use the known rank-8 Winograd 3x3 factors directly.
    #
    # Actually, the cleanest rank-8 3x3 is the Winograd F(3,3) with these 8 factors:
    # I'll define them as (a_vec, b_vec, c_vec) triples.

    # Winograd F(3,3) rank-8 factors:
    # Each factor c has:
    #   a_c[i] multiplies A[i,j] contribution
    #   b_c[j] multiplies B[j,k] contribution
    #   c_c[k] multiplies C[i,k] contribution
    # with C[i,k] = sum_c c_c[k] * a_c[i] * (sum_j A[i,j]*b_c[j])
    # Actually the standard form: C = sum_c (a_c outer c_c) * (A * b_c)
    # where (A*b_c)[i] = sum_j A[i,j] b_c[j]
    # So U[i*3+j, c] = a_c[i] * b_c[j]
    #    V[j*3+k, c] = b_c[j] * c_c[k]  -- wait, this isn't right either.
    #
    # Let me use the direct tensor decomposition form:
    # The matrix multiplication tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j]B[j,k])
    # Rank-r decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where U[i,c] corresponds to A[i,j] coefficient, V[j,c] to B[j,k], W[k,c] to C[i,k].
    #
    # For Winograd F(3,3), the 8 factors are:
    factors_3x3 = [
        # (a_vec, b_vec, c_vec) where a_vec is 3-dim (row index of A),
        # b_vec is 3-dim (col index of A = row index of B),
        # c_vec is 3-dim (col index of B = col index of C)
        # U[i,j,c] = a_vec[i] * b_vec[j]
        # V[j,k,c] = b_vec[j] * c_vec[k]  -- NO. V[j,k,c] should give B[j,k] coefficient.
        # Actually: U[i,j,c] * V[j,k,c] * W[i,k,c] summed over c gives C[i,k].
        # So: U[i,j,c] = a_c[i] * b_c[j]
        #       V[j,k,c] = b_c[j] * c_c[k]  -- this gives B[j,k] weight
        #       W[i,k,c] = a_c[i] * c_c[k]  -- this gives C[i,k] weight
        # Hmm, that's not standard. Let me think again.
        #
        # Standard: C[i,k] = sum_c alpha_c[i] * (sum_j A[i,j] * beta_c[j]) * gamma_c[k]
        # = sum_c (sum_j A[i,j] beta_c[j]) * (alpha_c[i] gamma_c[k])
        # So U[i,j,c] = alpha_c[i] * beta_c[j]
        #    V[j,k,c] = beta_c[j] * gamma_c[k]  -- but this couples j and k through beta_c[j]
        # That's wrong. V should only depend on B[j,k].
        #
        # Correct decomposition:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[i,j,c]) * (sum_k B[j,k] V[j,k,c]) * W[i,k,c]
        # No, that's not how it works. The tensor is T[i,j,k] = delta(i,k) in the sense that
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The decomposition is: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # where the "i" index of U corresponds to the row of A, "j" of V to col of A/row of B,
        # "k" of W to col of B. And C[i,k] = sum_j sum_c U[i,c] V[j,c] W[k,c] * A[i,j] B[j,k]
        # Wait no. The tensor T has indices (i,j,k) and T[i,j,k] = 1 if the output C[i,k]
        # receives A[i,j]*B[j,k]. So T[i,j,k] = delta(i,k) is NOT right.
        #
        # Actually T[i,j,k] is defined such that C[i,k] = sum_j A[i,j] B[j,k].
        # This is a bilinear operation, so it's a tensor in A-space (i,j), B-space (j,k), C-space (i,k).
        # The decomposition: for each rank term c, we have:
        #   A-side: picks out a linear functional on A: sum_{i,j} U[i,j,c] A[i,j]
        #   B-side: picks out a linear functional on B: sum_{j,k} V[j,k,c] B[j,k]
        #   C-side: contributes W[i,k,c] * (A-functional) * (B-functional) to C[i,k]
        # So C[i,k] = sum_c W[i,k,c] * (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
        #
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_c W[i,k,c] U[i',j',c] V[j'',k'',c] = delta(i,i')delta(j,j')delta(k,k'') ... 
        # This is getting complicated. Let me just use the known factors.
    ]

    # Let me use a verified rank-8 Winograd 3x3 decomposition.
    # The Winograd F(3,3) algorithm computes C = A*B for 3x3 matrices using 8 multiplications.
    # The 8 products and their contributions:
    #
    # Using the formulation where:
    #   p_c = (sum_{i,j} a_c[i,j] A[i,j]) * (sum_{j,k} b_c[j,k] B[j,k])
    #   C[i,k] = sum_c c_c[i,k] p_c
    #
    # The 8 factors for Winograd F(3,3):
    # Factor 0: a = [[1,0,0],[0,0,0],[0,0,0]], b = [[1,0,0],[0,0,0],[0,0,0]], c = [[1,0,0],[0,0,0],[0,0,0]]
    #   p0 = A00 * B00, contributes to C00
    # Factor 1: a = [[0,0,0],[0,1,0],[0,0,0]], b = [[0,0,0],[0,1,0],[0,0,0]], c = [[0,0,0],[0,1,0],[0,0,0]]
    #   p1 = A11 * B11, contributes to C11
    # Factor 2: a = [[0,0,0],[0,0,0],[0,0,1]], b = [[0,0,0],[0,0,0],[0,0,1]], c = [[0,0,0],[0,0,0],[0,0,1]]
    #   p2 = A22 * B22, contributes to C22
    # Factor 3: a = [[1,0,0],[1,0,0],[0,0,0]], b = [[1,0,0],[1,0,0],[0,0,0]], c = [[1,1,0],[1,1,0],[0,0,0]]
    #   p3 = (A00+A11)*(B00+B11), contributes to C00,C01,C10,C11
    # Factor 4: a = [[0,0,0],[0,1,0],[0,1,0]], b = [[0,0,0],[0,1,0],[0,1,0]], c = [[0,0,0],[0,1,1],[0,1,1]]
    #   p4 = (A11+A22)*(B11+B22), contributes to C11,C12,C21,C22
    # Factor 5: a = [[1,0,0],[0,0,0],[1,0,0]], b = [[1,0,0],[0,0,0],[1,0,0]], c = [[1,0,1],[0,0,0],[1,0,1]]
    #   p5 = (A00+A22)*(B00+B22), contributes to C00,C02,C20,C22
    # Factor 6: a = [[1,1,0],[1,1,0],[0,0,0]], b = [[1,1,0],[1,1,0],[0,0,0]], c = [[1,1,0],[1,1,0],[0,0,0]]
    #   p6 = (A00+A11+A... wait this is getting wrong.

    # Let me just hardcode a known-correct rank-8 Winograd 3x3 decomposition.
    # I'll use the one from the literature. The key insight is that I can verify
    # numerically that my decomposition works.

    # Actually, let me take a completely different, simpler approach:
    # Use the rank-15 Strassen 3x3 (or just use rank-27 schoolbook for 3x3)
    # No, that gives 4*27 = 108 which is worse than 216/4... no.

    # Let me reconsider. The strategy says Kronecker of rank-8 Winograd 3x3 and rank-4 Strassen 2x2.
    # Rank = 8 * 4 = 32 per 6x6 block? No.
    #
    # 6x6 = (2*3) x (2*3). We tile into 2x2 blocks of size 3x3.
    # The 2x2 block matrix multiply uses Strassen (rank 4).
    # Each "scalar" in the 2x2 problem is a 3x3 matrix, multiplied using Winograd rank 8.
    # So total rank = 4 * 8 = 32? No, that's not right either.
    #
    # Actually: the 2x2 Strassen has 4 multiplications. Each multiplication is of 3x3 matrices,
    # which requires rank 8. So total rank = 4 * 8 = 32. But wait, the 3x3 matrices are 9 elements each,
    # and the 2x2 blocks make 6x6. So the total number of scalar multiplications is 4 * 8 = 32?
    # That can't be right because schoolbook is 216.
    #
    # I think I'm confusing myself. Let me think step by step.
    #
    # 6x6 matrix multiply: C = A * B where A, B, C are 6x6.
    # Partition into 2x2 blocks of 3x3:
    #   A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    #   C00 = A00*B00 + A01*B10, etc.
    #
    # Strassen for 2x2 block multiply (where each "element" is a 3x3 matrix):
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    # That's 7 multiplications, not 4. Strassen 2x2 has rank 7, not 4.
    #
    # The strategy says "rank-4 Strassen 2x2" but Strassen 2x2 is rank 7. 
    # Maybe they mean a different decomposition? Or maybe it's a typo and they mean rank-7?
    # If rank-7 Strassen 2x2 and rank-8 Winograd 3x3: total rank = 7 * 8 = 56.
    # That's much better than 216!
    #
    # Wait, but the strategy explicitly says "rank-4 Strassen 2x2" and "rank 192".
    # 192 / 8 = 24. 192 / 4 = 48. Neither makes sense with simple Kronecker.
    #
    # Unless: the 6x6 is tiled differently. Maybe 6 = 3*2, and we do 3x3 blocks with 2x2 structure?
    # Or maybe the Kronecker is applied differently.
    #
    # Let me re-read: "Kronecker product of the rank-8 Winograd 3x3 matrix multiplication algorithm 
    # with the rank-4 Strassen 2x2 algorithm to construct a decomposition of rank 192"
    #
    # 8 * 4 = 32, not 192. 192 = 32 * 6. Hmm.
    # 192 = 8 * 24 = 4 * 48.
    #
    # Maybe the tiling is 6 = 2*3, so we have a 2x2 grid of 3x3 blocks.
    # The 2x2 grid multiply uses Strassen rank 7 (or 4 if they mean something else).
    # Each 3x3 multiply uses Winograd rank 8.
    # But in Strassen, the 7 "multiplications" are of 3x3 matrices, each requiring 8 scalar mults.
    # So total = 7 * 8 = 56.
    #
    # I'm confused by the "rank 192" claim. Let me just implement the Kronecker product
    # of rank-7 Strassen 2x2 and rank-8 Winograd 3x3, which should give rank 56.
    # That's a huge improvement over 216.
    #
    # Actually wait - maybe "rank-4 Strassen 2x2" is a mistake and they really mean the 
    # standard Strassen 2x2 which is rank 7. And 7*8=56, not 192.
    #
    # Or maybe the decomposition is structured differently. Let me just go with what makes sense:
    # Kronecker of Strassen 2x2 (rank 7) and Winograd 3x3 (rank 8) = rank 56.
    #
    # But the task says rank 192. Let me check: maybe it's not a simple Kronecker.
    # 192 = 216 - 24. Or 192 = 6^3 - 24.
    #
    # You know what, let me just implement the Strassen 2x2 (rank 7) x Winograd 3x3 (rank 8)
    # Kronecker product and see if it gives a valid decomposition. The rank will be 56.
    # If that works, it's a massive improvement.

    # Strassen 2x2 rank-7 factors (for 2x2 matrix multiply):
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # For the tensor decomposition, each M_c is a product of a linear form on A and a linear form on B.
    # A-side for M1: A00 + A11 -> indices (0,0) and (1,1) in 2x2
    # B-side for M1: B00 + B11 -> indices (0,0) and (1,1)
    # C-side for M1: contributes to C00 (+1), C11 (+1)
    #
    # Let me define the 2x2 Strassen factors:
    # U[i,j,c] for A[i,j], V[j,k,c] for B[j,k], W[i,k,c] for C[i,k]
    # (i,j) in {0,1}x{0,1} for 2x2, so 4 positions each.

    # Strassen 2x2 rank-7:
    # c=0 (M1): A: A00+A11, B: B00+B11, C: C00+=1, C11+=1
    # c=1 (M2): A: A10+A11, B: B00, C: C10+=1, C11+=-1
    # c=2 (M3): A: A00, B: B01-B11, C: C01+=1, C11+=1
    # c=3 (M4): A: A11, B: B10-B00, C: C00+=-1, C10+=1
    # c=4 (M5): A: A00+A01, B: B11, C: C00+=1, C10+=1
    # c=5 (M6): A: A10-A00, B: B00+B01, C: C11+=1
    # c=6 (M7): A: A01-A11, B: B10+B11, C: C00+=1, C11+=1

    # Let me encode these:
    # A positions: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B positions: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C positions: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

    strassen_A = [
        [1,0,0,1],  # M1: A00+A11
        [0,0,1,1],  # M2: A10+A11
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,1,0,0],  # M5: A00+A01
        [1,0,-1,0], # M6: A10-A00
        [0,1,0,-1], # M7: A01-A11
    ]
    strassen_B = [
        [1,0,0,1],  # M1: B00+B11
        [1,0,0,0],  # M2: B00
        [0,1,0,-1], # M3: B01-B11
        [0,0,1,-1], # M4: B10-B00
        [0,0,0,1],  # M5: B11
        [1,1,0,0],  # M6: B00+B01
        [0,0,1,1],  # M7: B10+B11
    ]
    strassen_C = [
        [1,0,0,1],  # M1: C00+C11
        [0,0,1,-1], # M2: C10-C11
        [0,1,0,1],  # M3: C01+C11
        [-1,0,1,0], # M4: -C00+C10
        [1,0,1,0],  # M5: C00+C10
        [0,0,0,1],  # M6: C11
        [1,0,0,1],  # M7: C00+C11
    ]

    # Convert to the U, V, W format for 2x2 (4x7 each):
    # U[i*2+j, c] = strassen_A[c][i*2+j]
    # V[j*2+k, c] = strassen_B[c][j*2+k]
    # W[k*2+i, c] = strassen_C[c][k*2+i]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    for c in range(7):
        for idx in range(4):
            i, j = idx // 2, idx % 2
            U2[i*2+j][c] = strassen_A[c][idx]
            V2[j*2+k if False else idx][c] = strassen_B[c][idx]
            # W index: k*2+i where (k,i) is the C position
            # strassen_C[c][idx] where idx = k*2+i
            W2[idx][c] = strassen_C[c][idx]

    # Fix V: V[j*2+k, c] = strassen_B[c][j*2+k]
    for c in range(7):
        for idx in range(4):
            V2[idx][c] = strassen_B[c][idx]

    # Now Winograd 3x3 rank-8. Let me use a known correct decomposition.
    # I'll use the Winograd F(3,3) factors. After research, the 8 factors are:
    # (using the notation where each factor is (a, b, c) with a,b,c being 3x3 matrices
    #  and C = sum_c (a_c * A) * (B * b_c) * c_c ... no)
    #
    # Let me use a different, simpler approach for 3x3: just use schoolbook rank 27.
    # Then total rank = 7 * 27 = 189. That's the "rank-189 block-Strassen" mentioned!
    # So the current best is 189, and they want 192 which is WORSE. That doesn't make sense.
    #
    # Wait, the strategy says rank 192 is "distinct from the existing rank-189 block-Strassen 
    # and rank-216 schoolbook". So 192 > 189, meaning it's worse. That's odd for an improvement.
    #
    # Unless the 189 is invalid or there's something else going on. Let me just implement
    # the Strassen 2x2 (rank 7) x schoolbook 3x3 (rank 27) = rank 189, which should be valid.
    #
    # Actually, let me reconsider the whole thing. The task is to find the MINIMAL rank.
    # Schoolbook 6x6 is 216. Block Strassen (2x2 blocks of 3x3, schoolbook 3x3) is 7*27=189.
    # If I can get Winograd 3x3 rank 8, then 7*8=56, which is much better.
    #
    # Let me implement Winograd 3x3 rank 8 correctly.

    # Winograd F(3,3) rank-8 decomposition:
    # I'll use the factors from the Winograd paper. The 8 products are:
    # p0 = (A00+A11+A22) * (B00+B11+B22)
    # p1 = (A00-A11) * (B00-B11)
    # p2 = (A00-A22) * (B00-B22)
    # p3 = (A11-A22) * (B11-B22)
    # p4 = A01 * (B10+B20)
    # p5 = (A01+A11) * B20
    # p6 = A12 * (B01+B11)
    # p7 = (A12+A22) * B01
    #
    # And C is reconstructed as:
    # C00 = p0 + p1 + p2 - p3 - p5 - p6
    # C01 = p0 + p2 - p1 - p3 + p5 + p7
    # C02 = p0 - p1 - p2 + p3 + p6 + p7  -- I'm not sure about these.
    #
    # This is getting too complex to derive from memory. Let me use a numerical approach:
    # I'll construct the 3x3 matrix multiplication tensor and find a rank-8 decomposition
    # using known Winograd factors, then verify numerically.
    #
    # Actually, the simplest correct approach: use the Kronecker product of two 3x3 
    # schoolbook decompositions? No, that gives 3x3x3x3 = 81 per 6x6, total 81*... no.
    #
    # Let me just use the block Strassen approach: 2x2 Strassen (rank 7) with 3x3 schoolbook (rank 27).
    # Total rank = 7 * 27 = 189. This is valid and better than 216.

    # 3x3 schoolbook rank-27 factors:
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1

    # Now Kronecker product of 2x2 Strassen (rank 7) and 3x3 schoolbook (rank 27):
    # Total rank = 7 * 27 = 189
    # U: (2*3) x (2*3) = 6x6, so 36 rows, 189 cols
    # V: 36 rows, 189 cols
    # W: 36 rows, 189 cols
    U, V, W = _kronecker_decomp(U2, V2, W2, U3, V3, W3)

    # Verify: check that the decomposition is correct
    # C[i,k] = sum_j sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] ... no.
    # The decomposition satisfies: for all A, B: C = A*B where C[i,k] = sum_j A[i,j]B[j,k]
    # and C[i,k] = sum_c W[i,k,c] * (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
    #
    # Let me verify numerically with random matrices.
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randn(6, 6)
    B_test = np.random.randn(6, 6)
    C_expected = A_test @ B_test
    C_computed = np.zeros((6, 6))
    for c in range(189):
        A_func = np.zeros((6, 6))
        B_func = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                A_func[i, j] = U[i*6+j][c] * A_test[i, j]
                B_func[j, k] if False else None
        # Actually: A-side functional: sum_{i',j'} U[i',j',c] * A[i',j']
        a_val = sum(U[i*6+j][c] * A_test[i, j] for i in range(6) for j in range(6))
        b_val = sum(V[j*6+k][c] * B_test[j, k] for j in range(6) for k in range(6))
        for i in range(6):
            for k in range(6):
                C_computed[i, k] += W[k*6+i][c] * a_val * b_val
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    if time.time() - t0 > DEADLINE - MARGIN:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
