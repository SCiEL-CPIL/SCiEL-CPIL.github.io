"""Strassen-based rank-49 tensor decomposition for 4x4 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Build rank-49 Strassen decomposition for 4x4 matrices.
    # We use the tensor product of the rank-7 Strassen scheme for 2x2 with itself.
    # This gives 7*7 = 49 terms.
    
    n, m, p = DIMENSIONS  # 4, 4, 4

    # Strassen 2x2 scheme: 7 products M1..M7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]

    # For 4x4, we split into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2

    # The 49 products come from all pairs (s, t) where s,t in 1..7
    # For block C11: uses products from Strassen on (A11,A21) x (B11,B21)
    # Actually, the standard approach:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # Using Strassen on the block level:
    # Let S1 = A11+A22, S2 = A21+A22, S3 = A11-A21, S4 = A12-A22
    # T1 = B11+B22, T2 = B11-B22, T3 = B12+B21, T4 = B21+B22, T5 = B12-B22, T6 = B11+B12, T7 = B21-B22
    # M1 = S1*T1
    # M2 = S2*B11
    # M3 = S3*T3
    # M4 = S4*T4
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21-B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # Each of these 7 block-products (2x2 times 2x2) can itself be done with 7 scalar products.
    # So total = 7 * 7 = 49 scalar products.

    # Indexing:
    # A[i,j] with i,j in 0..3
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B.

    # We need to build U (16 x 49), V (16 x 49), W (16 x 49)
    # U[row, r] = coefficient of A[row//4*4 + row%4] in product r
    # V[col, r] = coefficient of B[col//4*4 + col%4] in product r  
    # W[out, r] = coefficient of C[out//4*4 + out%4] in product r

    # Let's define the 7 block-level linear forms first.
    # Block indices: (bi, bj) where bi,bj in {0,1} for 2x2 block position
    # Within block: (pi, pj) where pi,pj in {0,1} for 2x2 element position

    # A element (i,j): block (i//2, j//2), within-block (i%2, j%2)
    # B element (j,k): block (j//2, k//2), within-block (j%2, k%2)
    # C element (i,k): block (i//2, k//2), within-block (i%2, k%2)

    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49, but W[k*n+i] convention

    # Wait, W[k*n+i, r] means W is indexed by (k, i) where k is col of B / row of C?
    # C[i,k] = sum_j A[i,j] B[j,k]
    # W index: k*n + i, so W[k, i] corresponds to C[i, k]
    # So W[row, r] where row = k*n + i, and this is the coefficient for C[i,k]

    # Let me re-examine: W has shape (p*n, rank) = (16, 49)
    # W[k*n + i, r] is the coefficient of product r in C[i,k]

    # Define the 7 Strassen factors for 2x2:
    # Factor s (0-indexed) on left (A side):
    # s=0: (a+d)(e+h) -> A part: a+d, B part: e+h
    # s=1: (c+d)e -> A part: c+d, B part: e
    # s=2: a(f-h) -> A part: a, B part: f-h
    # s=3: d(g-e) -> A part: d, B part: g-e
    # s=4: (a+b)h -> A part: a+b, B part: h
    # s=5: (c-a)(e+f) -> A part: c-a, B part: e+f
    # s=6: (b-d)(g+h) -> A part: b-d, B part: g+h

    # For 2x2 A = [[a,b],[c,d]], indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # For 2x2 B = [[e,f],[g,h]], indices: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]

    # Left factors (coefficients for a,b,c,d):
    left_factors = [
        [1, 0, 0, 1],   # s=0: a+d
        [0, 0, 1, 1],   # s=1: c+d
        [1, 0, 0, 0],   # s=2: a
        [0, 0, 0, 1],   # s=3: d
        [1, 1, 0, 0],   # s=4: a+b
        [-1, 0, 1, 0],  # s=5: c-a
        [0, 1, 0, -1],  # s=6: b-d
    ]

    # Right factors (coefficients for e,f,g,h):
    right_factors = [
        [1, 0, 0, 1],   # s=0: e+h
        [1, 0, 0, 0],   # s=1: e
        [0, 1, 0, -1],  # s=2: f-h
        [0, -1, 1, 0],  # s=3: g-e
        [0, 0, 0, 1],   # s=4: h
        [1, 1, 0, 0],   # s=5: e+f
        [0, 0, 1, 1],   # s=6: g+h
    ]

    # Block-level Strassen:
    # C11 = M1 + M4 - M5 + M7  (block products indexed by s)
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Block output coefficients: for each block position (cbi, cbj) in C,
    # and each Strassen factor s, what's the coefficient?
    # cbi, cbj in {0,1} for 2x2 block position in C

    block_coeffs = {}
    # C11: M1+M4-M5+M7
    block_coeffs[(0, 0)] = {0: 1, 3: 1, 4: -1, 6: 1}
    # C12: M3+M5
    block_coeffs[(0, 1)] = {2: 1, 4: 1}
    # C21: M2+M4
    block_coeffs[(1, 0)] = {1: 1, 3: 1}
    # C22: M1-M2+M3+M6
    block_coeffs[(1, 1)] = {0: 1, 1: -1, 2: 1, 5: 1}

    # Now for each block product s (at block level), we do a 2x2 Strassen.
    # The block-level product s involves specific block combinations.
    # Let's identify which A-blocks and B-blocks are involved in each s.

    # s=0: (A11+A22)(B11+B22)
    #   A blocks: A11 (coeff 1), A22 (coeff 1)
    #   B blocks: B11 (coeff 1), B22 (coeff 1)
    # s=1: (A21+A22)B11
    #   A blocks: A21 (coeff 1), A22 (coeff 1)
    #   B blocks: B11 (coeff 1)
    # s=2: (A11-A21)(B12+B21)
    #   A blocks: A11 (coeff 1), A21 (coeff -1)
    #   B blocks: B12 (coeff 1), B21 (coeff 1)
    # s=3: (A12-A22)(B21+B22)
    #   A blocks: A12 (coeff 1), A22 (coeff -1)
    #   B blocks: B21 (coeff 1), B22 (coeff 1)
    # s=4: (A11+A12)B22
    #   A blocks: A11 (coeff 1), A12 (coeff 1)
    #   B blocks: B22 (coeff 1)
    # s=5: (A21-A11)(B11+B12)
    #   A blocks: A21 (coeff 1), A11 (coeff -1)
    #   B blocks: B11 (coeff 1), B12 (coeff 1)
    # s=6: (A12-A22)(B21-B22)
    #   A blocks: A12 (coeff 1), A22 (coeff -1)
    #   B blocks: B21 (coeff 1), B22 (coeff -1)

    # Define block involvement for each s:
    # a_blocks[s] = list of (block_idx, coeff) for A
    # b_blocks[s] = list of (block_idx, coeff) for B
    a_blocks = [
        [(0, 1), (3, 1)],      # s=0: A11, A22
        [(2, 1), (3, 1)],      # s=1: A21, A22
        [(0, 1), (2, -1)],     # s=2: A11, -A21
        [(1, 1), (3, -1)],     # s=3: A12, -A22
        [(0, 1), (1, 1)],      # s=4: A11, A12
        [(2, 1), (0, -1)],     # s=5: A21, -A11
        [(1, 1), (3, -1)],     # s=6: A12, -A22
    ]
    b_blocks = [
        [(0, 1), (3, 1)],      # s=0: B11, B22
        [(0, 1)],              # s=1: B11
        [(1, 1), (2, 1)],      # s=2: B12, B21
        [(2, 1), (3, 1)],      # s=3: B21, B22
        [(3, 1)],              # s=4: B22
        [(0, 1), (1, 1)],      # s=5: B11, B12
        [(2, 1), (3, -1)],     # s=6: B21, -B22
    ]

    # Block index mapping: block_idx -> (bi, bj) where bi is row block, bj is col block
    # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
    # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
    # C blocks: 0=C11, 1=C12, 2=C21, 3=C22

    block_pos = {
        0: (0, 0),  # *11
        1: (0, 1),  # *12
        2: (1, 0),  # *21
        3: (1, 1),  # *22
    }

    # For each block-level Strassen factor s (0..6), and for each of the 7 inner Strassen factors t (0..6):
    # The product index r = s*7 + t
    # The A-side: for each A block involved in s with coeff ca, and for each 2x2 element in that block,
    #   apply the left factor t to get the coefficient.
    # Similarly for B-side.
    # The C-side: for each C block (cbi, cbj) with block coeff cc for factor s, and for each 2x2 element,
    #   apply the right-side combination to get the coefficient.

    # Actually, the inner Strassen for a 2x2 product:
    # If we have 2x2 matrix X (from A blocks) and 2x2 matrix Y (from B blocks),
    # their product Z = X*Y is computed with 7 products M1..M7.
    # Z[i,j] = sum_t M_t * out_coeff[i,j,t]
    # where M_t uses left_factors[t] on X and right_factors[t] on Y.

    # So for the full 49-term decomposition:
    # Term r = s*7 + t:
    #   U side: X is a linear combination of A blocks. Each A block (bi,bj) with coeff ca
    #           contributes ca * left_factors[t][pi,pj] to U[A[bi*2+pi, bj*2+pj], r]
    #   V side: Y is a linear combination of B blocks. Each B block (bi,bj) with coeff cb
    #           contributes cb * right_factors[t][pi,pj] to V[B[bi*2+pi, bj*2+pj], r]
    #   W side: Z contributes to C blocks. Each C block (cbi,cbj) with block coeff cc for s
    #           contributes cc * out_coeff[pi,pj,t] to W[C[cbi*2+pi, cbj*2+pj], r]
    #           where out_coeff[pi,pj,t] is from the 2x2 Strassen output formula.

    # Output coefficients for 2x2 Strassen:
    # Z[0,0] = M1+M4-M5+M7 -> out[0,0] = {0:1, 3:1, 4:-1, 6:1}
    # Z[0,1] = M3+M5       -> out[0,1] = {2:1, 4:1}
    # Z[1,0] = M2+M4       -> out[1,0] = {1:1, 3:1}
    # Z[1,1] = M1-M2+M3+M6 -> out[1,1] = {0:1, 1:-1, 2:1, 5:1}

    out_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0, 1): {2: 1, 4: 1},
        (1, 0): {1: 1, 3: 1},
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1},
    }

    # Now build U, V, W
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            if time.time() > deadline:
                break

            # U side
            for (ab, ca) in a_blocks[s]:
                ab_i, ab_j = block_pos[ab]
                for pi in range(2):
                    for pj in range(2):
                        idx_a = (ab_i * 2 + pi) * m + (ab_j * 2 + pj)
                        U[idx_a][r] += ca * left_factors[t][pi * 2 + pj]

            # V side
            for (bb, cb) in b_blocks[s]:
                bb_i, bb_j = block_pos[bb]
                for pi in range(2):
                    for pj in range(2):
                        idx_b = (bb_i * 2 + pi) * p + (bb_j * 2 + pj)
                        V[idx_b][r] += cb * right_factors[t][pi * 2 + pj]

            # W side
            for (cbi, cbj), s_coeffs in block_coeffs.items():
                if s not in s_coeffs:
                    continue
                cc = s_coeffs[s]
                for pi in range(2):
                    for pj in range(2):
                        if t not in out_coeffs[(pi, pj)]:
                            continue
                        oc = out_coeffs[(pi, pj)][t]
                        idx_c = (cbi * 2 + pi) * n + (cbj * 2 + pj)
                        # W[k*n + i, r] where i is row of C, k is col of C
                        # C[i,k] where i = cbi*2+pi, k = cbj*2+pj
                        w_idx = (cbj * 2 + pj) * n + (cbi * 2 + pi)
                        W[w_idx][r] += cc * oc

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
