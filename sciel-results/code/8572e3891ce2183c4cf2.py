"""Matrix multiplication 2,2,13: rank-minimal exact tensor decomposition."""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 13

    # ------------------------------------------------------------------
    # Strategy: split the 2x13 matrix B into blocks and use Strassen for
    # 2x2x2 blocks where possible.
    #
    # B is 2x13. We can group the 13 columns as:
    #   - 3 groups of 4 columns = 12 columns  -> each 4-col group can be
    #     split into two 2-col sub-groups
    #   - 1 remaining column
    #
    # Alternative: 13 = 2 + 2 + 2 + 2 + 2 + 2 + 1
    # That's 6 groups of 2 + 1 group of 1.
    # Each 2x2x2 block uses Strassen rank 7.
    # Each 2x2x1 block uses rank 4.
    # Total rank = 6*7 + 4 = 42 + 4 = 46.
    #
    # Can we do better?  13 = 3 + 3 + 3 + 3 + 1
    # 2x2x3 blocks: rank of 2x2x3 matrix mult tensor is 10 (known result).
    # 4 blocks of 2x2x3 + 1 block of 2x2x1: 4*10 + 4 = 44.
    #
    # Or: 13 = 3 + 3 + 3 + 2 + 2: 3*10 + 2*7 = 30+14 = 44.
    # Or: 13 = 4 + 3 + 3 + 3: 2x2x4 rank? 2x2x4 = two 2x2x2 blocks -> 2*7=14.
    #   14 + 3*10 = 14+30 = 44.
    #
    # Let's try: 13 = 2+2+2+2+2+2+1 -> 6*7+4 = 46 (Strassen on 2-col blocks)
    # 13 = 3+3+3+3+1 -> 4*10+4 = 44
    # 13 = 3+3+3+2+2 -> 3*10+2*7 = 44
    #
    # We need the rank-10 decomposition of the 2x2x3 multiplication tensor.
    #
    # 2x2x3 tensor: A is 2x2, B is 2x3, C = A*B is 2x3.
    # Indices: A[i,j] i,j in {0,1}, B[j,k] j in {0,1}, k in {0,1,2},
    #          C[i,k] = sum_j A[i,j]*B[j,k].
    #
    # Known: rank of 2x2x3 is 10.
    # One explicit rank-10 decomposition (from Landsberg's work / standard):
    # We'll use the following 10 scalar products:
    #
    # Let A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[a*e+b*h, a*f+b*i, a*g+b*j],
    #      [c*e+d*h, c*f+d*i, c*g+d*j]]
    #
    # 10 products:
    # P1 = a*e
    # P2 = b*h
    # P3 = c*e
    # P4 = d*h
    # P5 = (a+d)*(e+h)
    # P6 = (a+b)*(f+i)
    # P7 = (c+d)*(f+i)
    # P8 = (a-b)*(g+j)
    # P9 = (c-d)*(g+j)
    # P10 = (a+c)*(e+f+g) + (b+d)*(h+i+j)?  -- need to verify.
    #
    # Actually let me use a known valid rank-10 decomposition.
    # The tensor T_{2,2,3} has border rank 10 and rank 10.
    #
    # Let me use a construction: view 2x2x3 as 2x2x2 + 2x2x1.
    # 2x2x2 has rank 7 (Strassen), 2x2x1 has rank 4.
    # But that gives 11, not 10. So the 10-rank decomposition is non-trivial.
    #
    # I'll use a known explicit decomposition from the literature.
    # Reference: "On the rank of small matrix multiplication" or similar.
    #
    # Actually, let me use a simpler approach: just use 2-col blocks (Strassen)
    # and 1-col block, giving rank 46. This is guaranteed correct.
    #
    # For rank 44, I need valid 2x2x3 rank-10 decompositions.
    # Let me try to construct one carefully.
    #
    # 2x2x3: A(2x2) * B(2x3) = C(2x3)
    # 6 entries of A, 6 entries of B, 6 entries of C.
    #
    # Let me try the following 10 products:
    # p1  = A[0,0]*B[0,0]          # a*e
    # p2  = A[0,1]*B[1,0]          # b*h
    # p3  = A[1,0]*B[0,0]          # c*e
    # p4  = A[1,1]*B[1,0]          # d*h
    # p5  = (A[0,0]+A[1,1])*(B[0,0]+B[1,0])   # (a+d)(e+h)
    # p6  = (A[0,0]+A[0,1])*(B[0,1]+B[1,1])   # (a+b)(f+i)
    # p7  = (A[1,0]+A[1,1])*(B[0,1]+B[1,1])   # (c+d)(f+i)
    # p8  = (A[0,0]-A[0,1])*(B[0,2]+B[1,2])   # (a-b)(g+j)
    # p9  = (A[1,0]-A[1,1])*(B[0,2]+B[1,2])   # (c-d)(g+j)
    # p10 = ?
    #
    # C[0,0] = a*e + b*h = p1 + p2
    # C[1,0] = c*e + d*h = p3 + p4
    # C[0,1] = a*f + b*i
    # C[1,1] = c*f + d*i
    # C[0,2] = a*g + b*j
    # C[1,2] = c*g + d*j
    #
    # From p6 = (a+b)(f+i) = a*f + a*i + b*f + b*i
    # From p7 = (c+d)(f+i) = c*f + c*i + d*f + d*i
    #
    # C[0,1] = a*f + b*i
    # C[1,1] = c*f + d*i
    #
    # p6 - p1 - p2 doesn't directly give a*f + b*i.
    #
    # Let me try a different set. Use the fact that:
    # a*f + b*i = (a+b)(f+i) - a*i - b*f
    # But a*i and b*f aren't single products in our set.
    #
    # Alternative: use p6 = (a+b)*(f+i) and p7 = (c+d)*(f+i).
    # a*f + b*i = p6 - a*i - b*f.  Still stuck.
    #
    # Let me try:
    # p6 = a*f + b*i  (but this is 2 products, not 1)
    #
    # OK, I think I need a proper rank-10 decomposition. Let me look at this differently.
    #
    # Actually, the simplest safe approach: use 2-col Strassen blocks.
    # 13 = 2*6 + 1, so 6 Strassen blocks (rank 7 each) + 1 rank-4 block = 46.
    #
    # Let me just implement rank 46 correctly.
    # ------------------------------------------------------------------

    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix mult using Strassen, rank 7.
        U has 4 rows (A entries), V has 4 rows (B entries), W has 4 rows (C entries).
        Convention: U[i,j] for A[i,j], V[j,k] for B[j,k], W[k,i] for C[i,k]
        Wait: the task says W[k*n+i, r]. For n=2,m=2,p=2 sub-block:
        W index = k*2 + i where k is col of B (=row of C col index), i is row of A (=row of C row index).
        So W row index = k*n_sub + i where n_sub=2.
        """
        # Strassen for 2x2:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (d-b)(f+g)
        # M5 = (a-e)(f+h)
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C[0,0] = M1 + M4 - M5 - M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M6 + M7
        #
        # A = [[a, b], [c, d]] indexed as [0,0],[0,1],[1,0],[1,1]
        # B = [[e, f], [g, h]] indexed as [0,0],[0,1],[1,0],[1,1]
        # C = [[C00, C01], [C10, C11]] indexed by (k*n+i):
        #   C00 -> k=0,i=0 -> idx 0
        #   C01 -> k=1,i=0 -> idx 1
        #   C10 -> k=0,i=1 -> idx 2
        #   C11 -> k=1,i=1 -> idx 3

        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1 = (a+d)(e+h)
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1; V[1][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # M2 = (c+d)(e)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # M3 = (a+b)(h)
        U[0][2] = 1; U[1][2] = 1
        V[1][2] = 1
        W[1][2] = 1

        # M4 = (d-b)(f+g)
        U[1][3] = -1; U[3][3] = 1
        V[0][3] = 1; V[1][3] = 1
        W[0][3] = 1; W[2][3] = 1

        # M5 = (a-e)(f+h)
        U[0][4] = 1
        V[0][4] = -1; V[1][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # M6 = (c-a)(e+f)
        U[0][5] = -1; U[2][5] = 1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # M7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[0][6] = 1  # g is B[1,0]=V[1][0], h is B[1,1]=V[1][1]
        # Wait: B is 2x2. B[j,k] where j is row, k is col.
        # V row index = j*p_sub + k = j*2+k.
        # B[0,0]=e -> V[0], B[0,1]=f -> V[1], B[1,0]=g -> V[2], B[1,1]=h -> V[3].
        # Let me redo with correct indexing.
        #
        # A[i,j]: U row = i*m_sub + j = i*2+j
        #   A[0,0]=a -> U[0], A[0,1]=b -> U[1], A[1,0]=c -> U[2], A[1,1]=d -> U[3]
        # B[j,k]: V row = j*p_sub + k = j*2+k
        #   B[0,0]=e -> V[0], B[0,1]=f -> V[1], B[1,0]=g -> V[2], B[1,1]=h -> V[3]
        # C[i,k]: W row = k*n_sub + i = k*2+i
        #   C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]

        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1 = (a+d)(e+h): U[0],U[3] * V[0],V[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C00, C11

        # M2 = (c+d)(e): U[2],U[3] * V[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1  # C10, -C11

        # M3 = (a+b)(h): U[0],U[1] * V[3]
        U[0][2] = 1; U[1][2] = 1
        V[3][2] = 1
        W[1][2] = 1  # C01

        # M4 = (d-b)(f+g): U[1],U[3] * V[1],V[2]
        U[1][3] = -1; U[3][3] = 1
        V[1][3] = 1; V[2][3] = 1
        W[0][3] = 1; W[2][3] = 1  # C00, C10

        # M5 = (a-e)(f+h): U[0] * V[0],V[1],V[3]
        U[0][4] = 1
        V[0][4] = -1; V[1][4] = 1; V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1  # -C00, C01

        # M6 = (c-a)(e+f): U[0],U[2] * V[0],V[1]
        U[0][5] = -1; U[2][5] = 1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C11

        # M7 = (b-d)(g+h): U[1],U[3] * V[2],V[3]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[1][6] = 1  # -C00, C01

        return U, V, W

    def schoolbook_2x2x1():
        """2x2x1: A(2x2)*B(2x1)=C(2x1). Rank 4.
        U: 4 rows, V: 2 rows, W: 2 rows.
        W index = k*n_sub + i, but p_sub=1 so k=0 always.
        W row = i (since k=0, k*2+i = i).
        """
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        for j in range(2):
            U[j][j] = 1
            V[j][j] = 1
            W[0][j] = 1  # C[0,0] gets A[0,j]*B[j,0]
        for j in range(2):
            col = 2 + j
            U[2 + j][col] = 1
            V[j][col] = 1
            W[1][col] = 1  # C[1,0] gets A[1,j]*B[j,0]
        return U, V, W

    # Build the full decomposition: 6 Strassen blocks (2 cols each) + 1 schoolbook block (1 col)
    U, V, W = strassen_2x2x2()
    U_sb, V_sb, W_sb = schoolbook_2x2x1()

    full_U = [row[:] for row in U]
    full_V = [row[:] for row in V]
    full_W = [row[:] for row in W]

    # Each Strassen block handles 2 columns of B, contributing 7 rank terms.
    # Block b handles columns 2b and 2b+1 of B (0-indexed), for b in 0..5.
    # The last block (b=6) handles column 12 only, using schoolbook rank-4.

    rank_so_far = 7
    for b in range(1, 6):
        # Append 7 more columns for the next Strassen block
        for r in range(4):
            full_U[r].append(0)
            full_W[r].append(0)
        for r in range(4):
            full_V[r].append(0)
        for c in range(7):
            for r in range(4):
                full_U[r].append(0)
                full_W[r].append(0)
            for r in range(4):
                full_V[r].append(0)

    # Hmm, this is getting complicated. Let me build it more carefully.

    # Total rank = 6*7 + 4 = 46
    total_rank = 46
    U_full = [[0] * total_rank for _ in range(4)]   # 2x2 A entries
    V_full = [[0] * total_rank for _ in range(2 * 13)]  # 2x13 B entries
    W_full = [[0] * total_rank for _ in range(13 * 2)]   # 13x2 C entries (W[k*n+i])

    # For each 2-col block b (columns 2b, 2b+1), columns 0..11:
    # Strassen terms go in rank columns b*7 .. b*7+6
    sU, sV, sW = strassen_2x2x2()
    for b in range(6):
        col_offset = b * 7
        for r in range(4):
            for c in range(7):
                U_full[r][col_offset + c] = sU[r][c]
        # V: for this block, B rows map to global V rows:
        #   local B[0,0] -> global V row 0*13 + 2b = 2b
        #   local B[0,1] -> global V row 0*13 + 2b+1 = 2b+1
        #   local B[1,0] -> global V row 1*13 + 2b = 13+2b
        #   local B[1,1] -> global V row 1*13 + 2b+1 = 13+2b+1
        v_map = [2*b, 2*b+1, 13+2*b, 13+2*b+1]
        for r in range(4):
            for c in range(7):
                V_full[v_map[r]][col_offset + c] = sV[r][c]
        # W: local C[i,k] -> global W row k*n + i = k*2 + i
        #   local C[0,0] -> W[0], local C[0,1] -> W[1], local C[1,0] -> W[2], local C[1,1] -> W[3]
        # But global: C row i, col k. W index = k*n + i = k*2 + i.
        # However, the column k in the sub-block corresponds to global column 2b+k.
        # W row = global_k * n + i = (2b+k)*2 + i.
        # Wait: W[k*n+i] where k is the output column index (0..12) and i is the output row (0..1).
        # So for sub-block column k_sub (0 or 1), global column = 2b + k_sub.
        # W row = (2b + k_sub) * 2 + i.
        w_map = {}
        for i in range(2):
            for k_sub in range(2):
                local_idx = k_sub * 2 + i  # local W row index
                global_idx = (2 * b + k_sub) * 2 + i
                w_map[local_idx] = global_idx
        for r in range(4):
            for c in range(7):
                W_full[w_map[r]][col_offset + c] = sW[r][c]

    # Last block: column 12, schoolbook 2x2x1
    col_offset = 6 * 7  # = 42
    sbU, sbV, sbW = schoolbook_2x2x1()
    for r in range(4):
        for c in range(4):
            U_full[r][col_offset + c] = sbU[r][c]
    # V mapping for column 12:
    #   local B[0,0] -> global V row 0*13+12 = 12
    #   local B[1,0] -> global V row 1*13+12 = 25
    v_map_sb = [12, 25]
    for r in range(2):
        for c in range(4):
            V_full[v_map_sb[r]][col_offset + c] = sbV[r][c]
    # W mapping: local C[0,0] -> global W row 12*2+0 = 24, local C[1,0] -> 12*2+1 = 25
    for r in range(2):
        for c in range(4):
            W_full[24 + r][col_offset + c] = sbW[r][c]

    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
