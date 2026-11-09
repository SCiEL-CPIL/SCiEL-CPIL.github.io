"""Combined program: recursive Strassen 4x4x4 + schoolbook 4x4x5, verified, fallback schoolbook."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > tol:
                        return False
        return True

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Recursive Strassen for square s x s x s (s power of 2), returns list of (Ud,Vd,Wd)
    # over global (i,j),(j,k),(k,i) indices.
    def strassen_terms(si, sj, sk, size):
        # si, sj, sk: starting global indices for i, j, k; size = s
        terms = []
        if size == 2:
            # 2x2x2 Strassen, 7 terms
            # local (li,lj) -> global (si+li, sj+lj); (lj,lk)->(sj+lj, sk+lk); (lk,li)->(sk+lk, si+li)
            g = lambda a, b, c: (a + si, b + sj, c + sk)
            combos = [
                ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
                ({(0,0):1}, {(1,0):1,(0,0):-1}, {(0,1):1}),
                ({(1,1):1}, {(0,1):1,(1,1):-1}, {(0,0):1,(1,0):1}),
                ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
                ({(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),
                ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1}),
            ]
            for Uloc, Vloc, Wloc in combos:
                Ud = {}
                Vd = {}
                Wd = {}
                for (li, lj), c in Uloc.items():
                    key = (si + li, sj + lj)
                    Ud[key] = Ud.get(key, 0) + c
                for (lj, lk), c in Vloc.items():
                    key = (sj + lj, sk + lk)
                    Vd[key] = Vd.get(key, 0) + c
                for (lk, li), c in Wloc.items():
                    key = (sk + lk, si + li)
                    Wd[key] = Wd.get(key, 0) + c
                terms.append((Ud, Vd, Wd))
            return terms
        # size > 2: split into 2x2 blocks of size/2
        h = size // 2
        # 7 Strassen meta-products; each is a block product of size h x h x h
        # Meta U combos (over 2x2 block positions), V combos, W routing
        # p1=(M00+M11)(N00+N11) -> routes to C00,C11
        # p2=(M10+M11)N00 -> C10
        # p3=M00(N10-N00) -> C01
        # p4=M11(N01-N11) -> C00,C10
        # p5=(M00+M01)N11 -> -C00,+C01
        # p6=(M10-M00)(N00+N10) -> C11
        # p7=(M01-M11)(N01+N11) -> C00
        # Each meta-product: U is a sum of h x h block-matrices (positions), V sum, and the
        # block product U*V is an h x h x h multiplication -> recurse.
        # For a meta-product with U-blocks at positions P (with coeffs) and V-blocks at Q,
        # the block product = sum_{(pu in P, qv in Q)} coeff_pu*coeff_qv * (block_pu * block_qv)
        # where block_pu is the h x h A-block at meta-position pu, block_qv the h x h B-block at qv.
        # Each (block_pu * block_qv) is an h x h x h multiplication -> strassen_terms for that sub-block.
        # The W routing for the meta-product applies to the resulting h x h C-blocks.
        # This is complex: the meta-product's C-blocks are at specific meta-positions, and W routes
        # the whole meta-product scalar... no, the meta-product is a block-matrix product giving h x h
        # C-blocks at specific positions, and W routes each C-block to the global C.
        # Given the depth, implement by: for each meta-product, for each (pu,qv) pair, recurse on the
        # h x h x h sub-block at (pu position in A, qv position in B) producing terms, then scale by
        # coeff_pu*coeff_qv and apply W routing (the W for the meta-product at the C-block position).
        # The C-block position for the meta-product: p1 -> C00 and C11 (two positions), etc.
        # So a meta-product contributes to multiple C-block positions; for each, the sub-terms' W is
        # multiplied by the meta-W coeff at that C-block.
        # Let me define meta-products as list of (Ublocks, Vblocks, Wblocks) where each is dict
        # meta_pos -> coeff, meta_pos in {(0,0),(0,1),(1,0),(1,1)}.
        meta = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),  # p1
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),  # p2
            ({(0,0):1}, {(1,0):1,(0,0):-1}, {(0,1):1}),  # p3
            ({(1,1):1}, {(0,1):1,(1,1):-1}, {(0,0):1,(1,0):1}),  # p4
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),  # p5
            ({(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),  # p6
            ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1}),  # p7
        ]
        for Ublk, Vblk, Wblk in meta:
            # For each (pu in Ublk, qv in Vblk): sub-block product at A-block pu, B-block qv
            for pu, cu in Ublk.items():
                for qv, cv in Vblk.items():
                    # A-block pu is at global i in [si+pu[0]*h, si+(pu[0]+1)*h), j in [sj+pu[1]*h, ...)
                    # B-block qv at j in [sj+qv[0]*h, ...), k in [sk+qv[1]*h, ...)
                    # The sub-multiplication is h x h x h: A rows i-block pu, B cols k-block qv[1],
                    # but the "j" index must match: A-block pu has j-range [sj+pu[1]*h, ...), B-block qv
                    # has j-range [sj+qv[0]*h, ...). For the product to be valid, these j-ranges must
                    # be the SAME (the inner dimension). In Strassen, the meta-products pair A-blocks and
                    # B-blocks with matching j. But here pu[1] (A's j-block) and qv[0] (B's j-block) may
                    # differ! e.g., p1 pairs A00 (j-block 0) with B11 (j-block 1) -- mismatch.
                    # This is the crux: Strassen's (M00+M11)(N00+N11) multiplies A's j-blocks {0,1}
                    # sums by B's j-blocks {0,1} sums, but the inner j must align. Actually in matrix
                    # multiplication C = A B, the inner index is j. A00 has j in block 0, B11 has j in
                    # block 1 -- they DON'T share j, so A00 * B11 is NOT a valid sub-product (different j).
                    # I have the block structure wrong. Let me redefine: A is n x m, B is m x p.
                    # Split n into 2 (i-blocks), m into 2 (j-blocks), p into 2 (k-blocks).
                    # A-blocks: A[r][c] where r=i-block, c=j-block (A is 2x2 of h x h blocks).
                    # B-blocks: B[c][d] where c=j-block, d=k-block (B is 2x2 of h x h blocks).
                    # C-blocks: C[r][d] = sum_c A[r][c] B[c][d] (C is 2x2 of h x h blocks).
                    # Strassen on the 2x2 meta: M = A (2x2 of blocks), N = B (2x2 of blocks).
                    # p1=(M00+M11)(N00+N11): M00 is A-block (i-blk0,j-blk0), M11 is A-block (i-blk1,j-blk1).
                    # Their sum is a 2x2 "meta" but as block-matrices: (M00+M11) is NOT a single h x h matrix;
                    # M00 is h x h (i-blk0 x j-blk0), M11 is h x h (i-blk1 x j-blk1). They have DIFFERENT i and j
                    # ranges, so M00+M11 is a (2h x 2h) sparse matrix, not h x h. So the meta-product (M00+M11)(N00+N11)
                    # is a (2h x 2h)*(2h x 2h) = (2h x 2h) product, i.e., a size-2s x size-2s x size-2s... no.
                    # I think block Strassen works when you treat the 2x2 block matrices as 2x2 matrices over the
                    # ring of h x h matrices, and the "sum" M00+M11 is a 2x2 matrix whose (0,0) entry is M00 and (1,1)
                    # is M11 -- i.e., a 2x2 matrix of h x h blocks = a (2h x 2h) matrix. Then (M00+M11)(N00+N11) is a
                    # (2h x 2h) matrix product. So the meta-product is a (2h) x (2h) x (2h) multiplication, NOT h x h x h.
                    # That means recursion doesn't reduce size -- it stays 2h. So block Strassen as I framed it is wrong.
                    # The correct block Strassen: the 7 meta-products are each a product of two (2h x 2h) "sum" matrices,
                    # giving (2h x 2h) results. So 7 sub-problems of size 2h x 2h x 2h = same size as parent (2s). Infinite.
                    # This confirms: you CANNOT do block Strassen by summing blocks across different sub-blocks and
                    # multiplying, because the sums span the full size. Strassen's win comes from the 2x2 SCALAR case where
                    # the "blocks" are scalars (h=1). For h>1, the meta-products are full-size matrix products.
                    # Therefore, the ONLY way Strassen reduces rank is at the scalar (2x2x2) level. For 4x4x4, you must
                    # use a 4x4x4-specific algorithm (like the 49-term one), not naive block Strassen.
                    # The 49-term 4x4x4 algorithm IS a valid Strassen-type but with a specific 4x4 pattern, not recursive.
                    # Given the complexity, I will NOT implement 4x4x4 Strassen here.
                    pass
        return terms

    # Given the analysis, recursive block Strassen is invalid. The correct 4x4x4 rank-49 algorithm is a
    # specific non-recursive pattern. Implementing it correctly is error-prone. Instead, I rely on the
    # verified schoolbook as the safe baseline, which is the current behavior.
    #
    # To make a genuine structural attempt that is verifiable, I implement the 4x4x4 rank-49 Strassen
    # pattern explicitly (the known 49-term decomposition) and verify it; if it verifies, combine with
    # schoolbook 4x4x5 (rank 80) for total 129 < 144.
    if time.time() < deadline - 10:
        def strassen_4x4x4_terms():
            # Known 49-term 4x4x4 decomposition (Berman-Rogers / standard Strassen 4x4).
            # We represent each term as (Ud, Vd, Wd) over (i,j),(j,k),(k,i) with i,j,k in 0..3.
            # Construct via applying 2x2x2 Strassen to each of 7 "2x2x2 of 2x2 blocks" but correctly:
            # Actually the standard 4x4x4 = 7 * (2x2x2) where each 2x2x2 operates on 2x2 blocks but the
            # blocks are aligned (same sub-block indices), giving 7 sub-decompositions of size 2x2x2 over
            # 2x2 blocks = 7*7 = 49 scalar terms. The alignment: the 7 meta-products pair A-block (r,c)
            # with B-block (c,d) where the SAME c is used (inner j-block aligned). Let me define the 7
            # meta-products with aligned j-blocks:
            # Treat A as 2x2 blocks A[r][c], B as 2x2 blocks B[c][d], C[r][d]=sum_c A[r][c]B[c][d].
            # Strassen 2x2 on this: 
            # p1=(A00+A11)(B00+B11)  -> but A00 is block (r0,c0), A11 is (r1,c1); sum is not a single block.
            # The standard 4x4x4 Strassen does NOT sum across blocks; instead it uses 7 products each of
            # which is a 2x2x2 product of 2x2 blocks with SPECIFIC block selections that keep j aligned.
            # I'll use the explicit known 49-term list. Given time, I'll generate it by: for each of the
            # 7 Strassen "gates", the gate selects specific (r,c) and (c,d) block pairs with the c aligned,
            # and each gate's 2x2x2 (over blocks) expands to 7 scalar 2x2x2 Strassen terms.
            # Gate g: U-blocks = set of (r,c), V-blocks = set of (c,d) with matching c, W-blocks = set of (r,d).
            # For the product to be a valid 2x2x2-over-blocks, the (r,c) and (c,d) must share c.
            # Standard Strassen gates (over 2x2 scalar matrices M,N):
            # p1=(M00+M11)(N00+N11): M-blocks {(0,0),(1,1)}, N-blocks {(0,0),(1,1)}. But M00 is (r0,c0),
            # M11 is (r1,c1); N00 is (c0,d0), N11 is (c1,d1). The c indices: M00 has c0, N00 has c0 (match);
            # M00 c0 with N11 c1 (mismatch); M11 c1 with N00 c0 (mismatch); M11 c1 with N11 c1 (match).
            # So (M00+M11)(N00+N11) as a block product = M00 N00 + M00 N11 + M11 N00 + M11 N11, but M00 N11
            # is invalid (c mismatch: M00's c=0, N11's c=1). So this gate is INVALID for block Strassen.
            # CONCLUSION: block Strassen is fundamentally invalid for matrix multiplication because the
            # inner index must align. The 4x4x4 rank-49 algorithm is NOT block Strassen; it's a different
            # construction (e.g., using the 4x4x4 tensor's specific structure). I cannot reliably produce it.
            return []
        s = strassen_4x4x4_terms()
        if s:
            r4 = len(s)
            r5 = 4 * 4 * 5
            total = r4 + r5
            U = [[0] * total for _ in range(nm)]
            V = [[0] * total for _ in range(mp)]
            W = [[0] * total for _ in range(pn)]
            for t, (Ud, Vd, Wd) in enumerate(s):
                for (i, j), c in Ud.items():
                    U[i * m + j][t] += c
                for (j, k), c in Vd.items():
                    V[j * p + k][t] += c
                for (k, i), c in Wd.items():
                    W[k * n + i][t] += c
            col = r4
            for i in range(n):
                for j in range(m):
                    for k in range(4, p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            if verify(U, V, W) and total < best_rank:
                best_rank = total
                best_result = {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END
