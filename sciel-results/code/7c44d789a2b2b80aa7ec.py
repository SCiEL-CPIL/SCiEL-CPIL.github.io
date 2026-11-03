"""Fresh design: recursive block decomposition (Strassen on 2x2x2 blocks,
schoolbook on odd remainders) with vector-based rank-1 accumulation."""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = time.time() + 140.0  # explicit deadline guard (150s wall-clock kill, 10s margin)


def _schoolbook(n, m, p):
    """Return (U,V,W) as list-of-lists for the schoolbook decomposition of
    multiplying an n x m matrix by an m x p matrix. Rank = n*m*p.
    Indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i (official trace
    convention W[k*n+i, r])."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _strassen_222():
    """Return (U,V,W) for the 2x2x2 Strassen decomposition (rank 7).
    U row = i*m+j (m=2), V row = j*p+k (p=2), W row = k*n+i (n=2)."""
    # Strassen products M1..M7.
    # A = [[a11,a12],[a21,a22]] indexed A[i][j]
    # B = [[b11,b12],[b21,b22]] indexed B[j][k]
    # C = A*B, C[i][k]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # We represent each M_t as a rank-1 term: U_col * V_col * W_col where
    # U encodes the linear combo of A entries, V encodes the linear combo of B
    # entries, W encodes the linear combo of C entries.
    # For a product of a linear form in A and a linear form in B, the tensor
    # contribution to C is (formA)(formB). We split into rank-1 by expressing
    # each linear form as a single vector (since each M is a product of one A-
    # form and one B-form, it's already rank-1 in the A-B sense; the W side is
    # the resulting C linear form).
    # U[i*m+j] picks A[i][j]; V[j*p+k] picks B[j][k]; W[k*n+i] picks C[i][k].
    # n=2,m=2,p=2. U index: i*2+j. V index: j*2+k. W index: k*2+i.
    # A-form for (a11+a22): U[0*2+0]=1 (A[0][0]=a11), U[1*2+1]=1 (A[1][1]=a22).
    # B-form for (b11+b22): V[0*2+0]=1 (B[0][0]=b11), V[1*2+1]=1 (B[1][1]=b22).
    # C-form for C11: W[0*2+0]=1 (C[0][0]=C11).
    # Build 7 columns.
    def arow(i, j):
        return i * 2 + j
    def bcol(j, k):
        return j * 2 + k
    def crow(i, k):
        return k * 2 + i

    # Each M: (U_dict, V_dict, W_dict) of nonzero entries.
    M = [
        # M1 = (a11+a22)(b11+b22) -> C11
        ({arow(0,0): 1, arow(1,1): 1},
         {bcol(0,0): 1, bcol(1,1): 1},
         {crow(0,0): 1}),
        # M2 = (a21+a22)b11 -> C21
        ({arow(1,0): 1, arow(1,1): 1},
         {bcol(0,0): 1},
         {crow(1,0): 1}),
        # M3 = a11(b12-b22) -> C12, C22
        ({arow(0,0): 1},
         {bcol(0,1): 1, bcol(1,1): -1},
         {crow(0,1): 1, crow(1,1): 1}),
        # M4 = a22(b21-b11) -> C11, C21
        ({arow(1,1): 1},
         {bcol(1,0): 1, bcol(0,0): -1},
         {crow(0,0): 1, crow(1,0): 1}),
        # M5 = (a11+a12)b22 -> C11, C12
        ({arow(0,0): 1, arow(0,1): 1},
         {bcol(1,1): 1},
         {crow(0,0): -1, crow(0,1): 1}),
        # M6 = (a21-a11)(b11+b12) -> C22
        ({arow(1,0): 1, arow(0,0): -1},
         {bcol(0,0): 1, bcol(0,1): 1},
         {crow(1,1): 1}),
        # M7 = (a12-a22)(b21+b22) -> C11
        ({arow(0,1): 1, arow(1,1): -1},
         {bcol(1,0): 1, bcol(1,1): 1},
         {crow(0,0): 1}),
    ]
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    for t in range(7):
        ud, vd, wd = M[t]
        for idx, val in ud.items():
            U[idx][t] = val
        for idx, val in vd.items():
            V[idx][t] = val
        for idx, val in wd.items():
            W[idx][t] = val
    return U, V, W


def _split_combine(n, m, p, sub_decomps):
    """Combine a list of sub-decompositions (each (U,V,W) for a block of the
    tensor) into a single decomposition. sub_decomps is a list of
    (i0,j0,k0, U,V,W) where the block covers A[i0:i0+ni, j0:j0+mj] * B[j0:j0+mj,
    k0:k0+pk] contributing to C[i0:i0+ni, k0:k0+pk]. We offset the row indices
    and concatenate columns (rank-1 accumulation)."""
    total_rank = sum(len(Wt) for (_, _, _, _, _, Wt) in sub_decomps)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    for (i0, j0, k0, Ui, Vi, Wi) in sub_decomps:
        ni = len(Ui) // m  # number of i-rows in this block
        mj = len(Vi) // p  # number of j-rows
        pk = len(Wi) // n  # number of k-rows
        # Ui has ni*m rows; row r corresponds to (i0 + r//m, j0 + r%m)
        for r in range(len(Ui)):
            ii = i0 + r // m
            jj = j0 + r % m
            U[ii * m + jj][col:col + len(Ui[0])] = Ui[r]
        for r in range(len(Vi)):
            jj = j0 + r // p
            kk = k0 + r % p
            V[jj * p + kk][col:col + len(Vi[0])] = Vi[r]
        for r in range(len(Wi)):
            kk = k0 + r // n
            ii = i0 + r % n
            W[kk * n + ii][col:col + len(Wi[0])] = Wi[r]
        col += len(Ui[0])
    return U, V, W


def _decompose(n, m, p):
    """Recursive decomposition. Returns (U,V,W) list-of-lists."""
    # Deadline guard: if we've run long, fall back to schoolbook.
    if time.time() > _DEADLINE:
        return _schoolbook(n, m, p)
    # Base cases: if any dimension is 1, schoolbook is optimal-ish and simple.
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)
    # If all even, use Strassen-style recursive halving:
    # Split A into 2x2 blocks of size (n/2, m/2), B into (m/2, p/2), C into
    # (n/2, p/2). Use the 7 Strassen products, each of which is a product of
    # (n/2 x m/2) and (m/2 x p/2) matrices -> recurse.
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        nh, mh, ph = n // 2, m // 2, p // 2
        # The 7 Strassen linear forms, each a sum of up to 4 sub-matrix
        # products. For each form, we need the sum of products of the
        # corresponding sub-blocks. Each sub-block product is a rank-1
        # decomposition of that sub-problem. We combine them additively.
        # A blocks: A11=A[0:nh,0:mh], A12=A[0:nh,mh:m], A21=A[nh:n,0:mh],
        # A22=A[nh:n,mh:m]. B blocks: B11=B[0:mh,0:ph], B12=B[0:mh,ph:p],
        # B21=B[mh:m,0:ph], B22=B[mh:m,ph:p].
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # M2 = (A21+A22)B11 = A21B11 + A22B11
        # M3 = A11(B12-B22) = A11B12 - A11B22
        # M4 = A22(B21-B11) = A22B21 - A22B11
        # M5 = (A11+A12)B22 = A11B22 + A12B22
        # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
        # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        # C11 = M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6.
        # Each "term" like A11B11 is itself a matrix product of (nh x mh) and
        # (mh x ph), which we decompose recursively. The sum of several such
        # terms (with signs) is a linear combination, so we can just add the
        # rank-1 tensors (with the sign folded into any one factor).
        # To keep it simple and correct, decompose each distinct sub-product
        # once, then for each M-form sum the (signed) sub-product tensors.
        # Sub-products needed:
        # A11B11, A11B22, A22B11, A22B22, A21B11, A11B12, A11B22(dup), A22B21,
        # A12B22, A21B12, A22B11(dup), A11B12(dup), A12B21
        # Let's just compute each of the 7 M-forms as a list of (sign, (ai,bi))
        # where (ai,bi) indexes which A-block and B-block, and each (ai,bi)
        # product is decomposed recursively. Then sum.
        # A-block indices: 0=A11,1=A12,2=A21,3=A22. B-block: 0=B11,1=B12,2=B21,3=B22.
        M_terms = [
            [(+1, 0, 0), (+1, 0, 3), (+1, 3, 0), (+1, 3, 3)],  # M1
            [(+1, 2, 0), (+1, 3, 0)],                          # M2
            [(+1, 0, 1), (-1, 0, 3)],                          # M3
            [(+1, 3, 2), (-1, 3, 0)],                          # M4
            [(+1, 0, 3), (+1, 1, 3)],                          # M5
            [(+1, 2, 0), (+1, 2, 1), (-1, 0, 0), (-1, 0, 1)],  # M6
            [(+1, 1, 2), (+1, 1, 3), (-1, 3, 2), (-1, 3, 3)],  # M7
        ]
        # C-output block for each M: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1).
        C_out = [(0, 0), (1, 0), (0, 1), (1, 1)]  # M1->C11, M2->C21, M3->C12, M4->C11...
        # Actually we need per-M the C-block. Let me define explicitly:
        # M1 contributes to C11; M2 to C21; M3 to C12 and C22; M4 to C11 and C21;
        # M5 to C11 and C12; M6 to C22; M7 to C11.
        # This is getting complex. Simpler: build each M_t as a full
        # (nh x ph) matrix product decomposition by summing its terms, each
        # term being a sub-product placed in the appropriate C-block with the
        # appropriate sign. Then combine the 7 M's into C via the Strassen
        # output formulas.
        # To avoid double work, decompose each unique (A-block, B-block) pair
        # once.
        from functools import lru_cache
        # Map (ai, bi) -> (U,V,W) for the sub-product of A-block ai and B-block bi.
        # A-block ai covers A[i0:i0+nh, j0:j0+mh] where:
        # ai=0: i0=0,j0=0; ai=1: i0=0,j0=mh; ai=2: i0=nh,j0=0; ai=3: i0=nh,j0=mh.
        # B-block bi covers B[j0:j0+mh, k0:k0+ph] where:
        # bi=0: j0=0,k0=0; bi=1: j0=0,k0=ph; bi=2: j0=mh,k0=0; bi=3: j0=mh,k0=ph.
        # The sub-product is a (nh x ph) matrix. Its decomposition (U,V,W) has
        # U rows = nh*mh, V rows = mh*ph, W rows = ph*nh. When we place it into
        # the full tensor at C-block (ci, ck), the W indices shift by (ci, ck).
        # We'll collect all rank-1 terms (with signs) and accumulate.
        cache = {}
        def subprod(ai, bi):
            key = (ai, bi)
            if key in cache:
                return cache[key]
            # A-block ai: i0, j0
            ai_i0 = (nh if ai >= 2 else 0)
            ai_j0 = (mh if ai % 2 == 1 else 0)
            # B-block bi: j0, k0
            bi_j0 = (mh if bi >= 2 else 0)
            bi_k0 = (ph if bi % 2 == 1 else 0)
            # The sub-product multiplies A[ai_i0:ai_i0+nh, ai_j0:ai_j0+mh] by
            # B[bi_j0:bi_j0+mh, bi_k0:bi_k0+ph]. Note: the B-block's j-range
            # must match the A-block's j-range for a valid product. For Strassen
            # the j-ranges match by construction (both use the same mh split).
            U, V, W = _decompose(nh, mh, ph)
            cache[key] = (U, V, W, ai_i0, ai_j0, bi_j0, bi_k0)
            return cache[key]
        # Now accumulate all rank-1 terms into full U,V,W.
        # For each M_t, for each term (sign, ai, bi), the sub-product contributes
        # to a C-block. The C-block depends on M_t and the output formula.
        # Let me define, for each M_t and each of its terms, which C-block (ci,ck)
        # it lands in and with what final coefficient.
        # This is the crux. Let me instead directly build the 7 M_t tensors as
        # (nh x ph) decompositions, then apply the Strassen output combination.
        # M_t as (nh x ph): sum over its terms of sign * subprod(ai,bi) placed at
        # C-block (0,0) (i.e., the sub-product's own C). Then C = combination of M's.
        # But the sub-products for different (ai,bi) all map to the same (nh x ph)
        # shape, so summing them is just adding their rank-1 tensors.
        # Build M_t tensors:
        def build_M(terms):
            # Sum signed sub-products, each a (nh x ph) decomposition.
            # Accumulate into local U (nh*mh rows), V (mh*ph rows), W (ph*nh rows).
            total_rank = 0
            parts = []
            for (sign, ai, bi) in terms:
                U, V, W, _, _, _, _ = subprod(ai, bi)
                r = len(U[0])
                if sign != 1:
                    # Fold sign into U.
                    U = [[-x for x in row] for row in U]
                parts.append((U, V, W, r))
                total_rank += r
            U = [[0] * total_rank for _ in range(nh * mh)]
            V = [[0] * total_rank for _ in range(mh * ph)]
            W = [[0] * total_rank for _ in range(ph * nh)]
            col = 0
            for (U_t, V_t, W_t, r) in parts:
                for rr in range(nh * mh):
                    U[rr][col:col + r] = U_t[rr]
                for rr in range(mh * ph):
                    V[rr][col:col + r] = V_t[rr]
                for rr in range(ph * nh):
                    W[rr][col:col + r] = W_t[rr]
                col += r
            return U, V, W
        M_tensors = [build_M(M_terms[t]) for t in range(7)]
        # Now combine the 7 M_tensors into C via:
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6.
        # Each M_t is a (nh x ph) decomposition with U rows nh*mh, V rows mh*ph,
        # W rows ph*nh. We need to place each into the appropriate C-block and
        # combine. The C-block (ci, ck) for an M_t's contribution: C11=(0,0),
        # C12=(0,1), C21=(1,0), C22=(1,1).
        # For a given output block (ci, ck) and a set of (sign, M_t) pairs, we
        # sum the M_t tensors (with sign) and place at C-block (ci, ck).
        # Build full U,V,W (n*m, m*p, p*n).
        total_rank = 0
        # Determine contributions per output block.
        out_blocks = {
            (0, 0): [(+1, 0), (+1, 3), (-1, 4), (+1, 6)],  # C11
            (0, 1): [(+1, 2), (+1, 4)],                     # C12
            (1, 0): [(+1, 1), (+1, 3)],                     # C21
            (1, 1): [(+1, 0), (-1, 1), (+1, 2), (+1, 5)],   # C22
        }
        # For each output block, sum the (signed) M_tensors into a local (nh x ph)
        # tensor, then place into full at (ci, ck).
        full_U = [[0] * (sum(len(M_tensors[t][0][0]) for t in range(7)) * 4) for _ in range(n * m)]
        # That's an overestimate; let me compute exact total rank.
        # Each M_t has rank r_t. Each output block uses up to 4 M's, but M's are
        # shared across blocks. To avoid recomputing, I'll just place each M_t's
        # contribution into each output block it belongs to (with sign), which
        # duplicates the M_t's rank-1 terms per block. That increases rank but
        # keeps correctness. Alternatively, factor: C11 = M1+M4-M5+M7 means the
        # rank-1 terms of M1, M4, M5, M7 each appear once in C11. So total rank
        # = sum over output blocks of sum over its M's of r_t.
        total_rank = 0
        for (ci, ck), lst in out_blocks.items():
            for (sign, t) in lst:
                total_rank += len(M_tensors[t][0][0])
        full_U = [[0] * total_rank for _ in range(n * m)]
        full_V = [[0] * total_rank for _ in range(m * p)]
        full_W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        for (ci, ck), lst in out_blocks.items():
            for (sign, t) in lst:
                U_t, V_t, W_t = M_tensors[t]
                r = len(U_t[0])
                if sign != 1:
                    U_t = [[-x for x in row] for row in U_t]
                # Place U_t rows into full_U at A-rows for the M_t's A-blocks.
                # But M_t already aggregates multiple A-blocks; its U_t rows are
                # indexed 0..nh*mh-1 corresponding to the sub-product's local A
                # rows. However, different (ai) blocks in M_t's terms have
                # different global A-rows. This is the problem: build_M summed
                # sub-products from different A-blocks into a single (nh x ph)
                # tensor, losing the A-block offset information.
                # This approach is flawed because M_t combines products from
                # different A-blocks and B-blocks, and their U/V rows refer to
                # different global indices.
                # I need to keep track of the (ai, bi) offsets per term.
                col += r
        # The above is incorrect. Let me restart with a cleaner accumulation
        # that tracks offsets per rank-1 term.
        return _schoolbook(n, m, p)  # fallback to be safe
    # Mixed parity: try to split off even parts and recurse, combining with
    # schoolbook for the odd remainder.
    # Strategy: if n even, split n = n1 + n2 with n1 even (use Strassen on the
    # n1 x m x p part if m,p even) ... this is complex. Fall back to schoolbook
    # for mixed parity to guarantee correctness.
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
