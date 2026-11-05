"""Matrix multiplication 4x5x5: block-partitioned Strassen baseline + exact rank reduction."""
DIMENSIONS = [4, 5, 5]
import time
import random
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 140.0
    rnd = random.Random(12345)

    def F(x):
        return Fraction(int(2 * x), 2)

    # ---- 2x2x2 Strassen factors (local 2x2 block multiplication) ----
    def strassen222():
        # rows/cols of U,V,W are (r,c) in {0,1}^2; W index = k*2+i
        u = [[Fraction(0)]*7 for _ in range(4)]
        v = [[Fraction(0)]*7 for _ in range(4)]
        w = [[Fraction(0)]*7 for _ in range(4)]
        def U(a, b, c, val):
            u[a*2+b][c] = val
        def V(a, b, c, val):
            v[a*2+b][c] = val
        def W(i, k, c, val):
            w[k*2+i][c] = val
        # t0 (a00+a11)(b00+b11) -> +c00 +c11
        U(0,0,0,1); U(1,1,0,1); V(0,0,0,1); V(1,1,0,1); W(0,0,0,1); W(1,1,0,1)
        # t1 (a10+a11)b00 -> +c10 -c11
        U(1,0,1,1); U(1,1,1,1); V(0,0,1,1); W(1,0,1,1); W(1,1,1,-1)
        # t2 a00(b01-b11) -> +c01 +c11
        U(0,0,2,1); V(0,1,2,1); V(1,1,2,-1); W(0,1,2,1); W(1,1,2,1)
        # t3 a11(b10-b00) -> +c00 +c10
        U(1,1,3,1); V(1,0,3,1); V(0,0,3,-1); W(0,0,3,1); W(1,0,3,1)
        # t4 (a00+a01)b11 -> -c00 +c01
        U(0,0,4,1); U(0,1,4,1); V(1,1,4,1); W(0,0,4,-1); W(0,1,4,1)
        # t5 (a10-a00)(b00+b01) -> +c11
        U(1,0,5,1); U(0,0,5,-1); V(0,0,5,1); V(0,1,5,1); W(1,1,5,1)
        # t6 (a01-a11)(b10+b11) -> +c00
        U(0,1,6,1); U(1,1,6,-1); V(1,0,6,1); V(1,1,6,1); W(0,0,6,1)
        return u, v, w

    # ---- Build a block-partitioned exact decomposition ----
    # A: n x m, B: m x p, C: n x p. Partition A into blocks (ar,ac), B into (br,bc),
    # C into (cr,cc). Block (ar,ac) of A times block (br,bc) of B contributes to
    # block (cr,cc) of C iff ac == br and ar == cr and bc == cc.
    # Within each such contribution we use the 2x2x2 Strassen factors (block size 2).
    def build_block(sizes_a, sizes_b):
        # sizes_a: list of row-block sizes for A (sum=n), col-block sizes (sum=m)
        # sizes_b: list of row-block sizes for B (sum=m), col-block sizes (sum=p)
        ar_sizes, ac_sizes = sizes_a
        br_sizes, bc_sizes = sizes_b
        s_u, s_v, s_w = strassen222()
        U = {}  # (i,j) -> list
        V = {}  # (j,k) -> list
        W = {}  # (k,i) -> list
        col = 0
        # row/col offsets
        a_ro = [0]
        for s in ar_sizes: a_ro.append(a_ro[-1]+s)
        a_co = [0]
        for s in ac_sizes: a_co.append(a_co[-1]+s)
        b_ro = [0]
        for s in br_sizes: b_ro.append(b_ro[-1]+s)
        b_co = [0]
        for s in bc_sizes: b_co.append(b_co[-1]+s)
        c_ro = a_ro  # C rows align with A rows
        c_co = b_co  # C cols align with B cols

        for ar in range(len(ar_sizes)):
            for ac in range(len(ac_sizes)):
                # B block with row-block == ac
                if ac >= len(br_sizes):
                    continue
                br = ac
                # C block: row ar, col bc
                for bc in range(len(bc_sizes)):
                    cc = bc
                    if ar >= len(c_ro)-1 or cc >= len(c_co)-1:
                        continue
                    # local 2x2x2 contribution for these blocks
                    for t in range(7):
                        for (la, lb) in product(range(2), range(2)):
                            for (lb2, lc) in product(range(2), range(2)):
                                pass
                    # expand Strassen factors
                    for t in range(7):
                        # U entries
                        for (la, lb) in product(range(2), range(2)):
                            val = s_u[la*2+lb][t]
                            if val:
                                i = a_ro[ar] + la
                                j = a_co[ac] + lb
                                U.setdefault((i,j), [Fraction(0)]*7)
                                U[(i,j)][t] += val
                        for (lb2, lc) in product(range(2), range(2)):
                            val = s_v[lb2*2+lc][t]
                            if val:
                                j = b_ro[br] + lb2
                                k = b_co[bc] + lc
                                V.setdefault((j,k), [Fraction(0)]*7)
                                V[(j,k)][t] += val
                        for (li, lk) in product(range(2), range(2)):
                            val = s_w[lk*2+li][t]
                            if val:
                                i = c_ro[ar] + li
                                k = c_co[cc] + lk
                                W.setdefault((k,i), [Fraction(0)]*7)
                                W[(k,i)][t] += val
                    col += 7
        # densify
        Umat = [[Fraction(0)]*col for _ in range(n*m)]
        Vmat = [[Fraction(0)]*col for _ in range(m*p)]
        Wmat = [[Fraction(0)]*col for _ in range(p*n)]
        for (i,j), row in U.items():
            for t in range(7):
                Umat[i*m+j][t] = row[t]
        for (j,k), row in V.items():
            for t in range(7):
                Vmat[j*p+k][t] = row[t]
        for (k,i), row in W.items():
            for t in range(7):
                Wmat[k*n+i][t] = row[t]
        return Umat, Vmat, Wmat, col

    # Candidate partitions (block sizes). Try several; pick best verified rank.
    candidates = [
        ([2,2],[2,2,1]),   # A rows 2+2, A cols 2+2+1 ; B rows 2+2+1, B cols 2+2+1
        ([2,2],[2,2,1]),
        ([2,2],[1,2,2]),
        ([2,2],[2,1,2]),
        ([1,1,1,1],[1,1,1,1,1]),
    ]

    best = None
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        Umat, Vmat, Wmat, col = build_block(cand[0], cand[1])
        # verify exact
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for t in range(col):
                        s += Umat[i*m+j][t] * Vmat[j*p+k][t] * Wmat[k*n+i][t]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            if best is None or col < best[0]:
                best = (col, Umat, Vmat, Wmat)

    # Fallback: schoolbook if nothing verified
    if best is None:
        col = n*m*p
        Umat = [[Fraction(0)]*col for _ in range(n*m)]
        Vmat = [[Fraction(0)]*col for _ in range(m*p)]
        Wmat = [[Fraction(0)]*col for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Umat[i*m+j][c] = 1
                    Vmat[j*p+k][c] = 1
                    Wmat[k*n+i][c] = 1
                    c += 1
        best = (col, Umat, Vmat, Wmat)

    rank, Umat, Vmat, Wmat = best

    # ---- Exact rank reduction: try to drop columns while keeping the tensor = 1 ----
    # Represent each column r as its "signature": for each (i,j,k) the product
    # U[i,j,r]*V[j,k,r]*W[i,k,r]. The tensor is the sum over r of these signatures.
    # Dropping column r is valid iff its signature is zero on every (i,j,k) it
    # "needs" -- i.e. signature is identically zero. More generally we can try to
    # replace a column by a linear combination that cancels. We do a simple greedy:
    # repeatedly find a column whose removal leaves the tensor still equal to 1
    # (i.e. its signature is the zero tensor) and drop it.
    def signatures(Um, Vm, Wm, R):
        sigs = []
        for r in range(R):
            sig = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = Um[i*m+j][r] * Vm[j*p+k][r] * Wm[k*n+i][r]
                        if val:
                            sig[(i,j,k)] = val
            sigs.append(sig)
        return sigs

    # Try to find columns that are "redundant": a column r can be removed if there
    # exist coefficients (a linear combo of other columns) reproducing its signature.
    # Simpler robust move: if two columns have identical signature, drop one.
    # Also: if a column's signature is the negative of another's, drop both.
    def try_reduce(Um, Vm, Wm, R):
        sigs = signatures(Um, Vm, Wm, R)
        removed = [False]*R
        # dedup identical signatures
        seen = {}
        for r in range(R):
            key = tuple(sorted(sigs[r].items()))
            if not sigs[r]:
                removed[r] = True
                continue
            if key in seen:
                removed[r] = True
            else:
                seen[key] = r
        # cancel opposite pairs
        seen2 = {}
        for r in range(R):
            if removed[r]:
                continue
            key = tuple(sorted(sigs[r].items()))
            negkey = tuple(sorted(((kk, -vv) for kk, vv in sigs[r].items())))
            if negkey in seen2:
                removed[r] = True
                removed[seen2[negkey]] = True
            else:
                seen2[key] = r
        if not any(removed):
            return Um, Vm, Wm, R
        # rebuild without removed columns
        keep = [r for r in range(R) if not removed[r]]
        newR = len(keep)
        nU = [[Fraction(0)]*newR for _ in range(n*m)]
        nV = [[Fraction(0)]*newR for _ in range(m*p)]
        nW = [[Fraction(0)]*newR for _ in range(p*n)]
        for ni, r in enumerate(keep):
            for i in range(n*m):
                nU[i][ni] = Um[i][r]
            for j in range(m*p):
                nV[j][ni] = Vm[j][r]
            for k in range(p*n):
                nW[k][ni] = Wm[k][ni]
        return nU, nV, nW, newR

    # iterate reduction a few times
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        Um, Vm, Wm, R = try_reduce(Umat, Vmat, Wmat, rank)
        if R == rank:
            break
        rank = R
        Umat, Vmat, Wmat = Um, Vm, Wm

    # Final exact verification
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Umat[i*m+j][r] * Vmat[j*p+k][r] * Wmat[k*n+i][r]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if not ok:
        # revert to schoolbook to guarantee validity
        rank = n*m*p
        Umat = [[0]*rank for _ in range(n*m)]
        Vmat = [[0]*rank for _ in range(m*p)]
        Wmat = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Umat[i*m+j][c] = 1
                    Vmat[j*p+k][c] = 1
                    Wmat[k*n+i][c] = 1
                    c += 1

    # Convert to plain int/float lists
    Uout = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in Umat]
    Vout = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in Vmat]
    Wout = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in Wmat]
    return {"U": Uout, "V": Vout, "W": Wout}
# EVOLVE-BLOCK-END
