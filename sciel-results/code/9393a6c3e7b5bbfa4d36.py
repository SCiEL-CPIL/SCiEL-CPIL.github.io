"""Matrix multiplication 4x4 * 4x8 rank-minimal decomposition (Strassen-based)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _add_term(U, V, W, u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
    U[u_idx][col] += u_val
    V[v_idx][col] += v_val
    W[w_idx][col] += w_val

def construct():
    import numpy as np
    n, m, p = DIMENSIONS  # 4,4,8

    # We'll build a candidate rank-R decomposition via two 4x4 Strassen products.
    # Strassen 4x4 = 7 products of 2x2 blocks; each 2x2 matmul = 4 trivial scalar terms.
    # Total 28 terms per block, 56 for two blocks.

    # Represent each rank-1 term as (u_dict, v_dict, w_dict) where dicts map row->coef.
    # u index: i*m + j ; v index: j*p + k ; w index: k*n + i
    terms = []  # list of (u_idx,u_val,v_idx,v_val,w_idx,w_val)

    # 2x2 sub-block index helpers for 4x4 blocks.
    # A block (r,c) covers rows r*2..r*2+1, cols c*2..c*2+1.
    def a_idx(r, c):  # returns (i,j)
        return (r * 2 + (0, 1), c * 2 + (0, 1))  # not used directly

    # We'll directly expand: for a 2x2 block product X (from A block ar,ac) times Y (from B block br,bc),
    # contribution to C block (ar,bc)?? Actually Strassen recombines, so I'll compute the 7 M's as 2x2 matrices
    # then expand the final C = f(M1..M7) into scalar terms, then expand each M into its 2x2 trivial terms.
    # Simpler: do it at the scalar level using the explicit Strassen formulas on 2x2 blocks.

    # I'll build 2x2 "linear combination" blocks for A and B (each is 2x2 of linear forms in A),
    # then M = X*Y (2x2 matmul), then C_block = linear combo of M's.
    # To keep exactness, I'll expand everything to scalar (i,j,k) contributions.

    # Define A 2x2 submatrices symbolically: Ablk[ar][ac][a][c] = A[ar*2+a, ac*2+c]
    # Strassen for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7 ; C12 = M3+M5 ; C21 = M2+M4 ; C22 = M1-M2+M3+M6

    # Each M is a 2x2 matrix product of two 2x2 (linear) matrices.
    # Represent a 2x2 linear matrix as 4 entries, each a linear form: dict (i,j)->coef for A side,
    # and (j,k)->coef for B side.

    def zero_A2(): return [[0]*2 for _ in range(2)]
    def zero_B2(cb): return [[0]*2 for _ in range(2)]

    def Ablk(r, c):
        # returns 2x2 of A linear forms: entry [a][b] = {(r*2+a, c*2+b): 1}
        return [[{( (r*2+a), (c*2+b) ): 1} for b in range(2)] for a in range(2)]
    def Bblk(r, c, cb):
        # B block with column offset cb (0 or 4): entry [a][b] = { (r*2+a, cb + c*2+b): 1 }
        return [[{ ( (r*2+a), (cb + c*2+b) ): 1 } for b in range(2)] for a in range(2)]

    def add2(X, Y):
        return [[X[a][b] + Y[a][b] for b in range(2)] for a in range(2)]
    def sub2(X, Y):
        return [[X[a][b] - Y[a][b] for b in range(2)] for a in range(2)]
    def merge(d):
        # merge a dict of dict (or list of dicts) into one dict
        out = {}
        for dd in d:
            for k, v in dd.items():
                out[k] = out.get(k, 0) + v
        return out

    def A2sum(*blks):
        r = None
        for b in blks:
            r = b if r is None else add2(r, b)
        return r
    def A2diff(a, b):
        return sub2(a, b)

    A11 = Ablk(0,0); A12 = Ablk(0,1); A21 = Ablk(1,0); A22 = Ablk(1,1)

    def product_terms(X, Y, cb):
        # X: 2x2 of A-linear dicts, Y: 2x2 of B-linear dicts (with column offset cb baked in).
        # Returns list of (u_dict, v_dict, w_block_dict) where w_block is 2x2 dict of (a,b)->coef for the C block.
        # Actually for 2x2 matmul M = X*Y: M[a][b] = sum_c X[a][c]*Y[c][b]. Each product of linear forms -> rank terms.
        terms_out = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    # X[a][c] dict (i,j)->coef, Y[c][b] dict (j,k)->coef
                    # product is bilinear; expand: each (i,j) in X and (j2,k) in Y with j==j2
                    xd = X[a][c]; yd = Y[c][b]
                    # combine: for each (i,j) in xd and (j2,k) in yd where j==j2
                    for (i, j), xv in xd.items():
                        for (j2, k), yv in yd.items():
                            if j == j2:
                                u = {(i, j): xv}
                                v = {(j2, k): yv}
                                w = {((a), (b)): 1}  # position within this M block
                                terms_out.append((u, v, w))
        return terms_out

    # Build M1..M7
    cb = 0  # handled per call; for left block B uses cb=0, right cb=4
    def build_block(cb):
        B11 = Bblk(0,0,cb); B12 = Bblk(0,1,cb); B21 = Bblk(1,0,cb); B22 = Bblk(1,1,cb)
        M = {
            1: (A2sum(A11,A22), add2(B11,B22)),
            2: (A2sum(A21,A22), B11),
            3: (A11, sub2(B12,B22)),
            4: (A22, sub2(B21,B11)),
            5: (A2sum(A11,A12), B22),
            6: (A2diff(A21,A11), add2(B11,B12)),
            7: (A2diff(A12,A22), add2(B21,B22)),
        }
        # C blocks: C11 = M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        Cdef = {
            (0,0): [(1,1),(4,1),(-5,1),(7,1)],
            (0,1): [(3,1),(5,1)],
            (1,0): [(2,1),(4,1)],
            (1,1): [(1,1),(-2,1),(3,1),(6,1)],
        }
        # collect terms: (u_dict, v_dict, w_dict) where w_dict maps (cr,cblock)->coef ; we need global w index (k*n+i)
        terms_out = []
        for (cr, cc), entries in Cdef.items():
            for (mid, sgn) in entries:
                for (u, v, w) in product_terms(M[mid][0], M[mid][1], cb):
                    for (wa, wb), wv in w.items():
                        # global C position: row = cr*2 + wa, col = cb + cc*2 + wb
                        i = cr * 2 + wa
                        k = cb + cc * 2 + wb
                        widx = k * n + i
                        terms_out.append((u, v, {widx: sgn * wv}))
        return terms_out

    all_terms = build_block(0) + build_block(4)

    # Deduplicate: sum terms that have identical (u,v) by merging w.
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(int))
    for (u, v, w) in all_terms:
        key = (tuple(sorted(u.items())), tuple(sorted(v.items())))
        for widx, wv in w.items():
            grouped[key][widx] += wv
    cols = []
    for key, wdict in grouped.items():
        if any(x != 0 for x in wdict.values()):
            u = dict(key[0]); v = dict(key[1])
            cols.append((u, v, wdict))
    R = len(cols)

    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    for col, (u, v, w) in enumerate(cols):
        for (i, j), val in u.items():
            U[i*m + j][col] = val
        for (j, k), val in v.items():
            V[j*p + k][col] = val
        for (widx), val in w.items():
            W[widx][col] = val

    # Verify with NumPy
    def to_mat(M, rows, cols_):
        return np.array(M, dtype=float)
    Ua = to_mat(U, n*m, R); Va = to_mat(V, m*p, R); Wa = to_mat(W, p*n, R)
    # reconstruct C[i,k] = sum_r U[i*4+j? ] ... C[i,k] = sum_{j,r} U[i*m+j, r] V[j*p+k, r] * (W[k*n+i,r])? 
    # Actually tensor: C[i,k] = sum_r (sum_j U[i*m+j,r] * ? ) ... The decomposition: C = sum_r u_r v_r w_r where
    # (u_r ⊗ v_r) applied gives A·B contribution and w_r is output placement. Standard: 
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])? No, W multiplies.
    # Convention: C[i,k] = sum_r U[i*m+j,r] V[j*p+k, r] W[k*n+i, r] summed over j and r? 
    # The standard matmul tensor decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] V[j*p+k,r]) * W[k*n+i,r].
    # Let's verify by direct expansion.
    C_rec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(R):
                    s += Ua[i*m + j, r] * Va[j*p + k, r]
            C_rec[i, k] = s * Wa[k*n + i, r] if False else s  # placeholder
    # Correct reconstruction: C[i,k] = sum_r W[k*n+i,r] * sum_j (U[i*m+j,r] * V[j*p+k,r])? 
    # That's not bilinear in the right way. Let me use the canonical: 
    # C[i,k] = sum_r (u_r[i] * v_r[k] * w_r[i,k]) where u_r is linear in A indexed (i,j)? 
    # The matmul tensor: T[i,k, i,j, j,k] = delta. Decomposition: T = sum_r U_r[i,j] V_r[j,k] W_r[i,k] with C[i,k]=sum_r U_r[i,j]V_r[j,k]W_r[i,k] summed over j.
    C_rec = np.zeros((n, p))
    for r in range(R):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += Ua[i*m + j, r] * Va[j*p + k, r]
                C_rec[i, k] += s * Wa[k*n + i, r]
    # Test on random
    ok = True
    for trial in range(5):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        if not np.allclose(C_true, C_rec, atol=1e-9):
            ok = False
            break
    if not ok:
        # fall back to seed schoolbook
        rank = n*m*p
        Us = [[0]*rank for _ in range(n*m)]
        Vs = [[0]*rank for _ in range(m*p)]
        Ws = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j][column] = 1
                    Vs[j*p+k][column] = 1
                    Ws[k*n+i][column] = 1
                    column += 1
        return {"U": Us, "V": Vs, "W": Ws}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
