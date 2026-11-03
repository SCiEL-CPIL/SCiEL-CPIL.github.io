"""Rewritten SEARCH approach: generate/validate many rank candidates for 4x4 * 4x9."""
import time, random, itertools
from fractions import Fraction as F
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 110.0  # safety margin before evaluator kill

# ---------- validation over integers (avoid FP) ----------
def _to_int(mat, scale=None):
    # mat: list[list[FracOrIntOrF]] ; returns (int2d, scale_power_of_2)
    # Determine denominator LCM as power of two (we generate dyadic entries only)
    # We'll track max denominator power.
    maxden = 1
    nrow = len(mat)
    if nrow == 0:
        return [], 1
    ncol = len(mat[0])
    for r in range(nrow):
        row = mat[r]
        if len(row) != ncol:
            raise ValueError("ragged matrix")
        for v in row:
            if isinstance(v, F):
                if v.denominator & (v.denominator - 1):
                    raise ValueError("only powers of two denominators supported")
                d = v.denominator
                if d > maxden:
                    maxden = d
    s = 1
    while maxden % s != 0:
        s *= 2
    out = [[0] * ncol for _ in range(nrow)]
    for i in range(nrow):
        for j in range(ncol):
            v = mat[i][j]
            if isinstance(v, F):
                out[i][j] = int(v * s)
            else:
                out[i][j] = int(v) * s
    return out, s

def validate(U, V, W):
    # U: (n*m, r) list[list[F]], V: (m*p, r), W: (p*n, r)
    # Check sum_c U[(i,j),c] * V[(j,k),c] * W[(k,i),c] == delta_ij*delta_ki
    n, m, p = DIMENSIONS
    if len(U) != n*m: raise ValueError("U rows")
    if len(V) != m*p: raise ValueError("V rows")
    if len(W) != p*n: raise ValueError("W rows")
    r = len(U[0])
    for i in range(n):
        Ui = U[i*m + i]
        if len(Ui) != r: raise ValueError("U ragged")
        for j in range(m):
            Uj = U[i*m + j]
            if len(Uj) != r: raise ValueError("U ragged")
            for k in range(p):
                Wk = W[k*n + i]
                # sum over c of Uj_c * V[(j,k),c] * Wk_c
                Vk = V[j*p + k]
                tot = 0
                for c in range(r):
                    u = Uj[c]; v = Vk[c]; w = Wk[c]
                    # multiply Fractions
                    if isinstance(u, F) or isinstance(v, F) or isinstance(w, F):
                        tot += u * v * w
                    else:
                        tot += u * v * w
                    if tot != 0:
                        # early partial? not safe; keep full
                        pass
                expect = (1 if (k == i and i == j) else 0)
                if isinstance(tot, F):
                    if tot != F(expect,1):
                        return False
                else:
                    if int(tot) != expect:
                        return False
    # Also need the full tensor equality for all (i,j,k). The above only checked (k,i). 
    # We must check all (i,j,k): C[i,k] = sum_j A[i,j]B[j,k]; term (i,j,k) should be 1 iff j==i and k==j? 
    # Actually multiplication tensor T[i,j,k] = 1 if all indices free; rank-1 term corresponds to selecting i0,j0,k0
    # and contributes to C[i0,k0] when A[i0,j0]B[j0,k0]. The target tensor has T[i,j,k] = 1 for all i,j,k.
    # Wait: Matrix mult tensor M[i,j,k] = 1 for all i,j,k (since C_{ik} = sum_j A_{ij} B_{jk}).
    # The rank-1 U_{i,j}, V_{j,k}, W_{k,i} term produces product 1 only when i=j and j=k? No:
    # The term is U_{i,j} V_{j,k} W_{k,i}. For the full tensor to equal delta_{i i?}. Let's re-derive.
    # Standard embedding: U is (n*m) x rank indexed by (i,j). V is (m*p) x rank indexed by (j,k). W is (p*n) x rank indexed by (k,i).
    # Then C_{i,k} = sum_j sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}  ... but W has index (k,i) so it's a scalar per c.
    # The desired: C_{i,k} = sum_j A_{i,j} B_{j,k}. So the tensor T_{i,j,k} = 1 for all i,j,k.
    # Our reconstruction gives for each (i,j,k): sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} should equal 1 for ALL i,j,k.
    # So we need to check all i,j,k (not just k==i and i==j).
    return _validate_all(U, V, W, n, m, p, r)

def _validate_all(U, V, W, n, m, p, r):
    for i in range(n):
        for j in range(m):
            Uj = U[i*m + j]
            for k in range(p):
                Vk = V[j*p + k]
                Wk = W[k*n + i]
                tot = 0
                for c in range(r):
                    u = Uj[c]; v = Vk[c]; w = Wk[c]
                    tot += u * v * w
                if isinstance(tot, F):
                    if tot != F(1,1):
                        return False
                else:
                    if int(tot) != 1:
                        return False
    return True

# ---------- fast integer validation using scaled ints ----------
def _mat_to_int3(U, V, W, n, m, p, r):
    # Determine common scale
    maxden = 1
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if isinstance(v, F):
                    d = v.denominator
                    if d & (d - 1): raise ValueError("non-dyadic")
                    if d > maxden: maxden = d
    s = 1
    while maxden % s != 0: s *= 2
    
    def sc(mat):
        out = [[0]*len(mat[0]) for _ in range(len(mat))]
        for i in range(len(mat)):
            row = mat[i]
            for j in range(len(row)):
                v = row[j]
                if isinstance(v, F):
                    out[i][j] = int(v * s)
                else:
                    out[i][j] = int(v) * s
        return out
    
    Us = sc(U); Vs = sc(V); Ws = sc(W)
    s3 = s*s*s
    
    # Convert to numpy arrays for faster validation
    Us_np = np.array(Us, dtype=np.int64)
    Vs_np = np.array(Vs, dtype=np.int64)
    Ws_np = np.array(Ws, dtype=np.int64)
    
    # Reshape: Us_np is (n*m, r), Vs_np is (m*p, r), Ws_np is (p*n, r)
    for i in range(n):
        for j in range(m):
            Ui = Us_np[i*m + j]  # shape (r,)
            for k in range(p):
                Vk = Vs_np[j*p + k]  # shape (r,)
                Wk = Ws_np[k*n + i]  # shape (r,)
                # sum over c of Ui[c]*Vk[c]*Wk[c]
                tot = np.dot(Ui, Vk * Wk)
                if tot != s3:
                    return False
    return True

# ---------- candidate generators ----------
def schoolbook_candidates():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    yield r, U, V, W

def _strassen_2x2x2():
    # returns (r, U,V,W) for n=m=p=2
    n=m=p=2
    r = 7
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[0][1] = -1; V[4][1] = 1
    W[1][1] = 1
    # M3 = (A10+A11)*B01
    U[2][2] = 1; U[3][2] = 1
    V[5][2] = 1
    W[2][2] = 1
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[5][3] = -1
    W[3][3] = 1
    # M5 = (A00+A01)(B11+B00)
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1
    # M6 = (A01-A11)*(B10+B11)
    U[1][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[5][5] = 1
    W[2][5] = 1
    # M7 = (A10-A00)*(B00+B01)
    U[2][6] = 1; U[0][6] = -1
    V[0][6] = 1; V[4][6] = 1
    W[0][6] = 1
    return r, U, V, W

def block_structured_candidates():
    # Decompose 4x4*4x9 as blocks:
    # A 4x4 -> 2x2 blocks of 2x2
    # B 4x9 -> 2x? not clean; but we can split columns into 4+4+1
    # C 4x9
    # For A (4x4) * B_left (4x8) -> C_left (4x8): use Strassen outer (7) with 2x2 schoolbook inner? That gives 7*4=28 for 4x4*4x4, and similarly for B_right 4x8? messy.
    # Instead use schoolbook for rectangular blocks but with Strassen on 2x2 sub-blocks where square.
    # We'll generate several heuristics:
    n, m, p = DIMENSIONS

    # Candidate: block-diagonal Strassen for each 2x2 output block of A*A part, plus schoolbook for rest.
    # Simpler: Use Strassen for multiplying 2x2 blocks of A by 2x2 blocks of B (when B is square). Here B is 4x9, not square.
    # So fallback: use schoolbook for all (rank 144) already covered.
    # Try a partial: split B into two 4x4 blocks and one 4x1 block.
    # For each 4x4 block of B, use Strassen-outer/schoolbook-inner? That would be 7*4=28 per 4x4, times 2 =56, plus 16 for 4x1 =72.
    # But A is shared; U rows depend only on A indices (i,j). For each block of B, we add new terms.
    # Let's implement: For left B block (cols 0-3), use 28 terms. For right B block (cols 4-7), use 28 terms (same U, different V,W). For last col, schoolbook 16 terms.
    # Need mapping from Strassen term to global (i,j,k).
    # We'll build using a helper that, given a 4x4 multiplication of A (global) by B4 (a 4x4 slice), adds terms.
    def add_strassen_4x4(A_offset_i, B_offset_k, start_term):
        # We'll generate 28 terms: 7 Strassen M, each expanded as 2x2 schoolbook (4 scalar products)
        # For a 2x2 block: Ablock rows 2a..2a+1, cols 2b..2b+1 ; Bblock rows 2b..2b+1, cols 2c..2c+1
        # Global A index: i=2a+ii, j=2b+jj
        # Global B index: j=2b+jj, k= B_offset_k + 2c+kk
        # Global C output: i=2a+ii, k= B_offset_k + 2c+kk
        terms = []
        # Strassen M1..M7 with 2x2 inner schoolbook
        for M in range(7):
            # Determine which A sub-blocks and B sub-blocks involved
            if M == 0:
                # (A00+A11)(B00+B11)
                A_blocks = [(0,0), (1,1)]
                B_blocks = [(0,0), (0,0)]
                signs = [1, 1]
            elif M == 1:
                A_blocks = [(1,1)]
                B_blocks = [(1,0), (0,0)]
                signs = [1, -1]
            elif M == 2:
                A_blocks = [(1,0), (1,1)]
                B_blocks = [(0,1)]
                signs = [1, 1]
            elif M == 3:
                A_blocks = [(0,0)]
                B_blocks = [(0,1), (1,1)]
                signs = [1, -1]
            elif M == 4:
                A_blocks = [(0,0), (0,1)]
                B_blocks = [(1,1), (0,0)]
                signs = [1, 1]
            elif M == 5:
                A_blocks = [(0,1), (1,1)]
                B_blocks = [(1,0), (1,1)]
                signs = [1, -1]
            else: # M==6
                A_blocks = [(1,0), (0,0)]
                B_blocks = [(0,0), (0,1)]
                signs = [1, -1]
            # For each A_block, each B_block, do 2x2 schoolbook (4 scalar products)
            for aidx, (ba_r, ba_c) in enumerate(A_blocks):
                bidx = aidx if len(B_blocks) == len(A_blocks) else 0
                (bb_r, bb_c) = B_blocks[bidx]
                sign = signs[aidx]
                # 2x2 schoolbook
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            # A global: row=2*ba_r+ii, col=2*ba_c+jj
                            i = 2*ba_r + ii
                            j = 2*ba_c + jj
                            # B global: row=2*bb_r+jj, col=2*bb_c+kk
                            k = B_offset_k + 2*bb_c + kk
                            # term: U[(i,j)] *= sign, V[(j,k)] *= 1, W[(k,i)] *= 1
                            terms.append((i, j, k, sign))
        return terms

    # We'll collect terms as (i,j,k,coeff)
    all_terms = []
    # Left 4x4 block of B (cols 0-3)
    all_terms += add_strassen_4x4(None, 0, None)
    # Right 4x4 block of B (cols 4-7)
    all_terms += add_strassen_4x4(None, 4, None)
    # Last column (col 8): schoolbook 16 terms
    for i in range(n):
        for j in range(m):
            for k in range(1):
                k_g = 8
                all_terms.append((i, j, k_g, 1))

    # Now assemble U,V,W with rank = len(all_terms)
    r = len(all_terms)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    for c, (i, j, k, sgn) in enumerate(all_terms):
        U[i*m+j][c] = sgn
        V[j*p+k][c] = 1
        W[k*n+i][c] = 1
    yield r, U, V, W

def random_candidates(max_r=60, trials=200):
    n, m, p = DIMENSIONS
    rng = random.Random(12345)
    for trial in range(trials):
        r = rng.randint(24, max_r)
        def rand_entry():
            x = rng.random()
            if x < 0.7:
                return 0
            elif x < 0.85:
                return rng.choice([-1, 1])
            else:
                return F(rng.choice([-1, 1]), 2)
        U = [[rand_entry() for _ in range(r)] for _ in range(n*m)]
        V = [[rand_entry() for _ in range(r)] for _ in range(m*p)]
        W = [[rand_entry() for _ in range(r)] for _ in range(p*n)]
        try:
            if _mat_to_int3(U, V, W, n, m, p, r):
                yield r, U, V, W
                if trial > 20:
                    break
        except Exception:
            pass
        if time.time() > DEADLINE - 30:
            break

# ---------- main ----------
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    best = None  # (rank, U, V, W)
    
    def consider(r, U, V, W):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    # 1) known structured candidates
    for gen in [schoolbook_candidates, block_structured_candidates]:
        for r, U, V, W in gen():
            if time.time() > DEADLINE - 10:
                break
            if _mat_to_int3(U, V, W, n, m, p, r):
                consider(r, U, V, W)
    
    # 2) random search
    if time.time() < DEADLINE - 20:
        for r, U, V, W in random_candidates(max_r=80, trials=500):
            if time.time() > DEADLINE - 5:
                break
            consider(r, U, V, W)
            if best and best[0] <= 30:
                break
    
    if best is None:
        r, U, V, W = next(schoolbook_candidates())
        return {"U": U, "V": V, "W": W}
    
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
