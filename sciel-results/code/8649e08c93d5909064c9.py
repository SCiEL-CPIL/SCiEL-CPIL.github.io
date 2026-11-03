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
    n, m, p = DIMENSIONS

    # For a 4x4 * 4x4 multiplication, use Strassen on 2x2 blocks.
    # Strassen for 2x2*2x2 has 7 multiplications M1..M7.
    # Each M involves products of 2x2 blocks, which we expand as 2x2 schoolbook (4 scalar products each).
    # So 7 * 4 = 28 terms per 4x4*4x4.
    #
    # For 4x4 * 4x9: split B into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A*B1 + A*B2 + A*B3
    # Use 28 terms for A*B1, 28 terms for A*B2, 16 terms for A*B3 (schoolbook).
    # Total: 72 terms.
    #
    # Strassen formulas for 2x2 blocks:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = A11*(B10-B00)
    # M3 = (A10+A11)*B01
    # M4 = A00*(B01-B11)
    # M5 = (A00+A01)*(B11+B00)
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    #
    # For each M, the 2x2 result is computed, and we need to map back to scalar products.
    # Each 2x2 matrix product (X*Y) where X,Y are 2x2 requires 4 scalar products in schoolbook.
    # But in Strassen, M is a scalar "multiplication" of two 2x2 matrices.
    # We expand each such 2x2*2x2 into 4 scalar terms.

    def strassen_4x4_terms(B_col_offset):
        """Generate (i,j,k,coeff) terms for A(4x4) * B_slice(4x4) where B_slice has columns B_col_offset..B_col_offset+3."""
        terms = []
        # M1 = (A00+A11)*(B00+B11)
        # A00: i in {0,1}, j in {0,1}; A11: i in {2,3}, j in {2,3}
        # B00: j in {0,1}, k in {0,1}; B11: j in {2,3}, k in {2,3}
        # (A00+A11) has entries at (i,j) for both blocks with coeff +1
        # (B00+B11) has entries at (j,k) for both blocks with coeff +1
        # Product: sum over (i,j) in A00+A11, (j,k) in B00+B11 of A[i,j]*B[j,k]
        # But we're decomposing into rank-1 terms: each scalar product A[i,j]*B[j,k] is one term.
        # For M1, the "A part" is (A00+A11) and "B part" is (B00+B11).
        # The 2x2*2x2 expansion: for each of the 4 positions in the 2x2 result, we sum.
        # Actually, let's think differently.
        #
        # Standard approach: M1 = (A00+A11)*(B00+B11) is a 2x2 matrix.
        # (A00+A11) is a 4x4 matrix with A00 in top-left, A11 in bottom-right.
        # (B00+B11) is a 4x4 matrix with B00 in top-left, B11 in bottom-right.
        # Their product is a 4x4 matrix. But we only care about how it contributes to C.
        #
        # Actually, the correct way: each Strassen "multiplication" M is of two 2x2 matrices.
        # We expand M = X*Y (2x2*2x2) as 4 scalar products using schoolbook on the 2x2 blocks.
        # X has 4 entries, Y has 4 entries. X*Y[c,d] = sum_{j} X[c,j]*Y[j,d].
        # So 4 scalar products: (c=0,j=0,d=0), (c=0,j=1,d=0), etc. No, 2*2*2=8? No.
        # 2x2 * 2x2 = sum over j=0,1 of X[c,j]*Y[j,d] for each (c,d). That's 4 outputs, each sum of 2 terms.
        # But in tensor decomposition, each scalar product X[c,j]*Y[j,d] is a separate rank-1 term.
        # So 2*2*2 = 8 scalar products per 2x2*2x2? No: c in {0,1}, j in {0,1}, d in {0,1} -> 8 terms.
        # But that's just schoolbook for 2x2*2x2 which is rank 4 (since C[c,d] = sum_j X[c,j]Y[j,d], the tensor is sum_j e_c^T ⊗ e_j^T ⊗ e_d^T).
        # Wait, 2x2*2x2 has rank 4 (schoolbook). So each M (which is a 2x2*2x2 product) contributes 4 rank-1 terms.
        # 7 M's * 4 = 28. Correct.

        # Let me define the Strassen terms more carefully.
        # For M = X*Y where X is 2x2 (rows a=0,1; cols j=0,1) and Y is 2x2 (rows j=0,1; cols b=0,1):
        # The 4 rank-1 terms correspond to (a,j,b) for a=0,1; j=0,1; b=0,1.
        # U[(a,j)] gets X[a,j], V[(j,b)] gets Y[j,b], W[(b,a)] gets 1.

        # Now map to global indices for 4x4*4x4:
        # A block (br, bc): rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
        # B block (br, bc): rows 2*br to 2*br+1, cols B_col_offset + 2*bc to B_col_offset + 2*bc+1

        # M1 = (A00+A11)*(B00+B11)
        # X = A00+A11: X[a,j] = A[2*br_a + a, 2*bc_a + j] where (br_a,bc_a) in {(0,0),(1,1)}
        # Y = B00+B11: Y[j,b] = B[2*br_b + j, B_col_offset + 2*bc_b + b] where (br_b,bc_b) in {(0,0),(1,1)}
        # But X and Y are 2x2 matrices. The "A part" of M1 is not a single 2x2 block but a sum.
        # This means we can't directly use the 2x2*2x2 schoolbook expansion.
        #
        # Alternative: just enumerate all scalar products explicitly.
        # For M1: the contribution to C[i,k] is sum over j of A[i,j]*B[j,k] where A[i,j] is from A00 or A11,
        # and B[j,k] is from B00 or B11, with appropriate signs.
        #
        # Let me just hardcode the 28 terms by enumerating.

        # I'll use a different approach: for each of the 7 Strassen multiplications,
        # I'll list which (i,j) pairs in A and (j,k) pairs in B are involved, with signs.
        # Then each scalar product A[i,j]*B[j,k] is one rank-1 term.

        # M1: (A00+A11)*(B00+B11)
        # A entries: (0,0), (0,1), (1,0), (1,1) from A00; (2,2), (2,3), (3,2), (3,3) from A11
        # B entries: (0,0), (0,1), (1,0), (1,1) from B00; (2,0), (2,1), (3,0), (3,1) from B11
        # But the product is structured. Let me use the standard expansion.
        #
        # Actually, the simplest correct approach: use the known fact that 4x4*4x4 has rank at most 28
        # via Strassen, and just generate the terms by simulating the Strassen algorithm.

        # For each Strassen M, I'll define the 2x2 matrix X (from A) and Y (from B) as 2x2 arrays
        # of (global_i, global_j) or (global_j, global_k) indices with coefficients.
        # Then expand X*Y as 4 scalar products.

        def expand_2x2_product(X, Y, i_offset=0, j_offset=0, k_offset=0):
            """X: 2x2 list of (i, j, coeff) tuples for A entries
               Y: 2x2 list of (j, k, coeff) tuples for B entries
               Returns list of (i, j, k, coeff) for scalar products."""
            result = []
            for a in range(2):
                for b in range(2):
                    # C[a,b] = sum_j X[a,j] * Y[j,b]
                    for j in range(2):
                        # X[a,j] is (i, j_idx, coeff_x)
                        i_idx, j_idx, cx = X[a][j]
                        # Y[j,b] is (j_idx2, k_idx, coeff_y)
                        j_idx2, k_idx, cy = Y[j][b]
                        # j_idx should equal j_idx2 for consistency
                        result.append((i_idx, j_idx, k_idx, cx * cy))
            return result

        # M1: X = A00+A11 (2x2), Y = B00+B11 (2x2)
        # X[a,j]: a=0,1; j=0,1
        # A00: A[0+j, 0+j2] for j,j2 in 0,1 -> X[0][0]=A[0,0], X[0][1]=A[0,1], X[1][0]=A[1,0], X[1][1]=A[1,1]
        # A11: A[2+j, 2+j2] -> adds to X[0][0]=A[2,2], etc.
        # So X is a 2x2 matrix where each entry is a sum of 2 A entries.
        # This means X[a,j] is not a single scalar but a sum. We need to expand.
        #
        # I think the cleanest is to just list all 28 (i,j,k) tuples directly.
        # Let me use a lookup table.

        # For 4x4*4x4 Strassen, the 28 terms are:
        # M1: 4 terms from A00*B00, 4 from A00*B11, 4 from A11*B00, 4 from A11*B11 -> but that's 16, not 4.
        # I'm confusing myself. Let me just use schoolbook for 4x4*4x4 (16 terms) and accept rank 16*2+16=48.
        # Or better: use the known optimal for small cases.

        # Let me just generate schoolbook for each 4x4 block and the 4x1 block.
        # 4x4*4x4 schoolbook: 16 terms. Two of those: 32. Plus 4x1: 16. Total: 48.
        # That's better than 144.

        terms = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    terms.append((i, j, B_col_offset + k, 1))
        return terms

    # Use schoolbook for each 4x4 block of B
    all_terms = []
    all_terms += strassen_4x4_terms(0)  # B cols 0-3
    all_terms += strassen_4x4_terms(4)  # B cols 4-7
    for i in range(n):
        for j in range(m):
            all_terms.append((i, j, 8, 1))  # B col 8

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

def local_search_from_schoolbook():
    """Start from schoolbook (rank 144) and try to reduce rank by merging terms."""
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

    # Try to find pairs of terms that can be merged using Strassen-like identities.
    # For now, just return the schoolbook as a baseline.
    # A more sophisticated approach would use linear algebra to find dependencies.
    yield r, U, V, W

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
    for gen in [schoolbook_candidates, block_structured_candidates, local_search_from_schoolbook]:
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
