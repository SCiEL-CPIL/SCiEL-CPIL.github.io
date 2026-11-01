"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    Nm = n * m  # 6
    Mp = m * p  # 12
    Pn = p * n  # 8

    def build_schoolbook():
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(Nm)]
        V = [[Fraction(0)] * rank for _ in range(Mp)]
        W = [[Fraction(0)] * rank for _ in range(Pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = Fraction(1)
                    V[j * p + k][c] = Fraction(1)
                    W[k * n + i][c] = Fraction(1)
                    c += 1
        return U, V, W

    def verify(U, V, W):
        """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == delta_ij * delta_kk' ... 
        Actually the tensor: T[i,j,k] (input A[i,j], B[j,k]) -> output C[i,k]
        Decomposition: C[i,k] = sum_c U[i,k... ] wait.
        
        Convention: W[k*n+i, r]. U shape (n*m, rank), V (m*p, rank), W (p*n, rank).
        The multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k].
        Tensor T_{i,j,k} where the three modes are (i in [n], j in [m], k in [p]).
        Decomposition rank r: T_{i,j,k} = sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c}.
        Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
        """
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != Fraction(1):
                        return False
        return True

    def to_int_or_half(mat):
        """Convert a matrix of Fractions to a list of lists of int or half-int (float)."""
        out = []
        for row in mat:
            newrow = []
            for val in row:
                # val is Fraction; check if it is integer or half-integer
                num, den = val.numerator, val.denominator
                if den == 1:
                    newrow.append(int(num))
                elif den == 2:
                    newrow.append(Fraction(num, den) * 1)  # keep as float half
                    newrow[-1] = float(num) / 2
                else:
                    # not allowed; return None
                    return None
            out.append(newrow)
        return out

    # 1) Try the schoolbook, then greedily remove zero-effect columns.
    U, V, W = build_schoolbook()
    rank = n * m * p

    # Greedy elimination: remove a column if the decomposition still holds.
    def try_remove(U, V, W, c):
        r = len(U[0])
        U2 = [row[:c] + row[c+1:] for row in U]
        V2 = [row[:c] + row[c+1:] for row in V]
        W2 = [row[:c] + row[c+1:] for row in W]
        return U2, V2, W2

    # In pure schoolbook no column is removable (each column is unique). But we can
    # try a linear-combination reduction: combine two columns that have overlapping
    # support. This is the Strassen idea.

    # Known result: 2x3x4 can be done in rank 21. Let's attempt a constructive
    # reduction using Strassen-style identities on the 3x4 part (which can be
    # reduced) combined with the 2x3 part.

    # Strategy: represent the 3x4 multiplication in rank 10 (known minimal for 3x4? 
    # Actually 3x3 is 7, 3x4 minimal is 10? Let's just search).
    # Simpler: brute-force search over small rank with random rational initializations
    # is too slow. Instead use a known explicit 21-rank construction.

    # We'll do a greedy column-merging: find two columns c1,c2 such that their
    # "pattern" allows merging. For schoolbook, column c corresponds to (i,j,k).
    # Merging (i1,j1,k1) and (i2,j2,k2) is possible if they share structure.

    # Given time, the most reliable improvement is to return the schoolbook rank 24
    # unless we can quickly find a 21-rank. Let's attempt a targeted construction.

    # --- Targeted 21-rank construction for 2x3x4 ---
    # Use the fact that 2x3 = (2x2)+(2x1) and 3x4 can use Strassen.
    # We'll attempt: split the 3x3 core into Strassen (7) plus extra column for the 4th.
    # This is complex; instead we run a small exact search for rank 21 using
    # a parametric family and verify.

    # Parametric family: start from schoolbook and set up a system where we allow
    # some columns to be linear combinations. We'll use a randomized search over
    # the space of rank-21 decompositions with small rational entries.

    import random
    random.seed(42)

    def random_rational(bound=2):
        return Fraction(random.randint(-bound, bound))

    def search_rank(r, iterations=20000):
        # We'll fix a structure: take the schoolbook 24 columns and try to express
        # 3 of them as combinations of the other 21. Equivalently, find rank r by
        # solving for U,V,W with r columns.
        # Use a greedy: pick r random (i,j,k) "anchors" is wrong.
        # Instead: random init then verify is too slow to hit exact.
        # Better: use the known explicit construction below.
        return None

    # Explicit known 21-rank for 2x3x4:
    # We build it by reducing the 3x4 block. The 3x4 multiplication tensor has
    # rank 10 (minimal). Combined with the 2 on the left in a tensor-product-like
    # way but optimized gives 21. 
    # Construction: 
    #   Left 2x3: rows i=0,1; cols j=0,1,2.
    #   Right 3x4: rows j=0,1,2; cols k=0,1,2,3.
    #   Output 2x4: rows i; cols k.
    #
    # We'll use 3 Strassen columns for a 2x2 sub-block and handle the rest.
    # Given the complexity and time, we fall back to schoolbook if we cannot
    # verify a lower rank quickly.

    # Let's attempt a quick exact greedy: try to reduce schoolbook by merging
    # columns that share (i) or (k).
    # Actually, simplest reliable improvement: none guaranteed. Return schoolbook.
    # But we want to beat 1.0, so we MUST find rank < 24. The known bound is 21.

    # We'll do a focused search: assume rank 21 and solve via the "border rank"
    # is not exact. We need exact. 
    #
    # Final decision: implement a randomized search over rank 21 with the
    # constraint that entries are in {-1,0,1,1/2} and verify. Use many random
    # seeds but bounded by deadline.

    def random_decomposition(r, entry_set):
        U = [[random.choice(entry_set) for _ in range(r)] for _ in range(Nm)]
        V = [[random.choice(entry_set) for _ in range(r)] for _ in range(Mp)]
        W = [[random.choice(entry_set) for _ in range(r)] for _ in range(Pn)]
        return U, V, W

    entry_set = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]

    best = None
    for target_rank in [21, 22, 23]:
        if time.time() > deadline:
            break
        # Try random search
        for it in range(2000):
            if time.time() > deadline:
                break
            U, V, W = random_decomposition(target_rank, entry_set)
            if verify(U, V, W):
                conv = to_int_or_half(U), to_int_or_half(V), to_int_or_half(W)
                if None not in conv:
                    best = (target_rank, conv[0], conv[1], conv[2])
                    break
        if best is not None:
            break

    if best is not None:
        return {"U": best[1], "V": best[2], "W": best[3]}

    # Fallback: schoolbook (rank 24)
    convU = to_int_or_half(U)
    convV = to_int_or_half(V)
    convW = to_int_or_half(W)
    if None not in (convU, convV, convW):
        return {"U": convU, "V": convV, "W": convW}

    # Last resort: return as floats (should be ints)
    Uf = [[float(x) for x in row] for row in U]
    Vf = [[float(x) for x in row] for row in V]
    Wf = [[float(x) for x in row] for row in W]
    return {"U": Uf, "V": Vf, "W": Wf}
# EVOLVE-BLOCK-END
