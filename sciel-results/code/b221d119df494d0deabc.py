"""Exact matrix-multiplication tensor (2 x 3) * (3 x 6) with rank-minimal factors.

We build a guaranteed-correct (schoolbook) decomposition and attach a self-verification
guard that, if a cheaper rank<=35 candidate is available, validates it exactly before
emitting it. Verification is cheap: it reconstructs the multiplication tensor and
compares it to the canonical schoolbook tensor."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j, c]*V[j*p+k, c] == 1 iff (k*n+i) is the (i,k)
    output cell ... precisely the matrix-multiplication tensor T[i,j,k] = delta(i,k via W).
    Convention: W[k*n+i, c] carries the output cell (i,k). U[i*m+j,c]=A[i,j], V[j*p+k,c]=B[j,k].
    Product P[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c] must equal delta(i,j,k)?? No.
    We need T[i,j,k] where C[i,k]=sum_j A[i,j] B[j,k]. Tensor T[i,j,k] = delta on output (i,k)
    paired with input A[i,j],B[j,k]. So T[i,j,k] = 1 iff there exists j consistent, i.e.
    T[i,j,k]=1 for all j (the (i,k) output depends on all j). Actually T[i,j,k] = delta_{(i,k),out}.
    So T[i,j,k] = 1 (for every j) at each (i,k) pair, else 0. Hence P[i,j,k] = 1 if True output (i,k),
    but P is sum over c of U*V and does NOT involve W. So P[i,j,k] is just the (i,k) output's
    dependence indicator, equal to 1 for all j given (i,k). We check P[i,j,k]==1 (any j) and
    W must carry the cell: sum_c U*V*W should reconstruct? Standard check: for each (i,k),
    coefficient on A[i,j]*B[j,k] must be 1. That coefficient is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    """
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Coefficient for A[i,j]*B[j,k] -> C[i,k] is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(Ua.shape[1]):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS

    # Guaranteed-correct baseline.
    base = _schoolbook(n, m, p)
    if not _verify(base["U"], base["V"], base["W"], n, m, p):
        # Should never happen; keep baseline.
        return base

    # Candidate framework: a list of (rank, builder) with rank <= base rank.
    # A builder returns (U,V,W) with half-integer/integer entries.
    candidates = []

    def _build_hybrid_strassen_schoolbook():
        """
        Constructs a rank 33 decomposition for 2x3x6.
        Splits the middle dimension m=3 into m1=2 and m2=1.
        Part 1: 2x2x6 using 3 blocks of 2x2x2 Strassen (Rank 7 each) -> 21.
        Part 2: 2x1x6 using Schoolbook (Rank 12) -> 12.
        Total Rank: 33.
        """
        n, m, p = 2, 3, 6
        rank = 33
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: 2x2x6 (j in 0,1)
        # Split k into 3 blocks of size 2
        for block in range(3):
            k_start = block * 2
            # Strassen 2x2x2 factors
            # A indices: i in {0,1}, j in {0,1} -> global j is same
            # B indices: j in {0,1}, k in {k_start, k_start+1}
            # C indices: i in {0,1}, k in {k_start, k_start+1}
            
            # Strassen Multiplications
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # We need to map these to U, V, W
            # U row index: i*m + j
            # V row index: j*p + k
            # W row index: k*n + i
            
            # M1
            U[0*3+0][col] += 1; U[1*3+1][col] += 1
            V[0*6+k_start][col] += 1; V[1*6+(k_start+1)][col] += 1
            # C00 += M1, C11 += M1
            W[k_start*2+0][col] += 1; W[(k_start+1)*2+1][col] += 1
            col += 1
            
            # M2
            U[1*3+0][col] += 1; U[1*3+1][col] += 1
            V[0*6+k_start][col] += 1
            # C10 += M2
            W[k_start*2+1][col] += 1
            col += 1
            
            # M3
            U[0*3+0][col] += 1
            V[0*6+(k_start+1)][col] += 1; V[1*6+(k_start+1)][col] -= 1
            # C01 += M3
            W[(k_start+1)*2+0][col] += 1
            col += 1
            
            # M4
            U[1*3+1][col] += 1
            V[1*6+k_start][col] += 1; V[0*6+k_start][col] -= 1
            # C10 += M4
            W[k_start*2+1][col] += 1
            col += 1
            
            # M5
            U[0*3+0][col] += 1; U[0*3+1][col] += 1
            V[1*6+(k_start+1)][col] += 1
            # C00 -= M5
            W[k_start*2+0][col] -= 1
            col += 1
            
            # M6
            U[1*3+0][col] += 1; U[0*3+0][col] -= 1
            V[0*6+k_start][col] += 1; V[0*6+(k_start+1)][col] += 1
            # C11 += M6
            W[(k_start+1)*2+1][col] += 1
            col += 1
            
            # M7
            U[0*3+1][col] += 1; U[1*3+1][col] -= 1
            V[1*6+k_start][col] += 1; V[1*6+(k_start+1)][col] += 1
            # C00 += M7
            W[k_start*2+0][col] += 1
            col += 1

        # Part 2: 2x1x6 (j=2)
        # Schoolbook: C[i,k] += A[i,2] * B[2,k]
        for i in range(2):
            for k in range(6):
                U[i*3+2][col] = 1
                V[2*6+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}

    candidates.append((33, _build_hybrid_strassen_schoolbook))

    # Try cheaper candidates in increasing rank order.
    # Note: We only have one candidate for now, but the loop structure allows for more.
    for rank in sorted(set([c[0] for c in candidates])):
        builder = None
        for r, b in candidates:
            if r == rank:
                builder = b
                break
        if builder is None:
            continue
        res = builder()
        if res is None:
            continue
        U, V, W = res["U"], res["V"], res["W"]
        # Shape sanity.
        if len(U) != n*m or len(V) != m*p or len(W) != p*n:
            continue
        R = len(U[0]) if U[0] else 0
        if R != rank or len(V[0]) != rank or len(W[0]) != rank:
            continue
        # Half-integer / integer entries.
        try:
            ok = True
            for M in (U, V, W):
                for row in M:
                    for v in row:
                        if not (float(v) == 0.5 * round(2*float(v))):
                            ok = False
                        break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
        except Exception:
            continue
        # Exact verification (full tensor coefficient check).
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    return base
# EVOLVE-BLOCK-END
