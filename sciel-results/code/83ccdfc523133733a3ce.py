"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists of Fractions for schoolbook decomposition."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _combine(U1, V1, W1, U2, V2, W2, n, m, p):
    """
    Combine two decompositions of the same (n,m,p) multiplication tensor.
    The combined decomposition has rank r1 + r2 and is valid if both are valid.
    """
    r1 = len(W1[0]) if W1 else 0
    r2 = len(W2[0]) if W2 else 0
    total_rank = r1 + r2
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]
    for r in range(r1):
        for i in range(n * m):
            U[i][r] = U1[i][r]
        for j in range(m * p):
            V[j][r] = V1[j][r]
        for k in range(p * n):
            W[k][r] = W1[k][r]
    for r in range(r2):
        for i in range(n * m):
            U[i][r1 + r] = U2[i][r]
        for j in range(m * p):
            V[j][r1 + r] = V2[j][r]
        for k in range(p * n):
            W[k][r1 + r] = W2[k][r]
    return U, V, W


def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 @ 2x2.
    Returns (U, V, W) with rank 7.
    Convention: A is 2x2, B is 2x2, C = A@B is 2x2.
    U index: row = i*m + j = i*2 + j, col = r
    V index: row = j*p + k = j*2 + k, col = r
    W index: row = k*n + i = k*2 + i, col = r
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[Fraction(0)] * rank for _ in range(4)]
    V = [[Fraction(0)] * rank for _ in range(4)]
    W = [[Fraction(0)] * rank for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (a+b)(e+f)
    # M2 = (a+b)(g+h)
    # M3 = a(d-c)
    # M4 = e(h-f)
    # M5 = (a+d)(f+h)
    # M6 = (c+b)(e+d)
    # M7 = (a-c)(g+h)
    # C11 = M5 + M4 - M6 - M7
    # C12 = M1 + M7
    # C21 = M3 + M6
    # C22 = M5 + M1 - M3 - M4
    
    # A = [[a, b], [c, d]] so A[i,j]: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B = [[e, f], [g, h]] so B[j,k]: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C = [[C11, C12], [C21, C22]] so C[i,k]: C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    
    # U rows: 0=a, 1=b, 2=c, 3=d
    # V rows: 0=e, 1=f, 2=g, 3=h
    # W rows: 0=C11 (k=0,i=0), 1=C12 (k=1,i=0), 2=C21 (k=0,i=1), 3=C22 (k=1,i=1)
    
    # M1 = (a+b)(e+f)
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[1][0] = 1  # C12
    
    # M2 = (a+b)(g+h)
    U[0][1] = 1; U[1][1] = 1
    V[2][1] = 1; V[3][1] = 1
    W[0][1] = 1  # C11... wait, M2 contributes to C11? Let me recheck.
    # Actually: C11 = M5 + M4 - M6 - M7, C12 = M1 + M7, C21 = M3 + M6, C22 = M5 + M1 - M3 - M4
    # M2 is not used in the standard Strassen for C. Let me re-derive.
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me use this standard formulation.
    # A = [[A11, A12], [A21, A22]] = [[a, b], [c, d]]
    # B = [[B11, B12], [B21, B22]] = [[e, f], [g, h]]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me redo:
    U = [[Fraction(0)] * rank for _ in range(4)]
    V = [[Fraction(0)] * rank for _ in range(4)]
    W = [[Fraction(0)] * rank for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 and C22
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = a(g-h)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 += M3, C22 += M3
    
    # M4 = d(f-e)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1  # C11 -= M4, C21 += M4
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1  # C11 += M5, C12 += M5
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += M6
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1  # C11 -= M7
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact: C[i,k] = sum_j A[i,j]B[j,k] for all i,j,k."""
    rank = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Coefficient of A[i,j] * B[j,k] in C[i,k] should be 1
                # Coefficient in C[i',k'] for (i',k') != (i,k) should be 0
                total = Fraction(0)
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != Fraction(1):
                    return False
    # Check that cross terms vanish
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        if ip == i and kp == k:
                            continue
                        total = Fraction(0)
                        for r in range(rank):
                            total += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        if total != Fraction(0):
                            return False
    return True


def _block_decompose(n, m, p):
    """
    Recursive block decomposition: split into 2x2 blocks where possible,
    use Strassen for 2x2x2 sub-blocks, schoolbook for remainders.
    """
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2_2x2()
    
    # Try to split each dimension into 2 + remainder
    # For simplicity, if any dimension is even, split it into 2 + (d-2)
    # If all are odd, use schoolbook for the whole thing
    
    # Strategy: try all ways to partition n into n1+n2, m into m1+m2, p into p1+p2
    # where each part is >= 1, and recursively decompose each block.
    # The full decomposition is the sum of all block decompositions.
    
    best = None
    best_rank = n * m * p  # schoolbook rank
    
    # Get schoolbook as baseline
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U_s, V_s, W_s)
    
    # Try block decompositions
    for n1 in range(1, n):
        n2 = n - n1
        for m1 in range(1, m):
            m2 = m - m1
            for p1 in range(1, p):
                p2 = p - p1
                # Recursively decompose each of the 8 blocks
                try:
                    blocks = []
                    for (ni, mi, pi) in [(n1, m1, p1), (n1, m1, p2), (n1, m2, p1), (n1, m2, p2),
                                          (n2, m1, p1), (n2, m1, p2), (n2, m2, p1), (n2, m2, p2)]:
                        if ni == 2 and mi == 2 and pi == 2:
                            blocks.append(_strassen_2x2_2x2())
                        else:
                            blocks.append(_schoolbook(ni, mi, pi))
                    
                    total_rank = sum(len(b[0][0]) for b in blocks)
                    if total_rank < best_rank:
                        # Combine all blocks
                        U = [[Fraction(0)] * total_rank for _ in range(n * m)]
                        V = [[Fraction(0)] * total_rank for _ in range(m * p)]
                        W = [[Fraction(0)] * total_rank for _ in range(p * n)]
                        
                        col = 0
                        # Block (n1,m1,p1): A[0:n1, 0:m1] @ B[0:m1, 0:p1] -> C[0:n1, 0:p1]
                        for bi, (ni, mi, pi) in enumerate([(n1, m1, p1), (n1, m1, p2), (n1, m2, p1), (n1, m2, p2),
                                                          (n2, m1, p1), (n2, m1, p2), (n2, m2, p1), (n2, m2, p2)]):
                            i_off = 0 if bi < 4 else n1
                            j_off = 0 if bi % 4 < 2 else m1
                            k_off = 0 if bi % 2 == 0 else p1
                            
                            Ub, Vb, Wb = blocks[bi]
                            rb = len(Ub[0])
                            for r in range(rb):
                                for ii in range(ni):
                                    U[i_off + ii * mi + 0 * mi + 0][col + r] = Ub[ii * mi][r] if False else 0
                            # This indexing is getting complex. Let me use a simpler approach.
                        
                        # Actually, the block structure means:
                        # Block (i_off, j_off, k_off) with sizes (ni, mi, pi)
                        # U contribution: rows (i_off*mi + ii*mi + jj) for ii in [0,ni), jj in [0,mi)
                        # But this is the full U matrix of size (n*m) x rank
                        # For block at (i_off, j_off), the U rows are at indices:
                        #   row = (i_off + ii) * m + (j_off + jj) for ii in [0,ni), jj in [0,mi)
                        # V rows: (j_off + jj) * p + (k_off + kk) for jj in [0,mi), kk in [0,pi)
                        # W rows: (k_off + kk) * n + (i_off + ii) for kk in [0,pi), ii in [0,ni)
                        
                        U = [[Fraction(0)] * total_rank for _ in range(n * m)]
                        V = [[Fraction(0)] * total_rank for _ in range(m * p)]
                        W = [[Fraction(0)] * total_rank for _ in range(p * n)]
                        
                        col = 0
                        block_defs = [
                            (0, 0, 0, n1, m1, p1),
                            (0, 0, p2, n1, m1, p2),
                            (0, m2, 0, n1, m2, p1),
                            (0, m2, p2, n1, m2, p2),
                            (n2, 0, 0, n2, m1, p1),
                            (n2, 0, p2, n2, m1, p2),
                            (n2, m2, 0, n2, m2, p1),
                            (n2, m2, p2, n2, m2, p2),
                        ]
                        for bi, (i_off, j_off, k_off, ni, mi, pi) in enumerate(block_defs):
                            Ub, Vb, Wb = blocks[bi]
                            rb = len(Ub[0])
                            for r in range(rb):
                                for ii in range(ni):
                                    for jj in range(mi):
                                        U[(i_off + ii) * m + (j_off + jj)][col + r] = Ub[ii * mi + jj][r]
                                for jj in range(mi):
                                    for kk in range(pi):
                                        V[(j_off + jj) * p + (k_off + kk)][col + r] = Vb[jj * pi + kk][r]
                                for kk in range(pi):
                                    for ii in range(ni):
                                        W[(k_off + kk) * n + (i_off + ii)][col + r] = Wb[kk * ni + ii][r]
                            col += rb
                        
                        if _verify(U, V, W, n, m, p):
                            best_rank = total_rank
                            best = (U, V, W)
                except Exception:
                    pass
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    
    # Get schoolbook as baseline
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best = (U_s, V_s, W_s)
    best_rank = n * m * p
    
    # Try block decomposition
    if time.time() < deadline - 10:
        try:
            U_b, V_b, W_b = _block_decompose(n, m, p)
            if len(W_b[0]) < best_rank:
                best = (U_b, V_b, W_b)
                best_rank = len(W_b[0])
        except Exception:
            pass
    
    U, V, W = best
    rank = len(W[0])
    
    # Convert to lists of ints/floats
    U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in U]
    V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in V]
    W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
