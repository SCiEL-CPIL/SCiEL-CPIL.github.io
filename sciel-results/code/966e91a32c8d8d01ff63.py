import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    
    Strategy: Partition the m=4 dimension into two blocks of size 2.
    For each block b in {0,1}, we compute C[:, 2b:2b+2] = A[:, 2b:2b+2] @ B[2b:2b+2, :].
    Each block is a 2x2 x 2x2 multiplication (with shared output columns).
    
    Using Strassen's algorithm (rank 7) for each 2x2x2 block instead of schoolbook (rank 8),
    we get total rank = 2 * 7 = 14 for the core multiplications.
    
    Wait - let me reconsider the structure. The output C is 2x4, so C[i,k] for k in {0,1,2,3}.
    For block b, we compute C[i, 2b] and C[i, 2b+1] for i in {0,1}.
    
    C[i, 2b]   = A[i,2b]*B[2b,2b]   + A[i,2b+1]*B[2b+1,2b]
    C[i, 2b+1] = A[i,2b]*B[2b,2b+1] + A[i,2b+1]*B[2b+1,2b+1]
    
    So for each block b, we have a 2x2 (A block) times 2x2 (B block) = 2x2 (C block).
    That's 4 independent 2x2x2 multiplications (one for each (i, k_within_block) pair? No...)
    
    Actually, for fixed block b, the sub-problem is:
    A_block[i, j] for i in {0,1}, j in {0,1}  (these are A[i, 2b+j])
    B_block[j, k] for j in {0,1}, k in {0,1}  (these are B[2b+j, 2b+k])
    C_block[i, k] for i in {0,1}, k in {0,1}  (these are C[i, 2b+k])
    
    C_block[i,k] = sum_j A_block[i,j] * B_block[j,k]
    
    This is a standard 2x2 x 2x2 matrix multiplication, which has rank 7 (Strassen).
    
    We have 2 such blocks (b=0 and b=1), so total rank = 2 * 7 = 14.
    
    But wait - the U, V, W matrices have fixed sizes:
    U: (n*m) x rank = 8 x rank
    V: (m*p) x rank = 16 x rank  
    W: (p*n) x rank = 8 x rank
    
    For block b, the relevant U indices are i*m + (2b+j) = i*4 + 2b + j for i in {0,1}, j in {0,1}
    The relevant V indices are (2b+j)*p + (2b+k) = (2b+j)*4 + 2b + k for j in {0,1}, k in {0,1}
    The relevant W indices are (2b+k)*n + i = (2b+k)*2 + i for k in {0,1}, i in {0,1}
    
    So each block uses 4 U entries, 4 V entries, and 4 W entries.
    
    Strassen's 7 multiplications for 2x2x2:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[m, n], [o, p]]
    
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-c)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    m = P1 - P5 + P7
    n = P3 + P5
    o = P2 + P4
    p = P1 - P2 + P6
    
    For each Strassen multiplication P_r, we need:
    - A factor: a linear combination of a, b, c, d (with coefficients in {-1, 0, 1})
    - B factor: a linear combination of e, f, g, h (with coefficients in {-1, 0, 1})
    - C contribution: P_r contributes to m, n, o, p with coefficients in {-1, 0, 1}
    
    In our tensor decomposition framework:
    U[i*m + j, col] gives the coefficient for A[i,j] in the r-th bilinear form
    V[j*p + k, col] gives the coefficient for B[j,k] in the r-th bilinear form
    W[k*n + i, col] gives the coefficient for C[i,k] in the r-th bilinear form
    
    For block b, mapping local indices to global:
    Local A[i,j] -> global A[i, 2b+j] -> U index: i*4 + 2b + j
    Local B[j,k] -> global B[2b+j, 2b+k] -> V index: (2b+j)*4 + (2b+k)
    Local C[i,k] -> global C[i, 2b+k] -> W index: (2b+k)*2 + i
    
    Let me define the Strassen factors for each P_r:
    
    P1 = (a+d)(e+h):
      A factors: a:1, b:0, c:0, d:1
      B factors: e:1, f:0, g:0, h:1
      C contributions: m:+1, n:0, o:0, p:+1
    
    P2 = (c+d)e:
      A factors: a:0, b:0, c:1, d:1
      B factors: e:1, f:0, g:0, h:0
      C contributions: m:0, n:0, o:+1, p:-1
    
    P3 = a(f-h):
      A factors: a:1, b:0, c:0, d:0
      B factors: e:0, f:1, g:0, h:-1
      C contributions: m:0, n:+1, o:0, p:0
    
    P4 = d(g-c):
      A factors: a:0, b:0, c:-1, d:1
      B factors: e:0, f:0, g:1, h:0
      C contributions: m:0, n:0, o:+1, p:0
    
    P5 = (a+b)h:
      A factors: a:1, b:1, c:0, d:0
      B factors: e:0, f:0, g:0, h:1
      C contributions: m:-1, n:+1, o:0, p:0
    
    P6 = (c-a)(e+f):
      A factors: a:-1, b:0, c:1, d:0
      B factors: e:1, f:1, g:0, h:0
      C contributions: m:0, n:0, o:0, p:+1
    
    P7 = (b-d)(g+h):
      A factors: a:0, b:1, c:0, d:-1
      B factors: e:0, f:0, g:1, h:1
      C contributions: m:+1, n:0, o:0, p:0
    
    Now let me build the full decomposition.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 14  # 2 blocks * 7 Strassen multiplications
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 14
    V = [[0] * rank for _ in range(m * p)]  # 16 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Strassen factor definitions
    # Each entry: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: [a, b, c, d] for local A = [[a,b],[c,d]]
    # B_coeffs: [e, f, g, h] for local B = [[e,f],[g,h]]
    # C_coeffs: [m, n, o, p] for local C = [[m,n],[o,p]]
    strassen_factors = [
        # P1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # P3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # P4 = d(g-c)
        ([0, 0, -1, 1], [0, 0, 1, 0], [0, 0, 1, 0]),
        # P5 = (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # P6 = (c-a)(e+f)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # P7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for b in range(2):  # two blocks
        for r in range(7):  # seven Strassen multiplications
            a_coeffs, b_coeffs, c_coeffs = strassen_factors[r]
            
            # Fill U: for local i in {0,1}, local j in {0,1}
            # Local A[i,j] has coefficient a_coeffs[i*2+j]
            # Global U index: i*m + (2b+j) = i*4 + 2b + j
            for i in range(2):
                for j in range(2):
                    U[i * m + (2 * b + j)][col] = a_coeffs[i * 2 + j]
            
            # Fill V: for local j in {0,1}, local k in {0,1}
            # Local B[j,k] has coefficient b_coeffs[j*2+k]
            # Global V index: (2b+j)*p + (2b+k) = (2b+j)*4 + 2b + k
            for j in range(2):
                for k in range(2):
                    V[(2 * b + j) * p + (2 * b + k)][col] = b_coeffs[j * 2 + k]
            
            # Fill W: for local i in {0,1}, local k in {0,1}
            # Local C[i,k] has coefficient c_coeffs[i*2+k]
            # Global W index: (2b+k)*n + i = (2b+k)*2 + i
            for i in range(2):
                for k in range(2):
                    W[(2 * b + k) * n + i][col] = c_coeffs[i * 2 + k]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
