"""Matrix multiplication 2x2 by 2x7 - rank-minimal decomposition."""
import time

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Construct a rank-25 tensor decomposition for M(2,2,7).
    
    Uses Strassen's algorithm (rank 7) for three 2x2x2 blocks (cols 0-1, 2-3, 4-5)
    and schoolbook (rank 4) for the remaining 2x2x1 block (col 6).
    Total rank: 7 + 7 + 7 + 4 = 25.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7

    rank = 25
    U = [[0.0] * rank for _ in range(n * m)]   # (4, 25)
    V = [[0.0] * rank for _ in range(m * p)]   # (14, 25)
    W = [[0.0] * rank for _ in range(p * n)]   # (14, 25)

    r = 0

    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U indices: 0:(0,0)=a, 1:(0,1)=b, 2:(1,0)=c, 3:(1,1)=d
    # V indices: 0:(0,0)=e, 1:(0,1)=f, 2:(1,0)=g, 3:(1,1)=h
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> for C
    
    def add_strassen_block(col_offset):
        nonlocal r
        # m1 = (a+d)(e+h)
        # U: a=1, d=1; V: e=1, h=1; W: C00+=1, C01+=1, C10+=1, C11+=1? No.
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Term 1: (a+d)(e+h)
        U[0][r] = 1.0; U[3][r] = 1.0
        V[col_offset][r] = 1.0; V[col_offset+1][r] = 1.0 # e, h? No.
        # V index for (0, col_offset) is 0*p + col_offset = col_offset
        # V index for (1, col_offset+1) is 1*p + col_offset+1 = p + col_offset + 1
        # Wait, V is m x p. Index j*p + k.
        # e is B[0, col_offset] -> V[col_offset]
        # h is B[1, col_offset+1] -> V[p + col_offset + 1]
        V[col_offset][r] = 1.0
        V[p + col_offset + 1][r] = 1.0
        # W: C[0, col_offset] += 1, C[0, col_offset+1] += 1, C[1, col_offset] += 1, C[1, col_offset+1] += 1
        # W index k*n + i.
        # C[0, col_offset] -> W[col_offset*n + 0]
        # C[0, col_offset+1] -> W[(col_offset+1)*n + 0]
        # C[1, col_offset] -> W[col_offset*n + 1]
        # C[1, col_offset+1] -> W[(col_offset+1)*n + 1]
        W[col_offset*n][r] = 1.0
        W[(col_offset+1)*n][r] = 1.0
        W[col_offset*n + 1][r] = 1.0
        W[(col_offset+1)*n + 1][r] = 1.0
        r += 1

        # m2 = (c+d)e
        U[2][r] = 1.0; U[3][r] = 1.0
        V[col_offset][r] = 1.0
        W[col_offset*n + 1][r] = 1.0 # C10 += m2
        W[(col_offset+1)*n + 1][r] = -1.0 # C11 -= m2
        r += 1

        # m3 = a(f-h)
        U[0][r] = 1.0
        V[col_offset+1][r] = 1.0 # f is B[0, col_offset+1] -> V[col_offset+1]
        V[p + col_offset + 1][r] = -1.0 # h is B[1, col_offset+1] -> V[p + col_offset + 1]
        W[col_offset+1][r] = 1.0 # C01 += m3
        W[(col_offset+1)*n + 1][r] = 1.0 # C11 += m3
        r += 1

        # m4 = d(g-e)
        U[3][r] = 1.0
        V[p + col_offset][r] = 1.0 # g is B[1, col_offset] -> V[p + col_offset]
        V[col_offset][r] = -1.0 # e
        W[col_offset*n][r] = 1.0 # C00 += m4
        W[col_offset*n + 1][r] = 1.0 # C10 += m4
        r += 1

        # m5 = (a+b)h
        U[0][r] = 1.0; U[1][r] = 1.0
        V[p + col_offset + 1][r] = 1.0 # h
        W[col_offset*n][r] = -1.0 # C00 -= m5
        W[col_offset+1][r] = 1.0 # C01 += m5
        r += 1

        # m6 = (c-a)(e+f)
        U[2][r] = 1.0; U[0][r] = -1.0
        V[col_offset][r] = 1.0 # e
        V[col_offset+1][r] = 1.0 # f
        W[(col_offset+1)*n + 1][r] = 1.0 # C11 += m6
        r += 1

        # m7 = (b-d)(g+h)
        U[1][r] = 1.0; U[3][r] = -1.0
        V[p + col_offset][r] = 1.0 # g
        V[p + col_offset + 1][r] = 1.0 # h
        W[col_offset*n][r] = 1.0 # C00 += m7
        r += 1

    def add_schoolbook_block(col_offset):
        nonlocal r
        # C[0, col] = A[0,0]B[0,col] + A[0,1]B[1,col]
        # C[1, col] = A[1,0]B[0,col] + A[1,1]B[1,col]
        
        # Term 1: A[0,0] * B[0, col]
        U[0][r] = 1.0
        V[col_offset][r] = 1.0
        W[col_offset*n][r] = 1.0
        r += 1

        # Term 2: A[0,1] * B[1, col]
        U[1][r] = 1.0
        V[p + col_offset][r] = 1.0
        W[col_offset*n][r] = 1.0
        r += 1

        # Term 3: A[1,0] * B[0, col]
        U[2][r] = 1.0
        V[col_offset][r] = 1.0
        W[col_offset*n + 1][r] = 1.0
        r += 1

        # Term 4: A[1,1] * B[1, col]
        U[3][r] = 1.0
        V[p + col_offset][r] = 1.0
        W[col_offset*n + 1][r] = 1.0
        r += 1

    # Apply blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    add_schoolbook_block(6)

    return {"U": U, "V": V, "W": W}
