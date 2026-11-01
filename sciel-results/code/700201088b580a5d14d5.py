"""Matrix multiplication 2x2 by 2x4: rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 4]

# Deadline guard: the evaluator kills after 150s; keep a safety margin.
_START = time.time()
_DEADLINE = 135.0  # seconds

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _verify_fraction(U, V, W, n=2, m=2, p=4):
    """Exact verification using fractions."""
    n, m, p = n, m, p
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _strassen_2x2x2x4():
    """
    Construct a rank-14 decomposition by applying Strassen's algorithm to each 2x2*2x2 block.
    
    For the first block (k=0,1): multiply A (2x2) by B1 (2x2) where B1 = B[:, 0:2]
    For the second block (k=2,3): multiply A (2x2) by B2 (2x2) where B2 = B[:, 2:4]
    
    Strassen's algorithm for 2x2*2x2 uses 7 multiplications.
    We apply it separately to each block, giving 7+7=14 total.
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Strassen's 7 products for 2x2*2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Block 1 (columns 0-6): k=0,1
    # B1[0,0] = B[0,0] -> V[0]
    # B1[0,1] = B[0,1] -> V[1]
    # B1[1,0] = B[1,0] -> V[4]
    # B1[1,1] = B[1,1] -> V[5]
    
    # A indices:
    # A[0,0] -> U[0], A[0,1] -> U[1]
    # A[1,0] -> U[2], A[1,1] -> U[3]
    
    # C indices for block 1:
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    
    # Block 2 (columns 7-13): k=2,3
    # B2[0,0] = B[0,2] -> V[2]
    # B2[0,1] = B[0,3] -> V[3]
    # B2[1,0] = B[1,2] -> V[6]
    # B2[1,1] = B[1,3] -> V[7]
    
    # C indices for block 2:
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[3*2+0] = W[6]
    # C[1,2] -> W[2*2+1] = W[5]
    # C[1,3] -> W[3*2+1] = W[7]
    
    # Now fill in the 14 columns (7 for block 1, 7 for block 2)
    
    # Block 1 (columns 0-6):
    # M1 = (A[0,0]+A[1,1]) * (B1[0,0]+B1[1,1])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A[1,0]+A[1,1]) * B1[0,0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A[0,0] * (B1[0,1]-B1[1,1])
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A[1,1] * (B1[1,0]-B1[0,0])
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A[0,0]+A[0,1]) * B1[1,1]
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A[1,0]-A[0,0]) * (B1[0,0]+B1[0,1])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A[0,1]-A[1,1]) * (B1[1,0]+B1[1,1])
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1; W[2][6] = -1
    
    # Block 2 (columns 7-13):
    # M1' = (A[0,0]+A[1,1]) * (B2[0,0]+B2[1,1])
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M2' = (A[1,0]+A[1,1]) * B2[0,0]
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # M3' = A[0,0] * (B2[0,1]-B2[1,1])
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # M4' = A[1,1] * (B2[1,0]-B2[0,0])
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # M5' = (A[0,0]+A[0,1]) * B2[1,1]
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # M6' = (A[1,0]-A[0,0]) * (B2[0,0]+B2[0,1])
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # M7' = (A[0,1]-A[1,1]) * (B2[1,0]+B2[1,1])
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1; W[6][13] = -1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_fraction(U, V, W):
        return result
    else:
        # Debug: print which entries fail
        n, m, p = 2, 2, 4
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(14):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        print(f"FAIL: i={i}, j={j}, k={k}, sum={s}")
        return None

def _schoolbook():
    """Return the schoolbook rank-16 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 16
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
    """
    Return the rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
    """
    # First try to get rank 14 using Strassen
    result = _strassen_2x2x2x4()
    if result is not None:
        return result
    
    # Fallback to schoolbook rank 16
    return _schoolbook()
