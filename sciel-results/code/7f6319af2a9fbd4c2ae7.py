import time
import numpy as np

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()

def construct():
    """
    Explicit rank-28 construction for 2x3 by 3x5 matrix multiplication.
    
    Decomposition:
    - Split the 3x3 inner dimension into a 2x2 block and a 1x1 block
    - For the 2x2 block (2x2 by 2x5): use Strassen (rank 7) for 2x2x2 + schoolbook (rank 4) for 2x2x1 = rank 11
    - For the 1x1 block (2x1 by 1x5): schoolbook rank 5
    - Total: 11 + 5 = 16? Wait, let me recalculate.
    
    Actually, let me use the known optimal: rank 28.
    2x3x5 tensor. Known bounds: R(2,3,5) <= 28.
    
    Let me use a simpler explicit construction:
    Split 3 into 2+1 for the middle dimension.
    - (2x2) by (2x5): Strassen for 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = 11
    - (2x1) by (1x5): schoolbook 2x1x5 (rank 10)
    Total: 11 + 10 = 21? That's not 28.
    
    Wait, schoolbook 2x1x5 = 2*1*5 = 10. Yes.
    But we can do better. Let me just use the standard schoolbook rank 30 and see.
    
    Actually, let me use a known rank-28 construction.
    R(2,3,5): the known upper bound is 28.
    
    Let me construct it carefully:
    A is 2x3, B is 3x5, C = A*B is 2x5.
    
    I'll use the following approach:
    - For the first 2 columns of A and first 2 rows of B: use Strassen-like rank 7 for 2x2x2
    - Remaining terms done with schoolbook
    
    Actually, let me just use the schoolbook (rank 30) to be safe, since that's guaranteed correct.
    The seed already has rank 30, so I need to improve.
    
    Let me try rank 21 construction:
    Split m=3 into m1=2, m2=1.
    Block (i,j) where i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
    
    For j=0,1 (first 2 rows of B, first 2 columns of A):
      2x2x5 subproblem.
      Split p=5 into p1=2, p2=2, p3=1.
      2x2x2: Strassen rank 7
      2x2x2: Strassen rank 7
      2x2x1: schoolbook rank 4
      Subtotal: 18
    
    For j=2 (third row of B, third column of A):
      2x1x5 subproblem: schoolbook rank 10
    
    Total: 18 + 10 = 28.
    
    Let me verify this is correct by constructing it explicitly.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 6 x 28
    V = [[0]*rank for _ in range(m*p)]  # 15 x 28
    W = [[0]*rank for _ in range(p*n)]  # 10 x 28
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 Strassen (rank 7)
    # Strassen for C = A*B where A is 2x2, B is 2x2
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A10)B11
    # m6 = (A11+A01)(B00+B10)
    # m7 = (A01-A11)(B11+B10)
    # C00 = m1+m4-m5+m7
    # C01 = m3+m5
    # C10 = m2+m4
    # C11 = m1-m2+m3+m6
    
    # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B00 = B[0,0], B01 = B[0,1], B10 = B[1,0], B11 = B[1,1]
    # C00 = C[0,0], C01 = C[0,1], C10 = C[1,0], C11 = C[1,1]
    
    # U indices: i*m + j = i*3 + j
    # V indices: j*p + k = j*5 + k
    # W indices: k*n + i = k*2 + i
    
    # Strassen m1: (A00+A11)(B00+B11)
    # U: A00 -> U[0*3+0]=U[0], A11 -> U[1*3+1]=U[4]
    # V: B00 -> V[0*5+0]=V[0], B11 -> V[1*5+1]=V[6]
    # W: C00 -> W[0*2+0]=W[0], C11 -> W[1*2+1]=W[3]
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # Strassen m2: (A10+A11)B00
    # U: A10 -> U[1*3+0]=U[3], A11 -> U[4]
    # V: B00 -> V[0]
    # W: C10 -> W[0*2+1]=W[1]
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # Strassen m3: A00(B01-B11)
    # U: A00 -> U[0]
    # V: B01 -> V[0*5+1]=V[1], B11 -> V[6]
    # W: C01 -> W[1*2+0]=W[2]
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1
    col += 1
    
    # Strassen m4: A11(B10-B00)
    # U: A11 -> U[4]
    # V: B10 -> V[1*5+0]=V[5], B00 -> V[0]
    # W: C11 -> W[3]
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[3][col] = 1
    col += 1
    
    # Strassen m5: (A00+A10)B11
    # U: A00 -> U[0], A10 -> U[3]
    # V: B11 -> V[6]
    # W: C11 -> W[3]
    U[0][col] = 1; U[3][col] = 1
    V[6][col] = 1
    W[3][col] = -1
    col += 1
    
    # Strassen m6: (A11+A01)(B00+B10)
    # U: A11 -> U[4], A01 -> U[0*3+1]=U[1]
    # V: B00 -> V[0], B10 -> V[5]
    # W: C11 -> W[3]
    U[4][col] = 1; U[1][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[3][col] = 1
    col += 1
    
    # Strassen m7: (A01-A11)(B11+B10)
    # U: A01 -> U[1], A11 -> U[4]
    # V: B11 -> V[6], B10 -> V[5]
    # W: C00 -> W[0]
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 Strassen (rank 7)
    # B00' = B[0,2], B01' = B[0,3], B10' = B[1,2], B11' = B[1,3]
    # C00' = C[0,2], C01' = C[0,3], C10' = C[1,2], C11' = C[1,3]
    
    # m1: (A00+A11)(B00'+B11')
    U[0][col] = 1; U[4][col] = 1
    V[0*5+2][col] = 1; V[1*5+3][col] = 1  # V[2], V[8]
    W[2*2+0][col] = 1; W[3*2+1][col] = 1  # W[4], W[7]
    col += 1
    
    # m2: (A10+A11)B00'
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[2*2+1][col] = 1  # W[5]
    col += 1
    
    # m3: A00(B01'-B11')
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1  # V[0*5+3]=V[3], V[1*5+3]=V[8]
    W[3*2+0][col] = 1  # W[6]
    col += 1
    
    # m4: A11(B10'-B00')
    U[4][col] = 1
    V[7][col] = 1; V[2][col] = -1  # V[1*5+2]=V[7], V[2]
    W[7][col] = 1
    col += 1
    
    # m5: (A00+A10)B11'
    U[0][col] = 1; U[3][col] = 1
    V[8][col] = 1
    W[7][col] = -1
    col += 1
    
    # m6: (A11+A01)(B00'+B10')
    U[4][col] = 1; U[1][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # m7: (A01-A11)(B11'+B10')
    U[1][col] = 1; U[4][col] = -1
    V[8][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4} -> 2x2x1 schoolbook (rank 4)
    # B[0,4], B[1,4]; C[0,4], C[1,4]
    
    # C[0,4] += A[0,0]*B[0,4]
    U[0][col] = 1
    V[4][col] = 1  # V[0*5+4]
    W[9][col] = 1  # W[4*2+0]
    col += 1
    
    # C[0,4] += A[0,1]*B[1,4]
    U[1][col] = 1
    V[9][col] = 1  # V[1*5+4]
    W[9][col] = 1
    col += 1
    
    # C[1,4] += A[1,0]*B[0,4]
    U[3][col] = 1
    V[4][col] = 1
    W[10-1][col] = 1  # W[4*2+1]=W[9]... wait W has 10 entries, indices 0-9
    W[9][col] = 1
    col += 1
    
    # C[1,4] += A[1,1]*B[1,4]
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # Block 4: i in {0,1}, j in {2}, k in {0,1,2,3,4} -> 2x1x5 schoolbook (rank 10)
    # A[0,2], A[1,2]; B[2,0..4]; C[0,0..4], C[1,0..4]
    
    for k in range(5):
        # C[0,k] += A[0,2]*B[2,k]
        U[0*3+2][col] = 1  # U[2]
        V[2*5+k][col] = 1  # V[10+k]
        W[k*2+0][col] = 1  # W[2k]
        col += 1
        
        # C[1,k] += A[1,2]*B[2,k]
        U[1*3+2][col] = 1  # U[5]
        V[2*5+k][col] = 1  # V[10+k]
        W[k*2+1][col] = 1  # W[2k+1]
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
