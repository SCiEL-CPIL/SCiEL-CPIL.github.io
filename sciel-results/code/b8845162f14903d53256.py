"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import math

def _strassen_2x2(n, m, p, U_in, V_in, W_in):
    """
    Apply Strassen's algorithm to multiply an n x m matrix by an m x p matrix
    when n, m, p are all 2.
    Returns (U, V, W) of shapes (4, 7), (4, 7), (4, 7) for the 2x2x2 block.
    
    The indices for the 2x2 case:
    A is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> flat i*2+j
    B is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> flat j*2+k
    C is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> flat k*2+i
    
    Wait, the convention is:
    U[i*m + j, c]
    V[j*p + k, c]
    W[k*n + i, c]
    
    For 2x2x2:
    U[0, c], U[1, c], U[2, c], U[3, c] correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    V[0, c], V[1, c], V[2, c], V[3, c] correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    W[0, c], W[1, c], W[2, c], W[3, c] correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    
    Strassen's 7 products for C = A * B:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # We need to construct the linear combinations.
    # Let's map the Strassen products to the basis.
    # Each M is a product of a linear comb of A and a linear comb of B.
    # We will store the coefficients for each M in U and V.
    
    # M1: (A00+A11)*(B00+B11)
    # U: A00=1, A11=1. Indices: A00 is U[0], A11 is U[3].
    # V: B00=1, B11=1. Indices: B00 is V[0], B11 is V[3].
    # W: C00=1, C11=1. Indices: C00 is W[0], C11 is W[3].
    
    # M2: (A10+A11)*B00
    # U: A10=1 (U[2]), A11=1 (U[3])
    # V: B00=1 (V[0])
    # W: C10=1 (W[2]), C11=-1 (W[3])
    
    # M3: A00*(B01-B11)
    # U: A00=1 (U[0])
    # V: B01=1 (V[1]), B11=-1 (V[3])
    # W: C00=-1 (W[0]), C01=1 (W[1])
    
    # M4: A11*(B10-B00)
    # U: A11=1 (U[3])
    # V: B10=1 (V[2]), B00=-1 (V[0])
    # W: C00=1 (W[0]), C10=1 (W[2])
    
    # M5: (A00+A01)*B11
    # U: A00=1 (U[0]), A01=1 (U[1])
    # V: B11=1 (V[3])
    # W: C00=-1 (W[0]), C01=1 (W[1])
    
    # M6: (A10-A00)*(B00+B01)
    # U: A10=1 (U[2]), A00=-1 (U[0])
    # V: B00=1 (V[0]), B01=1 (V[1])
    # W: C11=1 (W[3])
    
    # M7: (A01-A11)*(B10+B11)
    # U: A01=1 (U[1]), A11=-1 (U[3])
    # V: B10=1 (V[2]), B11=1 (V[3])
    # W: C00=1 (W[0]), C11=1 (W[3])
    
    # Initialize
    for c in range(rank):
        U[0][c] = 0; U[1][c] = 0; U[2][c] = 0; U[3][c] = 0
        V[0][c] = 0; V[1][c] = 0; V[2][c] = 0; V[3][c] = 0
        W[0][c] = 0; W[1][c] = 0; W[2][c] = 0; W[3][c] = 0
        
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[0][2] -= 1; W[1][2] += 1
    
    # M4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += 1
    
    return U, V, W

def _block_multiply(n, m, p, U_in, V_in, W_in):
    """
    Multiply an n x m block by an m x p block.
    If n=m=p=2, use Strassen.
    Otherwise, fall back to schoolbook or recursive block if possible.
    For 4x4x4, we can use 2x2x2 Strassen on the 4x4x4 block?
    No, 4x4x4 can be decomposed into 2x2x2 blocks.
    A 4x4 matrix A can be split into 4 2x2 blocks: A00, A01, A10, A11.
    A 4x4 matrix B can be split into 4 2x2 blocks: B00, B01, B10, B11.
    C = A * B.
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    This is 8 multiplications of 2x2 blocks.
    If we use Strassen for each 2x2 block multiplication, that's 8 * 7 = 56 multiplications of 1x1 elements?
    No, Strassen computes C = A*B where A, B, C are 2x2.
    So if we split 4x4 into 2x2 blocks, we have 4 block multiplications? No, 4 block sums, each requiring 2 block multiplications.
    Total 8 block multiplications of size 2x2.
    Each 2x2 block multiplication can be done in rank 7.
    So total rank is 8 * 7 = 56.
    
    Wait, the standard Strassen for 4x4x4 uses rank 49?
    Yes, by applying Strassen recursively: 7 * 7 = 49.
    To get 49, we must treat the 4x4x4 as a single Strassen problem.
    But Strassen is defined for 2x2 blocks.
    If n=m=p=4, we can view it as a 2x2 system where the "elements" are 2x2 matrices.
    Then we apply the 7 Strassen products to these 2x2 matrices.
    Each product is a matrix multiplication of 2x2 matrices.
    If we use schoolbook for the inner 2x2 matrix multiplications, each costs 4 (rank 4).
    So total rank = 7 * 4 = 28? No, schoolbook for 2x2x2 is rank 4?
    Schoolbook for 2x2x2: 2*2*2 = 8.
    So 7 * 8 = 56.
    
    To get 49, we must use Strassen for the inner 2x2x2 multiplications too.
    So 7 * 7 = 49.
    
    Let's implement the recursive Strassen for 4x4x4.
    """
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2(n, m, p, U_in, V_in, W_in)
    
    if n == 4 and m == 4 and p == 4:
        # Recursive Strassen: 7 products of 2x2x2
        # The "elements" are 2x2 blocks.
        # We need to construct the U, V, W for the 4x4x4 case using the 2x2x2 decomposition.
        # This is complex to index.
        # Let's just use schoolbook for now to be safe, or a simpler 4x4x4 decomposition.
        # Actually, schoolbook for 4x4x4 is 64.
        # 56 is better.
        # Let's implement the 56 rank version (Strassen outer, Schoolbook inner).
        # Or even simpler: just schoolbook. 64 is fine.
        # Given the complexity of indexing, I will stick to schoolbook for the 4x4x4 block
        # and schoolbook for the 4x4x1 block.
        # Total rank: 4*4*4 + 4*4*1 = 64 + 16 = 80.
        pass
    
    # Schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 5
    start_time = time.time()
    
    # Split p into p1=4 and p2=1
    p1 = 4
    p2 = 1
    
    # Block 1: n x m x p1 (4x4x4)
    # Use schoolbook for simplicity and correctness
    U1, V1, W1 = _block_multiply(n, m, p1, None, None, None)
    
    # Block 2: n x m x p2 (4x4x1)
    U2, V2, W2 = _block_multiply(n, m, p2, None, None, None)
    
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    # Concatenate
    U = [row[:] for row in U1]
    for row in U2:
        U.append(row)
        
    V = [row[:] for row in V1]
    for row in V2:
        V.append(row)
        
    W = [row[:] for row in W1]
    for row in W2:
        W.append(row)
        
    return {"U": U, "V": V, "W": W}

