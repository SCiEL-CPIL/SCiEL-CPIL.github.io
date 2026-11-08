"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W matrices for rank-7 decomposition of 2x2x2 matrix multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U: [i*2+j, k]
    V: [i*2+j, k]
    W: [i*2+j, k]
    """
    # Strassen's 7 products
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Mapping: index = i*2 + j
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _construct_2x2x6():
    """
    Construct rank-21 decomposition for 2x2x6.
    Splits 6 into 3 blocks of 2.
    U shape: (4, 21), V shape: (12, 21), W shape: (12, 21)
    """
    U_str, V_str, W_str = _strassen_2x2x2()
    
    U = np.zeros((4, 21), dtype=float)
    V = np.zeros((12, 21), dtype=float)
    W = np.zeros((12, 21), dtype=float)
    
    for block in range(3):
        offset = block * 7
        U[:, offset:offset+7] = U_str
        # V indices: row j (0..5), col k
        # Block 0: j=0,1 -> rows 0,1
        # Block 1: j=2,3 -> rows 2,3
        # Block 2: j=4,5 -> rows 4,5
        V[block*4:(block+1)*4, offset:offset+7] = V_str
        # W indices: row i (0,1), col j (0..5)
        # W is (p*n, rank) = (6*2, rank) = (12, rank)
        # Index = j*2 + i
        # Block 0: j=0,1 -> rows 0,1,2,3
        # Block 1: j=2,3 -> rows 4,5,6,7
        # Block 2: j=4,5 -> rows 8,9,10,11
        W[block*4:(block+1)*4, offset:offset+7] = W_str
        
    return U, V, W

def _construct_2x1x6():
    """
    Construct rank-12 decomposition for 2x1x6.
    U shape: (2, 12), V shape: (6, 12), W shape: (12, 12)
    """
    U = np.zeros((2, 12), dtype=float)
    V = np.zeros((6, 12), dtype=float)
    W = np.zeros((12, 12), dtype=float)
    
    col = 0
    for i in range(2):
        for j in range(6):
            U[i, col] = 1
            V[j, col] = 1
            # W index: k*n + i ? No, W is (p*n, rank).
            # For 2x1x6: n=2, m=1, p=6.
            # W[k*n+i, col]. k is 0..5 (p), i is 0..1 (n).
            # Index = k*2 + i.
            W[j*2 + i, col] = 1
            col += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 2, 5, 6
    # Split m=5 into m1=4, m2=1
    m1, m2 = 4, 1
    
    # Part 1: 2 x 4 x 6
    # Split m1=4 into m1a=2, m1b=2
    m1a, m1b = 2, 2
    
    # U1a: 2x2x6 -> U (4, 21), V (12, 21), W (12, 21)
    U1a, V1a, W1a = _construct_2x2x6()
    # U1b: 2x2x6 -> U (4, 21), V (12, 21), W (12, 21)
    U1b, V1b, W1b = _construct_2x2x6()
    
    # Part 2: 2 x 1 x 6
    # U2: 2x1x6 -> U (2, 12), V (6, 12), W (12, 12)
    U2, V2, W2 = _construct_2x1x6()
    
    # Total rank
    rank = 21 + 21 + 12 # 54
    
    # Assemble U (n*m, rank) = (2*5, 54) = (10, 54)
    U = np.zeros((10, rank), dtype=float)
    # V (m*p, rank) = (5*6, 54) = (30, 54)
    V = np.zeros((30, rank), dtype=float)
    # W (p*n, rank) = (6*2, 54) = (12, 54)
    W = np.zeros((12, rank), dtype=float)
    
    # Indices for U: i*m + j
    # i is 0..1 (n=2)
    # j is 0..4 (m=5)
    # j=0,1 -> m1a (2x2)
    # j=2,3 -> m1b (2x2)
    # j=4   -> m2 (1x1? No, 2x1)
    
    # U1a corresponds to A[0:2, 0:2]
    # U1a shape (4, 21). Rows are i*2+j for i in 0..1, j in 0..1.
    # In full U, rows are i*5 + j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 5. i=1, j=1 -> 6.
    
    U[0:2, 0:21] = U1a[0:2, :]
    U[5:7, 0:21] = U1a[2:4, :]
    
    # U1b corresponds to A[0:2, 2:4]
    # U1b shape (4, 21).
    # In full U:
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 7. i=1, j=3 -> 8.
    
    U[2:4, 21:42] = U1b[0:2, :]
    U[7:9, 21:42] = U1b[2:4, :]
    
    # U2 corresponds to A[0:2, 4:5]
    # U2 shape (2, 12). Rows i=0,1.
    # In full U:
    # i=0, j=4 -> 4.
    # i=1, j=4 -> 9.
    
    U[4:5, 42:54] = U2[0:1, :]
    U[9:10, 42:54] = U2[1:2, :]
    
    # Indices for V: j*p + k
    # j is 0..4 (m=5)
    # k is 0..5 (p=6)
    
    # V1a: B[0:2, 0:6]
    # V1a shape (12, 21). Rows j*6+k for j in 0..1, k in 0..5.
    # In full V:
    # j=0, k=0..5 -> 0..5
    # j=1, k=0..5 -> 6..11
    # This matches V1a rows 0..11 directly.
    V[0:12, 0:21] = V1a
    
    # V1b: B[2:4, 0:6]
    # V1b shape (12, 21).
    # In full V:
    # j=2, k=0..5 -> 12..17
    # j=3, k=0..5 -> 18..23
    V[12:24, 21:42] = V1b
    
    # V2: B[4:5, 0:6]
    # V2 shape (6, 12).
    # In full V:
    # j=4, k=0..5 -> 24..29
    V[24:30, 42:54] = V2
    
    # Indices for W: k*n + i
    # k is 0..5 (p=6)
    # i is 0..1 (n=2)
    # This is the same for all parts.
    
    W[0:12, 0:21] = W1a
    W[0:12, 21:42] = W1b
    W[0:12, 42:54] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
