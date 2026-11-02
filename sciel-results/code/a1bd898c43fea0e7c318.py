"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x10 multiplication using a hybrid approach:
    - Use Strassen (rank 7) for pairs of columns where it helps
    - Use schoolbook (rank 4) for individual columns
    
    Key insight: For a 2x2 * 2x1 multiplication, the tensor rank is 4 (schoolbook).
    For a 2x2 * 2x2 multiplication, the tensor rank is 7 (Strassen).
    
    If we split 2x10 into 5 blocks of 2x2, we get rank 5*7 = 35.
    If we split 2x10 into 10 columns of 2x1, we get rank 10*4 = 40.
    
    But we can mix: use Strassen for some pairs and schoolbook for others.
    Actually, let's think about this differently.
    
    The B matrix is 2x10. We can view it as:
    - 5 blocks of 2x2 -> rank 5*7 = 35
    - 10 blocks of 2x1 -> rank 10*4 = 40
    
    So 35 is better than 40. But can we do better?
    
    Let's consider: what if we use Strassen for 2 pairs (4 columns) and schoolbook for 6 columns?
    Rank = 2*7 + 6*4 = 14 + 24 = 38. Worse.
    
    What about 1 pair Strassen + 8 schoolbook? 7 + 32 = 39. Worse.
    
    So the pure Strassen approach (35) seems optimal among these simple combinations.
    
    However, there might be a more clever decomposition. Let's try to verify if 35 is
    indeed the minimum by checking the flattening rank lower bound.
    
    For now, return the rank-35 Strassen decomposition, which is verified correct.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return _construct_rank35()
    
    n, m, p = DIMENSIONS
    
    # Build the rank-35 Strassen decomposition
    result = _construct_rank35()
    
    # Verify correctness with a random test
    if time.time() < _DEADLINE - 5:
        if not _verify_decomposition(result, n, m, p):
            # Fallback: return a trivially correct (but higher rank) schoolbook decomposition
            return _construct_schoolbook(n, m, p)
    
    return result


def _construct_rank35():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    The tensor for C = A B where A is 2x2 and B is 2x10 can be viewed as
    5 independent 2x2 x 2x2 multiplication problems, one for each pair of columns in B.
    
    The rank of the 2x2 x 2x2 matrix multiplication tensor is 7 (Strassen's algorithm).
    Since the 5 blocks are independent, the total rank is the sum of the ranks of the blocks.
    Total Rank = 5 * 7 = 35.
    
    The interface requires:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Indices:
    U index: i*m + j  (i in 0..1, j in 0..1) -> 0,1,2,3
    V index: j*p + k  (j in 0..1, k in 0..9) -> 0..19
    W index: k*n + i  (k in 0..9, i in 0..1) -> 0..19 (Official trace convention)
    
    Strassen's 7 products for multiplying 2x2 matrices M1 and M2:
    M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    Result M = M1 * M2:
    M[0,0] = P1 + P4 - P5 + P7
    M[0,1] = P3 + P5
    M[1,0] = P2 + P4
    M[1,1] = P1 - P2 + P3 + P6
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for the 7 products
    # A inputs: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B inputs: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # U coefficients (for A) and V coefficients (for B) for each of the 7 Strassen products
    strassen_A = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: (e+h)
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [-1, 0, 1, 0], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W coefficients for each Strassen product
    # Local C indices: 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    # P1 contributes to C[0,0] (+1) and C[1,1] (+1)
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    # P2 contributes to C[1,0] (+1) and C[1,1] (-1)
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    # P3 contributes to C[0,1] (+1) and C[1,1] (+1)
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    # P4 contributes to C[0,0] (+1) and C[1,0] (+1)
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    # P5 contributes to C[0,0] (-1) and C[0,1] (+1)
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    # P6 contributes to C[1,1] (+1)
    w_contrib[5][3] = 1
    
    # P7 contributes to C[0,0] (+1)
    w_contrib[6][0] = 1
    
    # Fill U, V, W
    # For each block b in 0..4:
    #   For each Strassen product r in 0..6:
    #     Global rank index col = b * 7 + r
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # Fill V
            k0 = 2 * b
            k1 = 2 * b + 1
            
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            # Fill W
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n, m, p, num_tests=50, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    import random
    random.seed(seed)
    
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    
    rank = len(U[0])
    
    for _ in range(num_tests):
        # Generate random integer matrices
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B
        C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
        
        # Compute using decomposition
        C_decomp = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for r in range(rank):
                    w_val = W[k*n+i][r]
                    if w_val == 0:
                        continue
                    u_dot = 0
                    for j1 in range(m):
                        u_dot += U[i*m+j1][r] * A[i][j1]
                    v_dot = 0
                    for j2 in range(m):
                        v_dot += V[j2*p+k][r] * B[j2][k]
                    val += w_val * u_dot * v_dot
                C_decomp[i][k] = val
        
        # Check if C_decomp equals C
        for i in range(n):
            for k in range(p):
                if C_decomp[i][k] != C[i][k]:
                    return False
    
    return True


def _construct_schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i*m + j)*p + k
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
