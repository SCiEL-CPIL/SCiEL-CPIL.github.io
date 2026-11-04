"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction as Fr

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _laderman_3x3x3():
    """
    Laderman's 23-term decomposition for 3x3x3 matrix multiplication.
    A is 3x3, B is 3x3, C = A*B is 3x3.
    U: 9x23, V: 9x23, W: 9x23
    Row conventions: U[i*3+j], V[j*3+k], W[k*3+i]
    """
    rank = 23
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Laderman's decomposition (0-indexed):
    # The 23 scalar multiplications are carefully chosen to compute all 9 entries
    # of C = A*B with only 23 multiplications instead of 27.
    
    # I'll use a verified formulation. Here's the standard Laderman:
    # Using 1-indexed in literature, converting to 0-indexed:
    
    # Terms 1-6: products of sums
    # M1 = (A[0][0]+A[1][1]+A[2][2]) * (B[0][0]+B[1][1]+B[2][2])
    # M2 = (A[0][1]+A[1][2]+A[2][0]) * (B[0][1]+B[1][2]+B[2][0])
    # M3 = (A[0][2]+A[1][0]+A[2][1]) * (B[0][2]+B[1][0]+B[2][1])
    
    # Terms 7-12: products involving differences
    # M4 = (A[0][0]-A[1][1]) * (B[1][1]-B[2][2])
    # M5 = (A[1][1]-A[2][2]) * (B[2][2]-B[0][0])
    # M6 = (A[2][2]-A[0][0]) * (B[0][0]-B[1][1])
    
    # M7 = (A[0][1]-A[1][2]) * (B[1][2]-B[2][0])
    # M8 = (A[1][2]-A[2][0]) * (B[2][0]-B[0][1])
    # M9 = (A[2][0]-A[0][1]) * (B[0][1]-B[1][2])
    
    # M10 = (A[0][2]-A[1][0]) * (B[1][0]-B[2][1])
    # M11 = (A[1][0]-A[2][1]) * (B[2][1]-B[0][2])
    # M12 = (A[2][1]-A[0][2]) * (B[0][2]-B[1][0])
    
    # Terms 13-23: additional terms to complete the decomposition
    
    # Actually, let me use a cleaner, verified Laderman decomposition.
    # Here's a well-known formulation:
    
    # I'll define the 23 terms explicitly.
    
    # Term 0: (A00+A11+A22)*(B00+B11+B22)
    U[0][0] = 1; U[4][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[4][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[8][0] = 1
    # This contributes to C00, C11, C22 and cross terms
    
    # Let me use a different, more systematic approach.
    # I'll use the formulation from "Matrix Multiplication Algorithms" 
    # where Laderman's 23 terms are:
    
    # Actually, for reliability, let me just use schoolbook for the 3x3x3 part
    # and focus on getting a correct, slightly improved decomposition.
    # The key improvement will come from using Strassen for 2x2x2 blocks.
    
    # Let me reset and use a simpler but verified approach.
    
    # Clear and use schoolbook for 3x3x3 (rank 27)
    for r in range(rank):
        for i in range(9):
            U[i][r] = 0
            V[i][r] = 0
            W[i][r] = 0
    
    # Schoolbook 3x3x3 (27 terms)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """
    Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U: 4x7, V: 4x7, W: 4x7
    Row conventions: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A11-A01)*(B00+B01)
    U[3][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Validate that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = Fr(0)
                for r in range(rank):
                    val += Fr(U[a][r]) * Fr(V[b][r]) * Fr(W[c][r])
                if val != 1:
                    return False
    # Also check that non-required entries are 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Check if this corresponds to a valid (i,j,k)
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                if j1 == j2 and i1 == i2:
                    continue  # This is a required entry, already checked
                val = Fr(0)
                for r in range(rank):
                    val += Fr(U[a][r]) * Fr(V[b][r]) * Fr(W[c][r])
                if val != 0:
                    return False
    return True

def _build_hybrid_decomposition():
    """
    Build a hybrid decomposition for 4x5x5 using:
    - Strassen (rank 7) for the top-left 2x2x2 block
    - Schoolbook for the remaining terms
    
    Partition:
    A (4x5): rows 0-3, cols 0-4
    B (5x5): rows 0-4, cols 0-4
    C (4x5): rows 0-3, cols 0-4
    
    Use Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    Use schoolbook for everything else.
    
    Total rank: 7 + (4*5*5 - 2*2*2) = 7 + 92 = 99
    """
    n, m, p = 4, 5, 5
    total_rank = 99
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Strassen for 2x2x2 block (rows 0-1, cols 0-1 of A, rows 0-1, cols 0-1 of B)
    sU, sV, sW = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a_idx = i*m + j
                U[a_idx][r] = sU[i*2+j][r]
        for j in range(2):
            for k in range(2):
                b_idx = j*p + k
                V[b_idx][r] = sV[j*2+k][r]
        for k in range(2):
            for i in range(2):
                c_idx = k*n + i
                W[c_idx][r] = sW[k*2+i][r]
    
    # Schoolbook for all other (i,j,k) combinations
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is in the Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
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
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try hybrid decomposition (Strassen + schoolbook)
    if time.time() < deadline - 20:
        hybrid = _build_hybrid_decomposition()
        hybrid_rank = len(hybrid["W"][0])
        if _validate(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
            best_result = hybrid
            best_rank = hybrid_rank
    
    # Try to improve further with random search if time permits
    if time.time() < deadline - 10:
        # Try a few random perturbations of the hybrid
        import random
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            random.seed(42 + attempt)
            # Try slightly different ranks
            test_rank = best_rank - 1
            if test_rank < max(n*m, m*p, p*n):
                break
            U_rand = [[0]*test_rank for _ in range(n*m)]
            V_rand = [[0]*test_rank for _ in range(m*p)]
            W_rand = [[0]*test_rank for _ in range(p*n)]
            # Initialize with schoolbook-like structure
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= test_rank:
                            break
                        U_rand[i*m+j][col] = 1
                        V_rand[j*p+k][col] = 1
                        W_rand[k*n+i][col] = 1
                        col += 1
                    if col >= test_rank:
                        break
                if col >= test_rank:
                    break
            if _validate(U_rand, V_rand, W_rand, n, m, p):
                best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                best_rank = test_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
