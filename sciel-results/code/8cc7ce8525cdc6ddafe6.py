"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = 150.0  # seconds

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    Rank 49 is the best known explicit decomposition for 4x4 matrix multiplication.
    """
    # Deadline guard: return immediately if we're close to the limit
    if time.time() - _START_TIME > _DEADLINE - 2.0:
        # Fallback: return schoolbook rank-64 if somehow we hit deadline
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                c = i*m + j
                U[i*m + j][c] = 1
                for k in range(p):
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition matrices
    # U2[i, j, c], V2[j, k, c], W2[k, i, c]
    # c ranges from 0 to 6
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M0: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M1: (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # M2: A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # M3: A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M4: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M5: (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M6: (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4 matrices via Kronecker product
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Precompute non-zero entries for efficiency
    U2_nz = {}
    for idx in range(4):
        for c in range(7):
            if U2[idx][c] != 0:
                if idx not in U2_nz:
                    U2_nz[idx] = []
                U2_nz[idx].append((c, U2[idx][c]))
    
    V2_nz = {}
    for idx in range(4):
        for c in range(7):
            if V2[idx][c] != 0:
                if idx not in V2_nz:
                    V2_nz[idx] = []
                V2_nz[idx].append((c, V2[idx][c]))
    
    W2_nz = {}
    for idx in range(4):
        for c in range(7):
            if W2[idx][c] != 0:
                if idx not in W2_nz:
                    W2_nz[idx] = []
                W2_nz[idx].append((c, W2[idx][c]))
    
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        idx1 = i1 * 2 + 0  # j1=0 for U
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            idx = i1 * 2 + j1
            idx2 = i2 * 2 + j2
            for c1, u1 in U2_nz.get(idx, []):
                for c2, u2 in U2_nz.get(idx2, []):
                    U[i * m + j][c1 * 7 + c2] = u1 * u2
                        
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            idx = j1 * 2 + k1
            idx2 = j2 * 2 + k2
            for c1, v1 in V2_nz.get(idx, []):
                for c2, v2 in V2_nz.get(idx2, []):
                    V[j * p + k][c1 * 7 + c2] = v1 * v2
                        
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            idx = k1 * 2 + i1
            idx2 = k2 * 2 + i2
            for c1, w1 in W2_nz.get(idx, []):
                for c2, w2 in W2_nz.get(idx2, []):
                    W[k * n + i][c1 * 7 + c2] = w1 * w2
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
