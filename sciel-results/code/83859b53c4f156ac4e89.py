"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in valid_indices:
                    continue
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for a single 2x2x2 matrix multiplication.
    U: 4x7, V: 4x7, W: 4x7
    A indices: A00=0, A01=1, A10=2, A11=3
    B indices: B00=0, B01=1, B10=2, B11=3
    C indices: C00=0, C01=1, C10=2, C11=3
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A10 + A11) * B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3 = A00 * (B01 - B11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A11 * (B10 - B00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A00 + A01) * B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _strassen_block(n, m, p):
    """
    Decomposition using 8 independent Strassen 2x2x2 blocks.
    Rank = 8 * 7 = 56.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Precompute the Strassen 2x2x2 decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    col = 0
    for t in range(num_blocks):
        # V indices for block t:
        # B[0, 2t] -> V[2t], B[0, 2t+1] -> V[2t+1]
        # B[1, 2t] -> V[16+2t], B[1, 2t+1] -> V[16+2t+1]
        
        # W indices for block t:
        # C[0, 2t] -> W[4t], C[0, 2t+1] -> W[4t+2]
        # C[1, 2t] -> W[4t+1], C[1, 2t+1] -> W[4t+3]
        
        for r in range(7):
            # U: same for all blocks (A00=0, A01=1, A10=2, A11=3)
            for a in range(4):
                if U_s[a][r] != 0:
                    U[a][col] += U_s[a][r]
            
            # V: map local indices 0,1,2,3 to global indices
            # local 0 -> B00 -> 2t
            # local 1 -> B01 -> 2t+1
            # local 2 -> B10 -> 16+2t
            # local 3 -> B11 -> 16+2t+1
            v_map = [2*t, 2*t+1, 16+2*t, 16+2*t+1]
            for b_local in range(4):
                if V_s[b_local][r] != 0:
                    V[v_map[b_local]][col] += V_s[b_local][r]
            
            # W: map local indices 0,1,2,3 to global indices
            # local 0 -> C00 -> 4t
            # local 1 -> C01 -> 4t+2
            # local 2 -> C10 -> 4t+1
            # local 3 -> C11 -> 4t+3
            w_map = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for c_local in range(4):
                if W_s[c_local][r] != 0:
                    W[w_map[c_local]][col] += W_s[c_local][r]
            
            col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # The Strassen block decomposition achieves rank 56, which is optimal
    # for this tensor (direct sum of 8 copies of 2x2x2, each of rank 7).
    U, V, W = _strassen_block(n, m, p)
    
    # Verify correctness
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 64) if verification fails
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
