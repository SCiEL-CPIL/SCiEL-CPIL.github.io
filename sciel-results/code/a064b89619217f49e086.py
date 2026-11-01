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
    
    # Check valid terms: sum(U[a,r] * V[b,r] * W[c,r]) should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms: sum(U[a,r] * V[b,r] * W[c,r]) should be 0
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

def _strassen_block(n, m, p):
    """
    Decomposition into 8 independent 2x2x2 multiplications using Strassen's algorithm.
    Rank = 8 * 7 = 56.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        v_B00 = 2*t
        v_B01 = 2*t+1
        v_B10 = 16+2*t
        v_B11 = 16+2*t+1
        w_C00 = 4*t
        w_C01 = 4*t+2
        w_C10 = 4*t+1
        w_C11 = 4*t+3
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] += 1
        U[3][col] += 1
        V[v_B00][col] += 1
        V[v_B11][col] += 1
        W[w_C00][col] += 1
        W[w_C11][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[2][col] += 1
        U[3][col] += 1
        V[v_B00][col] += 1
        W[w_C10][col] += 1
        W[w_C11][col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0][col] += 1
        V[v_B01][col] += 1
        V[v_B11][col] -= 1
        W[w_C01][col] += 1
        W[w_C11][col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[3][col] += 1
        V[v_B10][col] += 1
        V[v_B00][col] -= 1
        W[w_C00][col] += 1
        W[w_C10][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0][col] += 1
        U[1][col] += 1
        V[v_B11][col] += 1
        W[w_C00][col] -= 1
        W[w_C01][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2][col] += 1
        U[0][col] -= 1
        V[v_B00][col] += 1
        V[v_B01][col] += 1
        W[w_C11][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col] += 1
        U[3][col] -= 1
        V[v_B10][col] += 1
        V[v_B11][col] += 1
        W[w_C00][col] += 1
        col += 1
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    U, V, W = _strassen_block(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
