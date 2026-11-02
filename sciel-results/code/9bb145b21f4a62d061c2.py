"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
START_TIME = time.time()
TIME_LIMIT = 120

def construct():
    if time.time() - START_TIME > TIME_LIMIT - 5:
        # Fallback to schoolbook if timeout (unlikely for static construction)
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}

    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 7 blocks of 2x2
    for b in range(7):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # V indices for B block (j*p + k)
        v_b11 = 0 * p + k1
        v_b12 = 0 * p + k2
        v_b21 = 1 * p + k1
        v_b22 = 1 * p + k2
        
        # W indices for C block (k*n + i)
        w_c11 = k1 * n + 0
        w_c12 = k2 * n + 0
        w_c21 = k1 * n + 1
        w_c22 = k2 * n + 1
        
        # Strassen Terms
        # 1: (A11+A22)(B11+B22)
        U[0][col] += 1; U[3][col] += 1
        V[v_b11][col] += 1; V[v_b22][col] += 1
        W[w_c11][col] += 1; W[w_c22][col] += 1
        col += 1
        
        # 2: (A21+A22)B11
        U[2][col] += 1; U[3][col] += 1
        V[v_b11][col] += 1
        W[w_c21][col] += 1; W[w_c22][col] += 1
        col += 1
        
        # 3: A11(B12-B22)
        U[0][col] += 1
        V[v_b12][col] += 1; V[v_b22][col] -= 1
        W[w_c12][col] += 1; W[w_c22][col] += 1
        col += 1
        
        # 4: A22(B21-B11)
        U[3][col] += 1
        V[v_b21][col] += 1; V[v_b11][col] -= 1
        W[w_c11][col] += 1; W[w_c21][col] += 1
        col += 1
        
        # 5: (A11+A12)B22
        U[0][col] += 1; U[1][col] += 1
        V[v_b22][col] += 1
        W[w_c11][col] -= 1; W[w_c12][col] += 1
        col += 1
        
        # 6: (A21-A11)(B11+B12)
        U[2][col] += 1; U[0][col] -= 1
        V[v_b11][col] += 1; V[v_b12][col] += 1
        W[w_c22][col] += 1
        col += 1
        
        # 7: (A12-A22)(B21+B22)
        U[1][col] += 1; U[3][col] -= 1
        V[v_b21][col] += 1; V[v_b22][col] += 1
        W[w_c11][col] += 1
        col += 1

    # 1 block of 2x1 (k=14)
    k1 = 14
    v_b11 = 0 * p + k1
    v_b21 = 1 * p + k1
    
    w_c11 = k1 * n + 0
    w_c21 = k1 * n + 1
    
    # Term 1: A00 * B0,14 -> C0,14
    U[0][col] = 1
    V[v_b11][col] = 1
    W[w_c11][col] = 1
    col += 1
    
    # Term 2: A01 * B1,14 -> C0,14
    U[1][col] = 1
    V[v_b21][col] = 1
    W[w_c11][col] = 1
    col += 1
    
    # Term 3: A10 * B0,14 -> C1,14
    U[2][col] = 1
    V[v_b11][col] = 1
    W[w_c21][col] = 1
    col += 1
    
    # Term 4: A11 * B1,14 -> C1,14
    U[3][col] = 1
    V[v_b21][col] = 1
    W[w_c21][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
