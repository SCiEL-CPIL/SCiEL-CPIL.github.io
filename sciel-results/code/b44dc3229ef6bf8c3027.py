"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if val != 1:
                    return False

    # Check spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
                
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if val != 0:
                        return False
    return True

def _construct_strassen_2x16():
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # B indices for this block
        b00 = 2*t
        b01 = 2*t + 1
        b10 = 16 + 2*t
        b11 = 16 + 2*t + 1
        
        # C indices for this block
        c00 = 4*t
        c01 = 4*t + 1
        c10 = 4*t + 2
        c11 = 4*t + 3
        
        # Strassen intermediates
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 (c00) = M1 + M4 - M5 + M7
        # C01 (c01) = M2 + M4
        # C10 (c10) = M3 + M5
        # C11 (c11) = M1 - M2 + M3 + M6
        
        # Define (row, coeff) for U, V, W for each M
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        
        # M1
        U[0][col] += 1; U[3][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1; U[3][col] += 1
        V[b00][col] += 1
        W[c01][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c10][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1; U[0][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1; U[3][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _construct_strassen_2x16()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback (should not happen)
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
