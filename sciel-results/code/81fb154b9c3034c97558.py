"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all (i,j,k)."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _construct_schoolbook():
    """Standard rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_44():
    """
    Rank-44 decomposition for 2x3x8 multiplication.
    Uses Strassen for the 2x2 part and schoolbook for the 2x1 part.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    U = [[0] * 44 for _ in range(n * m)]   # 6 x 44
    V = [[0] * 44 for _ in range(m * p)]   # 24 x 44
    W = [[0] * 44 for _ in range(p * n)]   # 16 x 44
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # Split into 4 blocks of 2x2
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # V indices
        v00 = k0       # B[0, k0]
        v01 = k1       # B[0, k1]
        v10 = 8 + k0   # B[1, k0]
        v11 = 8 + k1   # B[1, k1]
        
        # W indices
        w00 = 2 * k0       # C[0, k0]
        w01 = 2 * k1       # C[0, k1]
        w10 = 2 * k0 + 1   # C[1, k0]
        w11 = 2 * k1 + 1   # C[1, k1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] = 1
        U[4][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[3][col] = 1
        U[4][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0][col] = 1
        V[v10][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[4][col] = 1
        V[v01][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0][col] = 1
        U[1][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[3][col] = 1
        U[0][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w01][col] = -1
        W[w10][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col] = 1
        U[4][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w10][col] = -1
        col += 1
        
    # Part 2: A[0:2, 2] * B[2, 0:8]
    for k in range(8):
        # A[0, 2] * B[2, k] -> C[0, k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[2 * k][col] = 1
        col += 1
        
        # A[1, 2] * B[2, k] -> C[1, k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[2 * k + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Try rank-44 construction first
    try:
        result = _construct_44()
        U, V, W = result["U"], result["V"], result["W"]
        n, m, p = DIMENSIONS
        if _verify(U, V, W, n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook rank-48
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
