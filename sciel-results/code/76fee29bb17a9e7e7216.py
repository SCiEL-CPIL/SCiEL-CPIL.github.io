"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
DEADLINE = 150.0

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A: 2x2 at (a_r, a_c)
        # B: 2x2 at (b_r, b_c)
        # C: 2x2 at (c_r, c_c)
        
        u00 = a_r * m + a_c
        u01 = a_r * m + a_c + 1
        u10 = (a_r + 1) * m + a_c
        u11 = (a_r + 1) * m + a_c + 1
        
        v00 = b_r * p + b_c
        v01 = b_r * p + b_c + 1
        v10 = (b_r + 1) * p + b_c
        v11 = (b_r + 1) * p + b_c + 1
        
        w00 = c_c * n + c_r
        w01 = (c_c + 1) * n + c_r
        w10 = c_c * n + c_r + 1
        w11 = (c_c + 1) * n + c_r + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A: 2x2 at (a_r, a_c)
        # B: 2x1 at (b_r, b_c)
        # C: 2x1 at (c_r, c_c)
        
        u00 = a_r * m + a_c
        u01 = a_r * m + a_c + 1
        u10 = (a_r + 1) * m + a_c
        u11 = (a_r + 1) * m + a_c + 1
        
        v0 = b_r * p + b_c
        v1 = (b_r + 1) * p + b_c
        
        w0 = c_c * n + c_r
        w1 = c_c * n + c_r + 1
        
        # T1: A00 B0
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        # T2: A01 B1
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1
        col += 1
        # T3: A10 B0
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        # T4: A11 B1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1

    # C1 (2x2) from B cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C2 (2x3) from B cols 2,3,4
    # C21 (2x1) from B col 2
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    
    # C22 (2x2) from B cols 3,4
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
