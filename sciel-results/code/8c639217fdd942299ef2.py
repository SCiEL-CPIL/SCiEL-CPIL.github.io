"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A_L is 2x2 from A (indices 0,1,3,4 in U)
    # B_L is 2x2 from B (indices j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1 in V)
    # C is 2x2 from W (indices k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1)
    def add_strassen(j0, j1, k0, k1):
        nonlocal col
        # A indices
        a00 = 0  # i=0, j=0
        a01 = 1  # i=0, j=1
        a10 = 3  # i=1, j=0
        a11 = 4  # i=1, j=1
        
        # B indices
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        # C indices
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        
        # m1 = (a00+a11)(b00+b11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        
        # m3 = a00(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] -= 1
        col += 1

    # 1. Strassen for A_L (j=0,1) and B_L1 (k=0,1)
    # A_L uses j=0,1. B_L1 uses k=0,1.
    # j0=0, j1=1, k0=0, k1=1
    add_strassen(0, 1, 0, 1)
    
    # 2. Strassen for A_L (j=0,1) and B_L2 (k=2,3)
    add_strassen(0, 1, 2, 3)
    
    # 3. Strassen for A_L (j=0,1) and B_L3 (k=4,5)
    add_strassen(0, 1, 4, 5)
    
    # 4. Outer product a_R (j=2) and b_R (all k)
    # a_R is row i=0,1 at j=2. Indices 5, 6 in U.
    # b_R is row j=2, k=0..5. Indices 12..17 in V.
    # C is all k, i=0,1.
    for k in range(p):
        for i in range(n):
            U[2 + i][col] = 1  # j=2, i=0,1
            V[2 * p + k][col] = 1  # j=2, k
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
