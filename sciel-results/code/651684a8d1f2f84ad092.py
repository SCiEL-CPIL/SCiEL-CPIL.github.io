"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (6, 17)
    # V: (m*p, rank) -> (9, 17)
    # W: (p*n, rank) -> (6, 17)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def idx_U(i, j): return i * m + j
    def idx_V(j, k): return j * p + k
    def idx_W(k, i): return k * n + i
    
    r = 0
    
    # --- Strassen for 2x2x2 block (A[0:2, 0:2] * B[0:2, 0:2]) ---
    # P1 = (A00 + A11)(B00 + B11)
    U[idx_U(0,0)][r] += 1.0
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(0,0)][r] += 1.0
    V[idx_V(1,1)][r] += 1.0
    W[idx_W(0,0)][r] += 1.0  # C00
    W[idx_W(1,1)][r] += 1.0  # C11
    r += 1
    
    # P2 = (A10 + A11)B00
    U[idx_U(1,0)][r] += 1.0
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(0,0)][r] += 1.0
    W[idx_W(0,1)][r] += 1.0  # C10
    W[idx_W(1,1)][r] += -1.0 # C11
    r += 1
    
    # P3 = A00(B01 - B11)
    U[idx_U(0,0)][r] += 1.0
    V[idx_V(0,1)][r] += 1.0
    V[idx_V(1,1)][r] += -1.0
    W[idx_W(1,0)][r] += 1.0  # C01
    W[idx_W(1,1)][r] += 1.0  # C11
    r += 1
    
    # P4 = A11(B10 - B00)
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(1,0)][r] += 1.0
    V[idx_V(0,0)][r] += -1.0
    W[idx_W(0,0)][r] += 1.0  # C00
    W[idx_W(0,1)][r] += 1.0  # C10
    r += 1
    
    # P5 = (A00 + A01)B11
    U[idx_U(0,0)][r] += 1.0
    U[idx_U(0,1)][r] += 1.0
    V[idx_V(1,1)][r] += 1.0
    W[idx_W(0,0)][r] += -1.0 # C00
    W[idx_W(1,0)][r] += 1.0  # C01
    r += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[idx_U(1,0)][r] += 1.0
    U[idx_U(0,0)][r] += -1.0
    V[idx_V(0,0)][r] += 1.0
    V[idx_V(0,1)][r] += 1.0
    W[idx_W(1,1)][r] += 1.0  # C11
    r += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[idx_U(0,1)][r] += 1.0
    U[idx_U(1,1)][r] += -1.0
    V[idx_V(1,0)][r] += 1.0
    V[idx_V(1,1)][r] += 1.0
    W[idx_W(0,0)][r] += 1.0  # C00
    r += 1
    
    # --- Outer Product for A[0:2, 2] * B[2, 0:2] ---
    # Q1 = A02 B20 -> C00
    U[idx_U(0,2)][r] += 1.0
    V[idx_V(2,0)][r] += 1.0
    W[idx_W(0,0)][r] += 1.0
    r += 1
    
    # Q2 = A02 B21 -> C01
    U[idx_U(0,2)][r] += 1.0
    V[idx_V(2,1)][r] += 1.0
    W[idx_W(1,0)][r] += 1.0
    r += 1
    
    # Q3 = A12 B20 -> C10
    U[idx_U(1,2)][r] += 1.0
    V[idx_V(2,0)][r] += 1.0
    W[idx_W(0,1)][r] += 1.0
    r += 1
    
    # Q4 = A12 B21 -> C11
    U[idx_U(1,2)][r] += 1.0
    V[idx_V(2,1)][r] += 1.0
    W[idx_W(1,1)][r] += 1.0
    r += 1
    
    # --- Matrix-Vector for A[0:2, 0:3] * B[0:3, 2] ---
    # R1 = A00 B02 -> C02
    U[idx_U(0,0)][r] += 1.0
    V[idx_V(0,2)][r] += 1.0
    W[idx_W(2,0)][r] += 1.0
    r += 1
    
    # R2 = A01 B12 -> C02
    U[idx_U(0,1)][r] += 1.0
    V[idx_V(1,2)][r] += 1.0
    W[idx_W(2,0)][r] += 1.0
    r += 1
    
    # R3 = A02 B22 -> C02
    U[idx_U(0,2)][r] += 1.0
    V[idx_V(2,2)][r] += 1.0
    W[idx_W(2,0)][r] += 1.0
    r += 1
    
    # R4 = A10 B02 -> C12
    U[idx_U(1,0)][r] += 1.0
    V[idx_V(0,2)][r] += 1.0
    W[idx_W(2,1)][r] += 1.0
    r += 1
    
    # R5 = A11 B12 -> C12
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(1,2)][r] += 1.0
    W[idx_W(2,1)][r] += 1.0
    r += 1
    
    # R6 = A12 B22 -> C12
    U[idx_U(1,2)][r] += 1.0
    V[idx_V(2,2)][r] += 1.0
    W[idx_W(2,1)][r] += 1.0
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
