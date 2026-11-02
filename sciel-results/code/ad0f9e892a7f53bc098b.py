"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 25: 3 Strassen blocks (7 each) + 1 Schoolbook block (4)
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal col
        # B indices
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        
        # C indices
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        
        # A indices
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        
        # M1: (A00 + A11)(B00 + B11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2: (A10 + A11)B00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3: A00(B01 - B11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] += -1
        W[c01][col] += 1
        col += 1
        
        # M4: A11(B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] += -1
        W[c00][col] += -1
        W[c10][col] += 1
        col += 1
        
        # M5: (A00 + A01)B11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c01][col] += 1
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[a10][col] += 1
        U[a00][col] += -1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c00][col] += 1
        col += 1
        
        # M7: (A01 - A11)(B10 + B11)
        U[a01][col] += 1
        U[a11][col] += -1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c11][col] += 1
        col += 1

    # Helper to add a Schoolbook block for column k
    def add_schoolbook_block(k):
        nonlocal col
        # B indices
        b0 = 0 * p + k
        b1 = 1 * p + k
        
        # C indices
        c0 = k * n + 0
        c1 = k * n + 1
        
        # A indices
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        
        # T1: A00 B0 -> C0
        U[a00][col] += 1
        V[b0][col] += 1
        W[c0][col] += 1
        col += 1
        
        # T2: A01 B1 -> C0
        U[a01][col] += 1
        V[b1][col] += 1
        W[c0][col] += 1
        col += 1
        
        # T3: A10 B0 -> C1
        U[a10][col] += 1
        V[b0][col] += 1
        W[c1][col] += 1
        col += 1
        
        # T4: A11 B1 -> C1
        U[a11][col] += 1
        V[b1][col] += 1
        W[c1][col] += 1
        col += 1

    # Apply blocks
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_schoolbook_block(6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
