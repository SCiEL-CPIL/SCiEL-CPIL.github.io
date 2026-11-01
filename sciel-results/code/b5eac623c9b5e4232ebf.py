import time
import sys

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a Strassen block for B columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal col
        # B indices: j*p + k
        # B00: 0*p + k0 = k0
        # B01: 0*p + k1 = k1
        # B10: 1*p + k0 = 7 + k0
        # B11: 1*p + k1 = 7 + k1
        
        # W indices: k*n + i
        # C00: k0*2 + 0 = 2*k0
        # C01: k1*2 + 0 = 2*k1
        # C10: k0*2 + 1 = 2*k0 + 1
        # C11: k1*2 + 1 = 2*k1 + 1
        
        # 1. (A00 + A11)(B00 + B11) -> C00 + C11
        U[0][col] += 1 # A00
        U[3][col] += 1 # A11
        V[k0][col] += 1 # B00
        V[7 + k1][col] += 1 # B11
        W[2 * k0][col] += 1 # C00
        W[2 * k1 + 1][col] += 1 # C11
        col += 1
        
        # 2. (A10 + A11)B00 -> C10
        U[2][col] += 1 # A10
        U[3][col] += 1 # A11
        V[k0][col] += 1 # B00
        W[2 * k0 + 1][col] += 1 # C10
        col += 1
        
        # 3. A00(B01 - B11) -> C01
        U[0][col] += 1 # A00
        V[k1][col] += 1 # B01
        V[7 + k1][col] += -1 # B11
        W[2 * k1][col] += 1 # C01
        col += 1
        
        # 4. A11(B10 - B00) -> C10
        U[3][col] += 1 # A11
        V[7 + k0][col] += 1 # B10
        V[k0][col] += -1 # B00
        W[2 * k0 + 1][col] += 1 # C10
        col += 1
        
        # 5. (A00 + A01)B11 -> C01
        U[0][col] += 1 # A00
        U[1][col] += 1 # A01
        V[7 + k1][col] += 1 # B11
        W[2 * k1][col] += 1 # C01
        col += 1
        
        # 6. (A10 - A00)(B00 + B01) -> C11
        U[2][col] += 1 # A10
        U[0][col] += -1 # A00
        V[k0][col] += 1 # B00
        V[k1][col] += 1 # B01
        W[2 * k1 + 1][col] += 1 # C11
        col += 1
        
        # 7. (A01 - A11)(B10 + B11) -> C00 - C11
        U[1][col] += 1 # A01
        U[3][col] += -1 # A11
        V[7 + k0][col] += 1 # B10
        V[7 + k1][col] += 1 # B11
        W[2 * k0][col] += -1 # C00
        W[2 * k1 + 1][col] += 1 # C11
        col += 1

    # Function to add a schoolbook block for B column k0
    def add_schoolbook_block(k0):
        nonlocal col
        # B indices: k0, 7+k0
        # W indices: 2*k0, 2*k0+1
        
        # 1. A00 B00 -> C00
        U[0][col] += 1
        V[k0][col] += 1
        W[2 * k0][col] += 1
        col += 1
        
        # 2. A01 B10 -> C00
        U[1][col] += 1
        V[7 + k0][col] += 1
        W[2 * k0][col] += 1
        col += 1
        
        # 3. A10 B00 -> C10
        U[2][col] += 1
        V[k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        # 4. A11 B10 -> C10
        U[3][col] += 1
        V[7 + k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1

    # Blocks 0,1
    add_strassen_block(0, 1)
    # Blocks 2,3
    add_strassen_block(2, 3)
    # Blocks 4,5
    add_strassen_block(4, 5)
    # Block 6
    add_schoolbook_block(6)
    
    return {"U": U, "V": V, "W": W}

