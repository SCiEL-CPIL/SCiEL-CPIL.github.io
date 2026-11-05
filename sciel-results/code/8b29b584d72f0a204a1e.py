import time
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    rank = 17
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add to matrices
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][val] if False else None # Placeholder, will use direct assignment
        pass

    # We will build the columns explicitly
    
    # Col 0: M1 = (A00 + A11)(B00 + B11)
    # U: A00(0), A11(4)
    # V: B00(0), B11(4)
    # W: C00(0), C11(3)
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # Col 1: M2 = (A10 + A11)B00
    # U: A10(3), A11(4)
    # V: B00(0)
    # W: C10(1) +1, C11(3) -1
    U[3][1] += 1; U[4][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    
    # Col 2: M3 = A00(B01 - B11)
    # U: A00(0)
    # V: B01(1) +1, B11(4) -1
    # W: C01(2) +1, C11(3) +1
    U[0][2] += 1
    V[1][2] += 1; V[4][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    # Col 3: M4 = A11(B10 - B00)
    # U: A11(4)
    # V: B10(3) +1, B00(0) -1
    # W: C00(0) +1, C10(1) +1
    U[4][3] += 1
    V[3][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    # Col 4: M5 = (A00 + A01)B11
    # U: A00(0), A01(1)
    # V: B11(4)
    # W: C00(0) -1, C01(2) +1
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    # Col 5: M6 = (A10 - A00)(B00 + B01)
    # U: A10(3) +1, A00(0) -1
    # V: B00(0) +1, B01(1) +1
    # W: C11(3) +1
    U[3][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # Col 6: M7 = (A01 - A11)(B10 + B11)
    # U: A01(1) +1, A11(4) -1
    # V: B10(3) +1, B11(4) +1
    # W: C00(0) +1, C11(3) +1
    U[1][6] += 1; U[4][6] -= 1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1; W[3][6] += 1
    
    # Col 7: A00 B02
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # Col 8: A01 B12
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # Col 9: A10 B02
    U[3][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # Col 10: A11 B12
    U[4][10] += 1
    V[5][10] += 1
    W[5][10] += 1
    
    # Col 11: A02 B20
    U[2][11] += 1
    V[6][11] += 1
    W[0][11] += 1
    
    # Col 12: A02 B21
    U[2][12] += 1
    V[7][12] += 1
    W[2][12] += 1
    
    # Col 13: A02 B22
    U[2][13] += 1
    V[8][13] += 1
    W[4][13] += 1
    
    # Col 14: A12 B20
    U[5][14] += 1
    V[6][14] += 1
    W[1][14] += 1
    
    # Col 15: A12 B21
    U[5][15] += 1
    V[7][15] += 1
    W[3][15] += 1
    
    # Col 16: A12 B22
    U[5][16] += 1
    V[8][16] += 1
    W[5][16] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
