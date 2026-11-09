import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # We use 5 blocks of 2 columns in B.
    # Each block uses rank 11.
    # Total rank = 55.
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(p // 2):
        # Block b corresponds to B columns 2b, 2b+1
        # C columns 2b, 2b+1
        
        # Strassen part: A_left (cols 0,1) * B_top (rows 0,1)
        # A_left indices in U: 
        # A_00: 0*3+0 = 0
        # A_01: 0*3+1 = 1
        # A_10: 1*3+0 = 3
        # A_11: 1*3+1 = 4
        
        # B_top indices in V:
        # B_0,2b: 0*10 + 2b
        # B_0,2b+1: 0*10 + 2b+1
        # B_1,2b: 1*10 + 2b
        # B_1,2b+1: 1*10 + 2b+1
        
        # C indices in W:
        # C_0,2b: (2b)*2 + 0
        # C_0,2b+1: (2b+1)*2 + 0
        # C_1,2b: (2b)*2 + 1
        # C_1,2b+1: (2b+1)*2 + 1
        
        # Strassen Multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] += 1
        U[4][col] += 1
        V[0*10 + 2*b][col] += 1
        V[1*10 + 2*b+1][col] += 1
        # C00 += M1
        W[(2*b)*2 + 0][col] += 1
        # C11 += M1
        W[(2*b+1)*2 + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[3][col] += 1
        U[4][col] += 1
        V[0*10 + 2*b][col] += 1
        # C10 += M2
        W[(2*b)*2 + 1][col] += 1
        # C11 -= M2
        W[(2*b+1)*2 + 1][col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0][col] += 1
        V[0*10 + 2*b+1][col] += 1
        V[1*10 + 2*b+1][col] -= 1
        # C01 += M3
        W[(2*b+1)*2 + 0][col] += 1
        # C11 += M3
        W[(2*b+1)*2 + 1][col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[4][col] += 1
        V[1*10 + 2*b][col] += 1
        V[0*10 + 2*b][col] -= 1
        # C00 += M4
        W[(2*b)*2 + 0][col] += 1
        # C10 += M4
        W[(2*b)*2 + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0][col] += 1
        U[1][col] += 1
        V[1*10 + 2*b+1][col] += 1
        # C00 -= M5
        W[(2*b)*2 + 0][col] -= 1
        # C01 += M5
        W[(2*b+1)*2 + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[3][col] += 1
        U[0][col] -= 1
        V[0*10 + 2*b][col] += 1
        V[0*10 + 2*b+1][col] += 1
        # C11 += M6
        W[(2*b+1)*2 + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col] += 1
        U[4][col] -= 1
        V[1*10 + 2*b][col] += 1
        V[1*10 + 2*b+1][col] += 1
        # C00 += M7
        W[(2*b)*2 + 0][col] += 1
        # C10 -= M7
        W[(2*b)*2 + 1][col] -= 1
        col += 1
        
        # Schoolbook part: A_right (col 2) * B_bottom (row 2)
        # A_right indices in U:
        # A_02: 0*3+2 = 2
        # A_12: 1*3+2 = 5
        
        # B_bottom indices in V:
        # B_2,2b: 2*10 + 2b
        # B_2,2b+1: 2*10 + 2b+1
        
        # M8 = A02 * B2,2b
        U[2][col] += 1
        V[2*10 + 2*b][col] += 1
        # C0,2b += M8
        W[(2*b)*2 + 0][col] += 1
        col += 1
        
        # M9 = A02 * B2,2b+1
        U[2][col] += 1
        V[2*10 + 2*b+1][col] += 1
        # C0,2b+1 += M9
        W[(2*b+1)*2 + 0][col] += 1
        col += 1
        
        # M10 = A12 * B2,2b
        U[5][col] += 1
        V[2*10 + 2*b][col] += 1
        # C1,2b += M10
        W[(2*b)*2 + 1][col] += 1
        col += 1
        
        # M11 = A12 * B2,2b+1
        U[5][col] += 1
        V[2*10 + 2*b+1][col] += 1
        # C1,2b+1 += M11
        W[(2*b+1)*2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
