import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 solution using Strassen for 2x2 blocks
    # Split B into 4 blocks of width 2
    # For each block, compute A * B_block (2x4 * 4x2)
    # Split A into A_L, A_R (2x2)
    # B_block into B_top, B_bot (2x2)
    # C_block = A_L B_top + A_R B_bot
    # Each 2x2 mult uses 7 products
    # Total 4 * 2 * 7 = 56 products
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Strassen 2x2 multiplication coefficients
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    def add_strassen_block(a_offset, b_offset, c_offset, sign=1):
        # a_offset: index in A (i*m+j)
        # b_offset: index in B (j*p+k)
        # c_offset: index in C (k*n+i)
        nonlocal col
        # P1
        U[0*m + 0, col] = 1
        U[1*m + 1, col] = 1
        V[0*p + 0, col] = 1
        V[1*p + 1, col] = 1
        W[0*n + 0, col] = 1
        W[1*n + 1, col] = 1
        col += 1
        # P2
        U[1*m + 0, col] = 1
        U[1*m + 1, col] = 1
        V[0*p + 0, col] = 1
        W[0*n + 0, col] = 1
        W[1*n + 1, col] = 1
        col += 1
        # P3
        U[0*m + 0, col] = 1
        V[0*p + 1, col] = 1
        V[1*p + 1, col] = -1
        W[0*n + 0, col] = 1
        W[1*n + 1, col] = 1
        col += 1
        # P4
        U[1*m + 1, col] = 1
        V[1*p + 0, col] = 1
        V[0*p + 0, col] = -1
        W[0*n + 0, col] = 1
        W[1*n + 1, col] = 1
        col += 1
        # P5
        U[0*m + 0, col] = 1
        U[0*m + 1, col] = 1
        V[1*p + 1, col] = 1
        W[0*n + 0, col] = 1
        W[1*n + 1, col] = 1
        col += 1
        # P6
        U[1*m + 0, col] = 1
        U[0*m + 0, col] = -1
        V[0*p + 0, col] = 1
        V[0*p + 1, col] = 1
        W[0*n + 0, col] = 1
        W[1*n + 1, col] = 1
        col += 1
        # P7
        U[0*m + 1, col] = 1
        U[1*m + 1, col] = -1
        V[1*p + 0, col] = 1
        V[1*p + 1, col] = 1
        W[0*n + 0, col] = 1
        W[1*n + 1, col] = 1
        col += 1

    # This simple mapping is incorrect for the full tensor.
    # I need to map the Strassen products to the specific A and B indices.
    
    # Let's build it properly.
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for r in range(4): # 4 blocks of B
        # B block r is columns 2r, 2r+1
        # A is full 2x4
        # Split A into A_L (cols 0,1) and A_R (cols 2,3)
        # B_top is rows 0,1 of B block
        # B_bot is rows 2,3 of B block
        
        # We compute C_block = A_L B_top + A_R B_bot
        # This is sum of two 2x2 multiplications.
        
        # Multiplication 1: A_L * B_top
        # A_L indices: (0,0), (0,1), (1,0), (1,1) in A
        # B_top indices: (0, 2r), (0, 2r+1), (1, 2r), (1, 2r+1) in B
        # C indices: (2r, 0), (2r, 1), (2r+1, 0), (2r+1, 1) in C
        
        # Strassen for A_L * B_top
        # P1 = (A_L00 + A_L11)(B_top00 + B_top11)
        U[0*m + 0, col] = 1
        U[1*m + 1, col] = 1
        V[0*p + 2*r, col] = 1
        V[1*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P2 = (A_L10 + A_L11) B_top00
        U[1*m + 0, col] = 1
        U[1*m + 1, col] = 1
        V[0*p + 2*r, col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P3 = A_L00 (B_top01 - B_top11)
        U[0*m + 0, col] = 1
        V[0*p + (2*r+1), col] = 1
        V[1*p + (2*r+1), col] = -1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P4 = A_L11 (B_top10 - B_top00)
        U[1*m + 1, col] = 1
        V[1*p + 2*r, col] = 1
        V[0*p + 2*r, col] = -1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P5 = (A_L00 + A_L01) B_top11
        U[0*m + 0, col] = 1
        U[0*m + 1, col] = 1
        V[1*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P6 = (A_L10 - A_L00) (B_top00 + B_top01)
        U[1*m + 0, col] = 1
        U[0*m + 0, col] = -1
        V[0*p + 2*r, col] = 1
        V[0*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P7 = (A_L01 - A_L11) (B_top10 + B_top11)
        U[0*m + 1, col] = 1
        U[1*m + 1, col] = -1
        V[1*p + 2*r, col] = 1
        V[1*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # Multiplication 2: A_R * B_bot
        # A_R indices: (0,2), (0,3), (1,2), (1,3) in A
        # B_bot indices: (2, 2r), (2, 2r+1), (3, 2r), (3, 2r+1) in B
        # C indices: same as above
        
        # P8 = (A_R00 + A_R11)(B_bot00 + B_bot11)
        U[0*m + 2, col] = 1
        U[1*m + 3, col] = 1
        V[2*p + 2*r, col] = 1
        V[3*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P9 = (A_R10 + A_R11) B_bot00
        U[1*m + 2, col] = 1
        U[1*m + 3, col] = 1
        V[2*p + 2*r, col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P10 = A_R00 (B_bot01 - B_bot11)
        U[0*m + 2, col] = 1
        V[2*p + (2*r+1), col] = 1
        V[3*p + (2*r+1), col] = -1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P11 = A_R11 (B_bot10 - B_bot00)
        U[1*m + 3, col] = 1
        V[3*p + 2*r, col] = 1
        V[2*p + 2*r, col] = -1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P12 = (A_R00 + A_R01) B_bot11
        U[0*m + 2, col] = 1
        U[0*m + 3, col] = 1
        V[3*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P13 = (A_R10 - A_R00) (B_bot00 + B_bot01)
        U[1*m + 2, col] = 1
        U[0*m + 2, col] = -1
        V[2*p + 2*r, col] = 1
        V[2*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1
        
        # P14 = (A_R01 - A_R11) (B_bot10 + B_bot11)
        U[0*m + 3, col] = 1
        U[1*m + 3, col] = -1
        V[3*p + 2*r, col] = 1
        V[3*p + (2*r+1), col] = 1
        W[(2*r)*n + 0, col] = 1
        W[(2*r+1)*n + 1, col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
