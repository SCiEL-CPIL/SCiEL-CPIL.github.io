import time

_DEADLINE = 145.0
_START = time.time()

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    # n=2, m=4, p=4
    # U: 8 x 28
    # V: 16 x 28
    # W: 8 x 28
    
    # We construct the rank 28 solution using 4 blocks of Strassen (rank 7 each).
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Indices for U (i*m + l):
    # A_L (cols 0,1): i=0,1; l=0,1 -> 0,1,4,5
    # A_R (cols 2,3): i=0,1; l=2,3 -> 2,3,6,7
    
    # Indices for V (l*p + k):
    # B_TL (rows 0,1; cols 0,1): l=0,1; k=0,1 -> 0,1,4,5
    # B_BL (rows 2,3; cols 0,1): l=2,3; k=0,1 -> 8,9,12,13
    # B_TR (rows 0,1; cols 2,3): l=0,1; k=2,3 -> 2,3,6,7
    # B_BR (rows 2,3; cols 2,3): l=2,3; k=2,3 -> 10,11,14,15
    
    # Indices for W (k*n + i):
    # C_L (cols 0,1): k=0,1; i=0,1 -> 0,2,1,3 (z00=0, z01=2, z10=1, z11=3)
    # C_R (cols 2,3): k=2,3; i=0,1 -> 4,6,5,7 (z00=4, z01=6, z10=5, z11=7)
    
    rank = 28
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    
    # Strassen coefficients for 2x2 block multiplication
    # M1: (x00+x11)(y00+y11)
    # M2: (x10+x11)y00
    # M3: x00(y01-y11)
    # M4: x11(y10-y00)
    # M5: (x00+x01)y11
    # M6: (x10-x00)(y00+y01)
    # M7: (x01-x11)(y10+y11)
    
    def add_block(col_offset, X, Y, Z):
        # X = [x00, x01, x10, x11]
        # Y = [y00, y01, y10, y11]
        # Z = [z00, z01, z10, z11]
        
        # M1
        U[X[0]][col_offset] += 1
        U[X[3]][col_offset] += 1
        V[Y[0]][col_offset] += 1
        V[Y[3]][col_offset] += 1
        W[Z[0]][col_offset] += 1
        W[Z[3]][col_offset] += 1
        
        # M2
        U[X[2]][col_offset+1] += 1
        U[X[3]][col_offset+1] += 1
        V[Y[0]][col_offset+1] += 1
        W[Z[2]][col_offset+1] += 1
        W[Z[3]][col_offset+1] -= 1
        
        # M3
        U[X[0]][col_offset+2] += 1
        V[Y[1]][col_offset+2] += 1
        V[Y[3]][col_offset+2] -= 1
        W[Z[1]][col_offset+2] += 1
        W[Z[3]][col_offset+2] += 1
        
        # M4
        U[X[3]][col_offset+3] += 1
        V[Y[2]][col_offset+3] += 1
        V[Y[0]][col_offset+3] -= 1
        W[Z[0]][col_offset+3] += 1
        W[Z[2]][col_offset+3] += 1
        
        # M5
        U[X[0]][col_offset+4] += 1
        U[X[1]][col_offset+4] += 1
        V[Y[3]][col_offset+4] += 1
        W[Z[0]][col_offset+4] -= 1
        W[Z[1]][col_offset+4] += 1
        
        # M6
        U[X[2]][col_offset+5] += 1
        U[X[0]][col_offset+5] -= 1
        V[Y[0]][col_offset+5] += 1
        V[Y[1]][col_offset+5] += 1
        W[Z[3]][col_offset+5] += 1
        
        # M7
        U[X[1]][col_offset+6] += 1
        U[X[3]][col_offset+6] -= 1
        V[Y[2]][col_offset+6] += 1
        V[Y[3]][col_offset+6] += 1
        W[Z[0]][col_offset+6] += 1

    # Block 1: A_L * B_TL -> C_L
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A_R * B_BL -> C_L
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A_L * B_TR -> C_R
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A_R * B_BR -> C_R
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U, "V": V, "W": W}
