"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    
    # Build M_{4,4,4} decomposition using 8 Strassen blocks (Rank 56)
    # Build M_{4,4,1} decomposition using schoolbook (Rank 16)
    # Total Rank = 56 + 56 + 16 = 128
    
    # 1. Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = _strassen_2x2()
    
    # 2. Construct M_{4,4,4} from 8 blocks
    # A is 4x4, B is 4x4, C is 4x4
    # Blocks:
    # A00 (0:2, 0:2), A01 (0:2, 2:4)
    # A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B00 (0:2, 0:2), B01 (0:2, 2:4)
    # B10 (2:4, 0:2), B11 (2:4, 2:4)
    # C00 (0:2, 0:2), C01 (0:2, 2:4)
    # C10 (2:4, 0:2), C11 (2:4, 2:4)
    
    # Products needed:
    # P1 = A00 B00 -> C00
    # P2 = A01 B10 -> C00
    # P3 = A00 B01 -> C01
    # P4 = A01 B11 -> C01
    # P5 = A10 B00 -> C10
    # P6 = A11 B10 -> C10
    # P7 = A10 B01 -> C11
    # P8 = A11 B11 -> C11
    
    # Strassen U, V, W shapes for 2x2x2:
    # U: (4, 7) indices (i,j)
    # V: (4, 7) indices (j,k)
    # W: (4, 7) indices (k,i)
    
    # We need to map local indices to global indices.
    # Local A index (i,j) in block (br, bc) corresponds to global A (br*2+i, bc*2+j)
    # Local B index (j,k) in block (br, bc) corresponds to global B (br*2+j, bc*2+k)
    # Local C index (k,i) in block (cr, cc) corresponds to global C (cr*2+k, cc*2+i)
    
    rank_444 = 56
    U_444 = [[0] * rank_444 for _ in range(16)]
    V_444 = [[0] * rank_444 for _ in range(16)]
    W_444 = [[0] * rank_444 for _ in range(16)]
    
    blocks = [
        (0, 0, 0, 0, 0, 0), # A00, B00, C00
        (0, 0, 1, 0, 0, 0), # A01, B10, C00 -> Wait, A01 is (0,1), B10 is (1,0). C00 is (0,0).
        # A row block 0, col block 1. B row block 1, col block 0.
        # Result C row block 0, col block 0.
        # Let's define block coords (row, col)
        # A(r_a, c_a), B(r_b, c_b), C(r_c, c_c)
        # Condition: r_c = r_a, c_c = c_b, and r_b = c_a.
        
        # P1: A(0,0) B(0,0) -> C(0,0). r_a=0, c_a=0. r_b=0, c_b=0. r_c=0, c_c=0.
        # P2: A(0,1) B(1,0) -> C(0,0). r_a=0, c_a=1. r_b=1, c_b=0. r_c=0, c_c=0.
        # P3: A(0,0) B(0,1) -> C(0,1). r_a=0, c_a=0. r_b=0, c_b=1. r_c=0, c_c=1.
        # P4: A(0,1) B(1,1) -> C(0,1). r_a=0, c_a=1. r_b=1, c_b=1. r_c=0, c_c=1.
        # P5: A(1,0) B(1,0) -> C(1,0). r_a=1, c_a=0. r_b=1, c_b=0. r_c=1, c_c=0.
        # P6: A(1,1) B(1,0) -> C(1,0). r_a=1, c_a=1. r_b=1, c_b=0. r_c=1, c_c=0.
        # P7: A(1,0) B(1,1) -> C(1,1). r_a=1, c_a=0. r_b=1, c_b=1. r_c=1, c_c=1.
        # P8: A(1,1) B(1,1) -> C(1,1). r_a=1, c_a=1. r_b=1, c_b=1. r_c=1, c_c=1.
    ]
    
    block_defs = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 1, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1)
    ]
    
    for idx, (ra, ca, rb, cb, rc, cc) in enumerate(block_defs):
        base = idx * 7
        for l in range(7):
            # U: A indices
            for i in range(2):
                for j in range(2):
                    val = U_s[i*2+j][l]
                    if val != 0:
                        U_444[(ra*2+i)*4 + (ca*2+j)][base+l] += val
            
            # V: B indices
            for j in range(2):
                for k in range(2):
                    val = V_s[j*2+k][l]
                    if val != 0:
                        V_444[(rb*2+j)*4 + (cb*2+k)][base+l] += val
            
            # W: C indices (k*n + i) -> (cc*2+k)*4 + (ra*2+i)
            for k in range(2):
                for i in range(2):
                    val = W_s[k*2+i][l]
                    if val != 0:
                        W_444[(cc*2+k)*4 + (ra*2+i)][base+l] += val

    # 3. Construct M_{4,4,1} (Schoolbook Rank 16)
    rank_441 = 16
    U_441 = [[0] * rank_441 for _ in range(16)]
    V_441 = [[0] * rank_441 for _ in range(4)] # B is 4x1
    W_441 = [[0] * rank_441 for _ in range(4)] # C is 4x1
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i*4+j][col] = 1
            V_441[j*1+0][col] = 1 # B index is j*1 + 0
            W_441[0*4+i][col] = 1 # C index is 0*4 + i
            col += 1

    # 4. Combine
    # M_{4,4,9} = M_{4,4,4} (cols 0-3) + M_{4,4,4} (cols 4-7) + M_{4,4,1} (col 8)
    # Note: The second M_{4,4,4} is for B columns 4-7.
    # The structure of B is 4x9.
    # B1 (4x4) corresponds to cols 0-3.
    # B2 (4x4) corresponds to cols 4-7.
    # B3 (4x1) corresponds to col 8.
    
    # U is the same for both 4x4 parts.
    # V is different: V1 for cols 0-3, V2 for cols 4-7.
    # W is different: W1 for cols 0-3, W2 for cols 4-7.
    
    total_rank = 56 + 56 + 16
    U_final = [[0] * total_rank for _ in range(16)]
    V_final = [[0] * total_rank for _ in range(36)] # 4x9
    W_final = [[0] * total_rank for _ in range(36)] # 9x4 -> p*n = 9*4 = 36
    
    # Part 1: M_{4,4,4} for B cols 0-3
    for i in range(16):
        for k in range(56):
            U_final[i][k] = U_444[i][k]
    
    for j in range(4):
        for k in range(4):
            for l in range(56):
                V_final[j*9+k][l] = V_444[j*4+k][l]
                
    for k in range(4):
        for i in range(4):
            for l in range(56):
                W_final[k*4+i][l] = W_444[k*4+i][l]

    # Part 2: M_{4,4,4} for B cols 4-7
    offset = 56
    for i in range(16):
        for k in range(56):
            U_final[i][k+offset] = U_444[i][k]
            
    for j in range(4):
        for k in range(4):
            for l in range(56):
                V_final[j*9+(k+4)][l+offset] = V_444[j*4+k][l]
                
    for k in range(4):
        for i in range(4):
            for l in range(56):
                W_final[(k+4)*4+i][l+offset] = W_444[k*4+i][l]

    # Part 3: M_{4,4,1} for B col 8
    offset2 = 112
    for i in range(16):
        for k in range(16):
            U_final[i][k+offset2] = U_441[i][k]
            
    for j in range(4):
        for k in range(16):
            V_final[j*9+8][k+offset2] = V_441[j][k]
            
    for i in range(4):
        for k in range(16):
            W_final[8*4+i][k+offset2] = W_441[i][k]

    return {"U": U_final, "V": V_final, "W": W_final}

def _strassen_2x2():
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3 (j*p+k, p=2)
    # C: 00, 10, 01, 11 -> 0, 2, 1, 3 (k*n+i, n=2)
    # Wait, C index is k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    U = [
        [1, 0, 1, 0, 1, -1, -1], # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, -1, 1, 1, 1, 1],  # B01
        [1, 0, 1, 0, 0, 0, 1],   # B10
        [1, 0, 0, -1, 1, 0, 1]   # B11
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],   # C00 (k=0, i=0)
        [0, 0, 0, 0, 1, 0, 1],   # C01 (k=0, i=1) -> Wait.
        # C00 = z1 + z4 - z5 + z7
        # C01 = z3 + z5
        # C10 = z2 + z4
        # C11 = z1 - z2 + z3 + z6
        
        # Let's re-verify W mapping.
        # W index is k*n+i.
        # C00 (k=0, i=0): idx 0. Coeffs: z1(1), z4(1), z5(-1), z7(1) -> [1, 0, 0, 1, -1, 0, 1]
        # C01 (k=0, i=1): idx 1. Coeffs: z3(1), z5(1) -> [0, 0, 1, 0, 1, 0, 0]
        # C10 (k=1, i=0): idx 2. Coeffs: z2(1), z4(1) -> [0, 1, 0, 1, 0, 0, 0]
        # C11 (k=1, i=1): idx 3. Coeffs: z1(1), z2(-1), z3(1), z6(1) -> [1, -1, 1, 0, 0, 1, 0]
        
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _schoolbook(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
