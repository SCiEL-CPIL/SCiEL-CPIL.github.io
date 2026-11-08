"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B = [A*B1, A*B2, A*B3]
    # Each 4x4 mult: rank 16 (schoolbook), 4x1 mult: rank 4
    # Total rank = 16 + 16 + 4 = 36
    
    total_rank = 36
    U_full = [[0]*total_rank for _ in range(16)]  # n*m = 16
    V_full = [[0]*total_rank for _ in range(36)]  # m*p = 36
    W_full = [[0]*total_rank for _ in range(36)]  # p*n = 36
    
    # Block 1: A*B1 where B1 = B[:, 0:4] (4x4)
    # For schoolbook 4x4: term (i,j,k) has U[i*4+j]=1, V[j*4+k]=1, W[k*4+i]=1
    # In full: U[i*4+j, c], V[j*9+k, c], W[k*4+i, c] for k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                c = i*16 + j*4 + k  # 0..15
                U_full[i*4+j][c] = 1
                V_full[j*9+k][c] = 1
                W_full[k*4+i][c] = 1
    
    # Block 2: A*B2 where B2 = B[:, 4:8] (4x4)
    # k in 4..7
    for i in range(4):
        for j in range(4):
            for k in range(4):
                c = 16 + i*16 + j*4 + k  # 16..31
                U_full[i*4+j][c] = 1
                V_full[j*9+(4+k)][c] = 1
                W_full[(4+k)*4+i][c] = 1
    
    # Block 3: A*B3 where B3 = B[:, 8] (4x1)
    # Only k=8
    for i in range(4):
        for j in range(4):
            c = 32 + i*4 + j  # 32..35
            U_full[i*4+j][c] = 1
            V_full[j*9+8][c] = 1
            W_full[8*4+i][c] = 1
    
    # Verify: sum_c U[i*4+j,c] * V[j*9+k,c] * W[k*4+i,c] = 1 for all i,j,k
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(36):
                    s += U_full[i*4+j][c] * V_full[j*9+k][c] * W_full[k*4+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to full schoolbook rank 144
        rank = 144
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(36)]
        W = [[0]*rank for _ in range(36)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j][col] = 1
                    V[j*9+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
