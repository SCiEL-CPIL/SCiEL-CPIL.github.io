import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(A_base, B_base, C_base, col_offset):
        a00, a01, a10, a11 = A_base, A_base + 1, A_base + m, A_base + m + 1
        b00, b01, b10, b11 = B_base, B_base + 1, B_base + p, B_base + p + 1
        c00, c01, c10, c11 = C_base, C_base + n, C_base + 1, C_base + n + 1
        
        # P1
        col = col_offset
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c10][col] += 1
        
        # P2
        col = col_offset + 1
        U[a01][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c01][col] += 1
        W[c10][col] -= 1
        
        # P3
        col = col_offset + 2
        U[a00][col] += 1
        V[b10][col] += 1
        V[b11][col] -= 1
        W[c00][col] -= 1
        W[c01][col] += 1
        
        # P4
        col = col_offset + 3
        U[a11][col] += 1
        V[b00][col] += 1
        V[b10][col] -= 1
        W[c00][col] -= 1
        W[c10][col] += 1
        
        # P5
        col = col_offset + 4
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c10][col] -= 1
        W[c11][col] += 1
        
        # P6
        col = col_offset + 5
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        
        # P7
        col = col_offset + 6
        U[a10][col] += 1
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    add_strassen_block(0, 0, 0, 0)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    add_strassen_block(2, 8, 0, 7)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    add_strassen_block(0, 2, 4, 14)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    add_strassen_block(2, 10, 4, 21)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Optional: Verify
    # U_np = np.array(U)
    # V_np = np.array(V)
    # W_np = np.array(W)
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             val = sum(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] for r in range(rank))
    #             if val != 1:
    #                 # Fallback to a valid higher rank solution if needed?
    #                 # But 28 is minimal, so if it fails, it's a bug.
    #                 pass

    return {"U": U, "V": V, "W": W}
