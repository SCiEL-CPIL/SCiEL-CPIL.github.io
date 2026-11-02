import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_left (2x2) * B_top (2x10)
    # A_left indices in U:
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    
    # B_top indices in V:
    # B[0,k] -> V[k]
    # B[1,k] -> V[10+k]
    
    # C_top indices in W:
    # C[0,k] -> W[2*k]
    # C[1,k] -> W[2*k+1]
    
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # V indices for B_top block
        v_b00 = k0       # B[0, k0]
        v_b01 = k1       # B[0, k0+1]
        v_b10 = 10 + k0  # B[1, k0]
        v_b11 = 10 + k1  # B[1, k0+1]
        
        # W indices for C_top block
        w_c00 = 2 * k0       # C[0, k0]
        w_c01 = 2 * k0 + 1   # C[0, k0+1]
        w_c10 = 2 * k1       # C[1, k0]
        w_c11 = 2 * k1 + 1   # C[1, k0+1]
        
        # U indices for A_left
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_a00][col] += 1.0
        U[u_a11][col] += 1.0
        V[v_b00][col] += 1.0
        V[v_b11][col] += 1.0
        W[w_c00][col] += 1.0
        W[w_c01][col] += 1.0
        W[w_c10][col] += 1.0
        W[w_c11][col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[u_a10][col] += 1.0
        U[u_a11][col] += 1.0
        V[v_b00][col] += 1.0
        W[w_c10][col] += 1.0
        W[w_c11][col] -= 1.0
        col += 1
        
        # M3 = A00(B11 - B01)
        U[u_a00][col] += 1.0
        V[v_b11][col] += 1.0
        V[v_b01][col] -= 1.0
        W[w_c01][col] += 1.0
        W[w_c11][col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[u_a11][col] += 1.0
        V[v_b10][col] += 1.0
        V[v_b00][col] -= 1.0
        W[w_c00][col] += 1.0
        W[w_c11][col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[u_a00][col] += 1.0
        U[u_a01][col] += 1.0
        V[v_b11][col] += 1.0
        W[w_c00][col] -= 1.0
        W[w_c10][col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_a10][col] += 1.0
        U[u_a00][col] -= 1.0
        V[v_b00][col] += 1.0
        V[v_b01][col] += 1.0
        W[w_c11][col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_a01][col] += 1.0
        U[u_a11][col] -= 1.0
        V[v_b10][col] += 1.0
        V[v_b11][col] += 1.0
        W[w_c00][col] += 1.0
        W[w_c11][col] += 1.0
        col += 1

    # Part 2: A_right (2x1) * B_bottom (1x10)
    # A_right indices in U:
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    
    # B_bottom indices in V:
    # B[2,k] -> V[20+k]
    
    # C_bottom indices in W:
    # C[0,k] -> W[2*k]
    # C[1,k] -> W[2*k+1]
    
    for k in range(10):
        # Term for C[0,k] = A[0,2] * B[2,k]
        U[2][col] += 1.0
        V[20 + k][col] += 1.0
        W[2 * k][col] += 1.0
        col += 1
        
        # Term for C[1,k] = A[1,2] * B[2,k]
        U[5][col] += 1.0
        V[20 + k][col] += 1.0
        W[2 * k + 1][col] += 1.0
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
