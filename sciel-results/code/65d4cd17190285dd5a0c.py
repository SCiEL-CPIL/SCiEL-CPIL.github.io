import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # 8 Strassen blocks of 2x2x2 (rank 7 each) -> 56
    # 1 Schoolbook block of 1x4x8 (rank 32) -> 32
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen part: Rows 0,1 of A
    # A blocks: 2x2 (rows 0,1) x 2x2 (cols 0,1) and 2x2 (rows 0,1) x 2x2 (cols 2,3)
    # B blocks: 2x2 (rows 0,1) x 2x2 (cols 0,1), etc.
    
    # Define the 2x2 Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate over the 8 blocks
    # j_block in {0, 1} corresponds to A cols {0,1} and {2,3}
    # k_block in {0, 1, 2, 3} corresponds to B cols {0,1}, {2,3}, {4,5}, {6,7}
    
    for j_block in range(2):
        for k_block in range(4):
            # Indices
            a_r0, a_r1 = 0, 1
            a_c0, a_c1 = 2 * j_block, 2 * j_block + 1
            b_r0, b_r1 = 2 * j_block, 2 * j_block + 1 # Same as A cols
            b_c0, b_c1 = 2 * k_block, 2 * k_block + 1
            
            # Global indices for U (A)
            u_00 = a_r0 * m + a_c0
            u_01 = a_r0 * m + a_c1
            u_10 = a_r1 * m + a_c0
            u_11 = a_r1 * m + a_c1
            
            # Global indices for V (B)
            v_00 = b_r0 * p + b_c0
            v_01 = b_r0 * p + b_c1
            v_10 = b_r1 * p + b_c0
            v_11 = b_r1 * p + b_c1
            
            # Global indices for W (C)
            # W index: k * n + i
            w_00 = b_c0 * n + a_r0
            w_01 = b_c1 * n + a_r0
            w_10 = b_c0 * n + a_r1
            w_11 = b_c1 * n + a_r1
            
            # Add 7 columns for this block
            
            # M1
            U[u_00][col_idx] += 1
            U[u_11][col_idx] += 1
            V[v_00][col_idx] += 1
            V[v_11][col_idx] += 1
            W[w_00][col_idx] += 1
            W[w_11][col_idx] += 1
            col_idx += 1
            
            # M2
            U[u_10][col_idx] += 1
            U[u_11][col_idx] += 1
            V[v_00][col_idx] += 1
            W[w_10][col_idx] += 1
            W[w_11][col_idx] += -1
            col_idx += 1
            
            # M3
            U[u_00][col_idx] += 1
            V[v_01][col_idx] += 1
            V[v_11][col_idx] += -1
            W[w_01][col_idx] += 1
            W[w_11][col_idx] += 1
            col_idx += 1
            
            # M4
            U[u_11][col_idx] += 1
            V[v_10][col_idx] += 1
            V[v_00][col_idx] += -1
            W[w_00][col_idx] += 1
            W[w_10][col_idx] += 1
            col_idx += 1
            
            # M5
            U[u_00][col_idx] += 1
            U[u_01][col_idx] += 1
            V[v_11][col_idx] += 1
            W[w_00][col_idx] += -1
            W[w_01][col_idx] += 1
            col_idx += 1
            
            # M6
            U[u_10][col_idx] += 1
            U[u_00][col_idx] += -1
            V[v_00][col_idx] += 1
            V[v_01][col_idx] += 1
            W[w_11][col_idx] += 1
            col_idx += 1
            
            # M7
            U[u_01][col_idx] += 1
            U[u_11][col_idx] += -1
            V[v_10][col_idx] += 1
            V[v_11][col_idx] += 1
            W[w_00][col_idx] += 1
            W[w_10][col_idx] += -1
            col_idx += 1

    # Schoolbook part: Row 2 of A
    # i = 2
    for j in range(m):
        for k in range(p):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            
            U[u_idx][col_idx] += 1
            V[v_idx][col_idx] += 1
            W[w_idx][col_idx] += 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
