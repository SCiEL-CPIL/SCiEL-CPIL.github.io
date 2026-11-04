import time

def construct():
    """
    Returns a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    Split B (3x8) into B_left (3x4) and B_right (3x4).
    C = A * B_left + A * B_right.
    
    For each half (3x4), split A (2x3) into A1 (2x2) and A2 (2x1),
    and the half of B into B1 (2x4) and B2 (1x4).
    C_half = A1 * B1 + A2 * B2.
    
    A1*B1 (2x2 * 2x4) is split into two 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 uses Strassen's algorithm (rank 7).
    Total for A1*B1: 14.
    
    A2*B2 (2x1 * 1x4) is an outer product (rank 8).
    
    Total per half: 14 + 8 = 22.
    Total: 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        U[u_map[0]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10+=1, C11+=-1
        U[u_map[2]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        W[w_map[2]][col] += 1
        W[w_map[3]][col] += -1
        col += 1
        
        # M3: A00(B01-B11) -> C01+=1, C11+=1
        U[u_map[0]][col] += 1
        V[v_map[1]][col] += 1
        V[v_map[3]][col] += -1
        W[w_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00+=1, C10+=1
        U[u_map[3]][col] += 1
        V[v_map[2]][col] += 1
        V[v_map[0]][col] += -1
        W[w_map[0]][col] += 1
        W[w_map[2]][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00+=-1, C01+=1
        U[u_map[0]][col] += 1
        U[u_map[1]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += -1
        W[w_map[1]][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11+=1
        U[u_map[2]][col] += 1
        U[u_map[0]][col] += -1
        V[v_map[0]][col] += 1
        V[v_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00+=1
        U[u_map[1]][col] += 1
        U[u_map[3]][col] += -1
        V[v_map[2]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        col += 1

    # Process left half (k = 0..3) and right half (k = 4..7)
    for half in range(2):
        # A1 is always cols 0,1 of A
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        
        # Block 1 of B1
        b_start = half * 4
        v_map_b1 = {
            0: 0 * p + (b_start + 0),
            1: 0 * p + (b_start + 1),
            2: 1 * p + (b_start + 0),
            3: 1 * p + (b_start + 1)
        }
        w_map_c1 = {
            0: (b_start + 0) * n + 0,
            1: (b_start + 1) * n + 0,
            2: (b_start + 0) * n + 1,
            3: (b_start + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b1, w_map_c1)
        
        # Block 2 of B1
        b_start2 = half * 4 + 2
        v_map_b2 = {
            0: 0 * p + (b_start2 + 0),
            1: 0 * p + (b_start2 + 1),
            2: 1 * p + (b_start2 + 0),
            3: 1 * p + (b_start2 + 1)
        }
        w_map_c2 = {
            0: (b_start2 + 0) * n + 0,
            1: (b_start2 + 1) * n + 0,
            2: (b_start2 + 0) * n + 1,
            3: (b_start2 + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b2, w_map_c2)
        
        # Part 2: A2 (2x1) * B2 (1x4) - outer product, rank 8
        for i in range(2):
            for k in range(4):
                u_idx = i * m + 2
                v_idx = 2 * p + (half * 4 + k)
                w_idx = (half * 4 + k) * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
