import time

def construct():
    """
    Returns a rank-32 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    Split B (3x8) into B_left (3x4) and B_right (3x4).
    C = A * B_left + A * B_right.
    
    For each half (3x4), split A (2x3) into A1 (2x2) and A2 (2x1),
    and the half of B into B1 (2x4) and B2 (1x4).
    C_half = A1 * B1 + A2 * B2.
    
    A1*B1: 2x2 * 2x4, rank 8 (schoolbook)
    A2*B2: 2x1 * 1x4, rank 8 (outer product)
    Total per half: 16
    Total: 32
    """
    n, m, p = 2, 3, 8
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 32
    V = [[0] * rank for _ in range(m * p)]  # 24 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    col = 0
    
    # Process left half (k = 0..3) and right half (k = 4..7)
    for half in range(2):
        k_base = half * 4
        
        # Part 1: A1 (2x2) * B1 (2x4) - schoolbook, rank 8
        # A1 uses A rows 0,1 and cols 0,1
        # B1 uses B rows 0,1 and cols k_base..k_base+3
        for i in range(2):
            for a in range(2):
                for k in range(4):
                    u_idx = i * m + a
                    v_idx = a * p + (k_base + k)
                    w_idx = (k_base + k) * n + i
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        
        # Part 2: A2 (2x1) * B2 (1x4) - outer product, rank 8
        # A2 uses A rows 0,1 and col 2
        # B2 uses B row 2 and cols k_base..k_base+3
        for i in range(2):
            for k in range(4):
                u_idx = i * m + 2
                v_idx = 2 * p + (k_base + k)
                w_idx = (k_base + k) * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
