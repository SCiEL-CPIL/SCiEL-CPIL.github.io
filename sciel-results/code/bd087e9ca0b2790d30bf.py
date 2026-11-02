import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses a block decomposition based on Strassen's algorithm for 2x2 matrices.
    This reduces the rank from 32 (schoolbook) to 28.
    """
    n, m, p = DIMENSIONS
    
    # Rank of Strassen 2x2x2 is 7
    # We have 4 such blocks
    rank = 4 * 7
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define Strassen factors for 2x2x2
    # U_local shape (4, 7), V_local shape (4, 7), W_local shape (4, 7)
    # Rows correspond to indices 0,1,2,3 for A, B, C respectively
    # A: 0=A11, 1=A12, 2=A21, 3=A22
    # B: 0=B11, 1=B12, 2=B21, 3=B22
    # C: 0=C11, 1=C21, 2=C12, 3=C22 (Note: W index is k*n+i, so for 2x2: 0=C00, 1=C10, 2=C01, 3=C11)
    
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11 + A22, A21+A22, A11, A22, A11+A12, A21-A11, A12-A22
        [0, 0, 0, 0, 1, 0, 1],   # A12 terms
        [0, 1, 0, 0, 0, 1, 0],   # A21 terms
        [1, 1, 0, 1, 0, 0, -1]   # A22 terms
    ], dtype=float).T # Transpose to (7, 4) for easier handling? No, keep (4, 7)
    
    # Let's re-verify the Strassen vectors carefully.
    # M1: (A11+A22)(B11+B22) -> C11+C22
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (C11 is idx 0, C22 is idx 3)
    
    # M2: (A21+A22)B11 -> C21+C22
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 1, 0, 1] (C21 is idx 1, C22 is idx 3)
    
    # M3: A11(B12-B22) -> C12+C22
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: [0, 0, 1, 1] (C12 is idx 2, C22 is idx 3)
    
    # M4: A22(B21-B11) -> C11+C21
    # U: [0, 0, 0, 1]
    # V: [-1, 0, 1, 0]
    # W: [1, 1, 0, 0] (C11 is idx 0, C21 is idx 1)
    
    # M5: (A11+A12)B22 -> C11-C21
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: [1, -1, 0, 0] (C11 is idx 0, C21 is idx 1)
    
    # M6: (A21-A11)(B11+B12) -> C22-C12
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # W: [0, 0, -1, 1] (C12 is idx 2, C22 is idx 3)
    
    # M7: (A12-A22)(B21+B22) -> C11-C22
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: [1, 0, 0, -1] (C11 is idx 0, C22 is idx 3)
    
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 0, -1, 0],
        [1, 1, 1, 0, 0, 1, -1]
    ], dtype=float)
    
    # Helper to place local matrices into global
    def place_block(U_g, V_g, W_g, u_map, v_map, w_map, u_l, v_l, w_l, offset):
        for c in range(7):
            for i in range(4):
                if u_l[i, c] != 0:
                    U_g[u_map[i], offset + c] += u_l[i, c]
                if v_l[i, c] != 0:
                    V_g[v_map[i], offset + c] += v_l[i, c]
                if w_l[i, c] != 0:
                    W_g[w_map[i], offset + c] += w_l[i, c]
    
    # Block 1: A(0:2, 0:2) * B(0:2, 0:2) -> C(0:2, 0:2)
    # A indices: 0,1,4,5
    # B indices: 0,1,4,5
    # C indices: 0,1,2,3
    u_map_1 = [0, 1, 4, 5]
    v_map_1 = [0, 1, 4, 5]
    w_map_1 = [0, 1, 2, 3]
    place_block(U, V, W, u_map_1, v_map_1, w_map_1, U_local, V_local, W_local, 0)
    
    # Block 2: A(0:2, 0:2) * B(0:2, 2:4) -> C(0:2, 2:4)
    # A indices: 0,1,4,5
    # B indices: 2,3,6,7
    # C indices: 4,5,6,7
    u_map_2 = [0, 1, 4, 5]
    v_map_2 = [2, 3, 6, 7]
    w_map_2 = [4, 5, 6, 7]
    place_block(U, V, W, u_map_2, v_map_2, w_map_2, U_local, V_local, W_local, 7)
    
    # Block 3: A(0:2, 2:4) * B(2:4, 0:2) -> C(0:2, 0:2)
    # A indices: 2,3,6,7
    # B indices: 8,9,12,13
    # C indices: 0,1,2,3
    u_map_3 = [2, 3, 6, 7]
    v_map_3 = [8, 9, 12, 13]
    w_map_3 = [0, 1, 2, 3]
    place_block(U, V, W, u_map_3, v_map_3, w_map_3, U_local, V_local, W_local, 14)
    
    # Block 4: A(0:2, 2:4) * B(2:4, 2:4) -> C(0:2, 2:4)
    # A indices: 2,3,6,7
    # B indices: 10,11,14,15
    # C indices: 4,5,6,7
    u_map_4 = [2, 3, 6, 7]
    v_map_4 = [10, 11, 14, 15]
    w_map_4 = [4, 5, 6, 7]
    place_block(U, V, W, u_map_4, v_map_4, w_map_4, U_local, V_local, W_local, 21)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
