import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Define 2x2 Strassen decomposition
    # U, V, W are lists of dicts or arrays.
    # Let's use lists of (U_dict, V_dict, W_dict)
    # U_dict: {(i,j): val}
    # V_dict: {(j,k): val}
    # W_dict: {(k,i): val}
    
    strassen_2x2 = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Function to shift indices
    def shift_indices(d, di, dj):
        return {(i+di, j+dj): v for (i,j), v in d.items()}
        
    def scale_indices(d, s):
        return {(i,j): v*s for (i,j), v in d.items()}
        
    # Step 2: Compose 2x2 * 2x4
    # A is 2x2, B is 2x4. B = [B1, B2] where B1, B2 are 2x2.
    # C = A B1 + A B2.
    # Decomp for A B1:
    # U from strassen, V from strassen shifted in B's columns? No, B1 is B[0:2, 0:2].
    # So V indices are (j, k) where k is 0,1.
    # W indices are (k, i) where k is 0,1.
    
    decomp_2x2_2x4 = []
    for U, V, W in strassen_2x2:
        # Part 1: A * B1
        # U is 2x2, V is 2x2 (cols 0-1), W is 2x2 (cols 0-1)
        decomp_2x2_2x4.append((U, V, W))
        
        # Part 2: A * B2
        # U is 2x2, V is 2x2 (cols 2-3), W is 2x2 (cols 2-3)
        V_shifted = shift_indices(V, 0, 2)
        W_shifted = shift_indices(W, 2, 0) # W is (k,i), k shifts by 2
        decomp_2x2_2x4.append((U, V_shifted, W_shifted))
        
    # Rank is 14.
    
    # Step 3: Compose 2x4 * 4x4
    # A is 2x4, B is 4x4.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x4.
    # C = A1 B1 + A2 B2.
    # We need decomp for 2x2 * 2x4. We have it (decomp_2x2_2x4).
    
    decomp_2x4_4x4 = []
    for U, V, W in decomp_2x2_2x4:
        # Part 1: A1 * B1
        # A1 is A[0:2, 0:2]. B1 is B[0:2, 0:4].
        # U is 2x2 (rows 0-1, cols 0-1).
        # V is 2x4 (rows 0-1, cols 0-3).
        # W is 4x2 (rows 0-3, cols 0-1).
        # In decomp_2x2_2x4, U is 2x2, V is 2x4, W is 4x2.
        # Wait, in decomp_2x2_2x4, the "B" was 2x4. So V was 2x4.
        # So this matches directly.
        decomp_2x4_4x4.append((U, V, W))
        
        # Part 2: A2 * B2
        # A2 is A[0:2, 2:4]. So U indices shift by (0, 2).
        # B2 is B[2:4, 0:4]. So V indices shift by (2, 0).
        # W is 4x2. The output C is 2x4.
        # In A1 B1, W was (k,i) with k in 0..3, i in 0..1.
        # In A2 B2, the product is still added to the same C.
        # So W indices do NOT shift.
        U_shifted = shift_indices(U, 0, 2)
        V_shifted = shift_indices(V, 2, 0)
        decomp_2x4_4x4.append((U_shifted, V_shifted, W))
        
    # Rank is 28.
    
    # Step 4: Compose 2x4 * 4x8
    # A is 2x4, B is 4x8.
    # B = [B1, B2] where B1, B2 are 4x4.
    # C = A B1 + A B2.
    # We need decomp for 2x4 * 4x4. We have it (decomp_2x4_4x4).
    
    decomp_2x4_4x8 = []
    for U, V, W in decomp_2x4_4x4:
        # Part 1: A * B1
        # B1 is B[0:4, 0:4].
        # V is 4x4 (rows 0-3, cols 0-3).
        # W is 8x2 (rows 0-7, cols 0-1).
        # In decomp_2x4_4x4, V was 4x4, W was 4x2?
        # No, in 2x4*4x4, C is 2x4. So W is 4x2.
        # Here B is 4x8. C is 2x8. So W is 8x2.
        # So for B1 (cols 0-3), W indices are (k,i) with k in 0..3.
        decomp_2x4_4x8.append((U, V, W))
        
        # Part 2: A * B2
        # B2 is B[0:4, 4:8].
        # V indices shift by (0, 4).
        # W indices shift by (4, 0).
        V_shifted = shift_indices(V, 0, 4)
        W_shifted = shift_indices(W, 4, 0)
        decomp_2x4_4x8.append((U, V_shifted, W_shifted))
        
    # Rank is 56.
    
    # Convert to matrices
    rank = len(decomp_2x4_4x8)
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r, (U, V, W) in enumerate(decomp_2x4_4x8):
        for (i,j), val in U.items():
            U_mat[i*m + j][r] = val
        for (j,k), val in V.items():
            V_mat[j*p + k][r] = val
        for (k,i), val in W.items():
            W_mat[k*n + i][r] = val
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
