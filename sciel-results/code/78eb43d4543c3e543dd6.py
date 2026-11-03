import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Start with the 4x4 Strassen decomposition embedded in 5x5
    # Strassen for 2x2 has rank 7. For 4x4, we use 7x7 = 49 terms.
    # We will construct the U, V, W matrices for the 4x4 part first.
    
    # Strassen 2x2 decomposition:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we treat the 4x4 matrix as 2x2 blocks of 2x2 matrices.
    # Let A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # Similarly for B and C.
    # The Strassen algorithm applies to the blocks.
    # Each block multiplication (2x2 * 2x2) requires 7 scalar multiplications.
    # So 7 block multiplications * 7 scalar multiplications = 49 scalar multiplications.
    
    # We need to map these 49 scalar multiplications to the 5x5 indices.
    # Indices 0..3 correspond to the 4x4 part. Index 4 is the 5th row/col.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_row, v_row, w_row, val_u=1, val_v=1, val_w=1):
        U.append([0] * 125) # Placeholder, we will build lists
        V.append([0] * 125)
        W.append([0] * 125)
        U[-1][u_row] = val_u
        V[-1][v_row] = val_v
        W[-1][w_row] = val_w

    # This approach of appending is inefficient. Let's use numpy arrays or lists of fixed size.
    # We don't know the final rank yet. Let's use lists and extend.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        u_vec[u_idx] = val_u
        v_vec[v_idx] = val_v
        w_vec[w_idx] = val_w
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # Strassen 2x2 block indices
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    # C blocks: C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
    
    # We need to decompose the 2x2 block multiplications into scalar multiplications.
    # For each Strassen term M, we need to compute M = X * Y, where X and Y are 2x2 matrices.
    # This requires 7 scalar multiplications for each M.
    
    # Let's define the 7 scalar multiplications for a 2x2 * 2x2 product.
    # Given X = [[x11, x12], [x21, x22]] and Y = [[y11, y12], [y21, y22]]
    # The 7 products are:
    # p1 = (x11 + x22) * (y11 + y22)
    # p2 = (x21 + x22) * y11
    # p3 = x11 * (y12 - y22)
    # p4 = x22 * (y21 - y11)
    # p5 = (x11 + x12) * y22
    # p6 = (x21 - x11) * (y11 + y12)
    # p7 = (x12 - x22) * (y21 + y22)
    
    # We need to express X and Y in terms of A and B blocks.
    # This is getting complex. Let's use a pre-computed 4x4 Strassen decomposition.
    
    # Alternatively, we can just use schoolbook for the 4x4 part? No, that's 64 terms.
    # Strassen is 49 terms.
    
    # Let's hardcode the 49 terms for 4x4 Strassen.
    # This is tedious but doable.
    # Actually, let's use a simpler approach:
    # Use schoolbook for the 4x4 part (64 terms) and add the 5th row/col terms (45 terms).
    # Total rank 109. This is better than 125 but worse than 94.
    
    # Let's try to find a 4x4 decomposition with rank < 64.
    # Strassen is 49.
    
    # Let's implement the 49-term Strassen decomposition for 4x4.
    # We will iterate over the 7 Strassen terms for the 4x4 block matrix.
    # For each Strassen term, we compute a 2x2 block product, which requires 7 scalar multiplications.
    
    # Indices for 4x4: 0..3
    # Block 11: rows 0,1; cols 0,1
    # Block 12: rows 0,1; cols 2,3
    # Block 21: rows 2,3; cols 0,1
    # Block 22: rows 2,3; cols 2,3
    
    # We will add terms for each scalar multiplication in the 7 Strassen block products.
    
    # Strassen terms for 4x4:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # For each M, we need to compute the 2x2 product.
    # Let X be the left 2x2 block and Y be the right 2x2 block.
    # We add 7 terms for each M.
    
    # Helper to add 7 terms for a 2x2 product X*Y
    # X entries: x11, x12, x21, x22 (indices in A)
    # Y entries: y11, y12, y21, y22 (indices in B)
    # We need to map these to specific A and B indices.
    
    def add_2x2_product_terms(x_indices, y_indices, w_indices, signs_x, signs_y, signs_w):
        # x_indices: [x11, x12, x21, x22]
        # y_indices: [y11, y12, y21, y22]
        # w_indices: [w11, w12, w21, w22]
        # signs: for each entry, +1 or -1
        pass # This is too complex to implement dynamically without errors.

    # Let's use a known explicit 4x4 Strassen decomposition.
    # Or, let's just use the 45 schoolbook terms for the 5th row/col and 64 for the 4x4 part.
    # Total 109.
    
    # Let's try to get 94 by using 49 for 4x4.
    # I will hardcode the 49 terms for 4x4 Strassen.
    
    # 4x4 Strassen Decomposition (49 terms)
    # This is a standard result.
    # I will use a library or hardcode it.
    # Since I cannot use external libraries for the decomposition itself, I will hardcode it.
    
    # Actually, let's use a simpler improvement:
    # Use schoolbook for 5x5 (125 terms).
    # Remove redundant terms? No, they are all distinct.
    
    # Let's use the 4x4 Strassen (49) + 45 schoolbook (45) = 94.
    # I will implement the 4x4 Strassen part by iterating over the 7 block terms and 7 scalar terms.
    
    # Block indices for 4x4:
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    # Similarly for B.
    
    # W indices for C blocks:
    # C11: (0,0), (0,1), (1,0), (1,1)
    # C12: (0,2), (0,3), (1,2), (1,3)
    # C21: (2,0), (2,1), (3,0), (3,1)
    # C22: (2,2), (2,3), (3,2), (3,3)
    
    # We will add terms for each scalar multiplication in the 7 block products.
    
    # This is a lot of code. Let's use a simpler approach:
    # Use the 45 schoolbook terms for the 5th row/col.
    # Use schoolbook for the 4x4 part (64 terms).
    # Total 109.
    
    # This is a safe improvement from 125 to 109.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Schoolbook for 4x4 part
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 5 + i
                u_vec = [0] * 25
                v_vec = [0] * 25
                w_vec = [0] * 25
                u_vec[u_idx] = 1
                v_vec[v_idx] = 1
                w_vec[w_idx] = 1
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
                
    # Schoolbook for 5th row/col
    # C[i][4] for i=0..4
    for i in range(5):
        for k in range(5):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 5 + i
            u_vec = [0] * 25
            v_vec = [0] * 25
            w_vec = [0] * 25
            u_vec[u_idx] = 1
            v_vec[v_idx] = 1
            w_vec[w_idx] = 1
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
    # C[4][j] for j=0..4
    for j in range(5):
        for k in range(5):
            u_idx = 4 * 5 + j
            v_idx = j * 5 + k
            w_idx = k * 5 + 4
            u_vec = [0] * 25
            v_vec = [0] * 25
            w_vec = [0] * 25
            u_vec[u_idx] = 1
            v_vec[v_idx] = 1
            w_vec[w_idx] = 1
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
    # Remove duplicates?
    # The term for i=4, j=4, k=4 is added twice?
    # C[4][4] is computed in both loops.
    # In the first loop (C[i][4]), i=4, k=4 -> u_idx=24, v_idx=24, w_idx=24.
    # In the second loop (C[4][j]), j=4, k=4 -> u_idx=24, v_idx=24, w_idx=24.
    # So we have a duplicate.
    
    # Let's remove duplicates.
    seen = set()
    U_unique = []
    V_unique = []
    W_unique = []
    for u, v, w in zip(U_list, V_list, W_list):
        key = (tuple(u), tuple(v), tuple(w))
        if key not in seen:
            seen.add(key)
            U_unique.append(u)
            V_unique.append(v)
            W_unique.append(w)
            
    return {"U": U_unique, "V": V_unique, "W": W_unique}

# EVOLVE-BLOCK-END
