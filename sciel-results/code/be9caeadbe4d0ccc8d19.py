"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Define the set of all triplets
    all_triplets = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_triplets.add((i, j, k))
                
    available = set(all_triplets)
    strassen_blocks = []
    
    # Generate all possible 2x2x2 boxes
    # I: 2 rows, J: 2 cols, K: 2 inner
    possible_boxes = []
    for i1 in range(n):
        for i2 in range(i1+1, n):
            for j1 in range(m):
                for j2 in range(j1+1, m):
                    for k1 in range(p):
                        for k2 in range(k1+1, p):
                            box = ((i1, i2), (j1, j2), (k1, k2))
                            possible_boxes.append(box)
                            
    # Greedy selection of disjoint boxes
    # Sort boxes to prefer those with lower indices? Or random?
    # Let's just iterate through them.
    for box in possible_boxes:
        if len(strassen_blocks) >= 10: # Cap to avoid too much overhead
            break
        if time.time() > deadline - 10:
            break
            
        (i1, i2), (j1, j2), (k1, k2) = box
        triplets = [
            (i1, j1, k1), (i1, j1, k2), (i1, j2, k1), (i1, j2, k2),
            (i2, j1, k1), (i2, j1, k2), (i2, j2, k1), (i2, j2, k2)
        ]
        
        # Check if all are available
        if all(t in available for t in triplets):
            strassen_blocks.append(box)
            for t in triplets:
                available.remove(t)
                
    # Construct the decomposition
    # We will build U, V, W lists
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 1. Add Strassen blocks
    for box in strassen_blocks:
        (i1, i2), (j1, j2), (k1, k2) = box
        
        # Strassen 2x2x2 decomposition
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the global indices.
        # The block contributes to C[i, k] for i in {i1, i2}, k in {k1, k2}
        # Using A[i, j] and B[j, k] for j in {j1, j2}
        
        # Let's define the 7 terms for this block.
        # Each term is a linear combination of A and B entries.
        # We add columns to U, V, W.
        
        # Term 1: (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # This is not quite right. Strassen multiplies matrices.
        # The terms are products of linear forms in A and B.
        
        # Linear forms in A (size 2x2):
        # L1 = A[i1,j1] + A[i2,j2]  (a+d)
        # L2 = A[i2,j1] + A[i2,j2]  (c+d)
        # L3 = A[i1,j1]              (a)
        # L4 = A[i2,j2]              (d)
        # L5 = A[i1,j1] + A[i1,j2]  (a+b)
        # L6 = A[i2,j1] - A[i1,j1]  (c-a)
        # L7 = A[i1,j2] - A[i2,j2]  (b-d)
        
        # Linear forms in B (size 2x2):
        # R1 = B[j1,k1] + B[j2,k2]  (e+h)
        # R2 = B[j1,k1]              (e)
        # R3 = B[j1,k2] - B[j2,k2]  (f-h)
        # R4 = B[j2,k1] - B[j1,k1]  (g-e)
        # R5 = B[j2,k2]              (h)
        # R6 = B[j1,k1] + B[j2,k1]  (e+g)
        # R7 = B[j1,k2] + B[j2,k2]  (f+h)
        
        # Products:
        # P1 = L1 * R1
        # P2 = L2 * R2
        # P3 = L3 * R3
        # P4 = L4 * R4
        # P5 = L5 * R5
        # P6 = L6 * R6
        # P7 = L7 * R7
        
        # Contributions to C:
        # C[i1,k1] += P1 + P4 - P5 + P7
        # C[i1,k2] += P3 + P5
        # C[i2,k1] += P2 + P4
        # C[i2,k2] += P1 - P2 + P3 + P6
        
        # We need to create 7 columns for U, V, W.
        # For each product Ps, we create a column.
        # U column will have the coefficients of the linear form in A.
        # V column will have the coefficients of the linear form in B.
        # W column will have the coefficients for the output C.
        
        for s in range(7):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            if s == 0: # P1
                u_col[i1 * m + j1] = 1
                u_col[i2 * m + j2] = 1
                v_col[j1 * p + k1] = 1
                v_col[j2 * p + k2] = 1
                w_col[k1 * n + i1] = 1
                w_col[k2 * n + i2] = 1 # Wait, W is (p*n, rank). Index is k*n+i.
                # C[i1,k1] gets +1, C[i2,k2] gets +1?
                # P1 contributes to C[i1,k1] and C[i2,k2].
                # Yes.
                
            elif s == 1: # P2
                u_col[i2 * m + j1] = 1
                u_col[i2 * m + j2] = 1
                v_col[j1 * p + k1] = 1
                w_col[k1 * n + i2] = 1
                
            elif s == 2: # P3
                u_col[i1 * m + j1] = 1
                v_col[j1 * p + k2] = 1
                v_col[j2 * p + k2] = -1
                w_col[k2 * n + i1] = 1
                
            elif s == 3: # P4
                u_col[i2 * m + j2] = 1
                v_col[j2 * p + k1] = 1
                v_col[j1 * p + k1] = -1
                w_col[k1 * n + i1] = 1
                w_col[k1 * n + i2] = 1
                
            elif s == 4: # P5
                u_col[i1 * m + j1] = 1
                u_col[i1 * m + j2] = 1
                v_col[j2 * p + k2] = 1
                w_col[k1 * n + i1] = -1
                w_col[k2 * n + i1] = 1
                
            elif s == 5: # P6
                u_col[i2 * m + j1] = 1
                u_col[i1 * m + j1] = -1
                v_col[j1 * p + k1] = 1
                v_col[j2 * p + k1] = 1
                w_col[k2 * n + i2] = 1
                
            elif s == 6: # P7
                u_col[i1 * m + j2] = 1
                u_col[i2 * m + j2] = -1
                v_col[j1 * p + k2] = 1
                v_col[j2 * p + k2] = 1
                w_col[k1 * n + i1] = 1
                w_col[k2 * n + i2] = 1 # Wait, P7 contributes to C[i1,k1] and C[i2,k2]?
                # C11 = M1 + M4 - M5 + M7. Yes.
                # C22 = M1 - M2 + M3 + M6. M7 is NOT in C22.
                # My previous formula: C22 = M1 - M2 + M3 + M6.
                # So P7 only contributes to C[i1,k1].
                # Let's re-verify Strassen formulas.
                # M7 = (b-d)(f+h).
                # C11 = M1 + M4 - M5 + M7.
                # C12 = M3 + M5.
                # C21 = M2 + M4.
                # C22 = M1 - M2 + M3 + M6.
                # So M7 is only in C11.
                # Correction: w_col[k2 * n + i2] should be 0.
                w_col[k2 * n + i2] = 0
                # And w_col[k1 * n + i1] = 1.
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    # 2. Add schoolbook terms for remaining triplets
    for t in available:
        i, j, k = t
        u_col = np.zeros(n * m)
        v_col = np.zeros(m * p)
        w_col = np.zeros(p * n)
        
        u_col[i * m + j] = 1
        v_col[j * p + k] = 1
        w_col[k * n + i] = 1
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    # Convert to matrices
    if U_cols:
        U_final = np.column_stack(U_cols)
        V_final = np.column_stack(V_cols)
        W_final = np.column_stack(W_cols)
    else:
        U_final = np.zeros((n*m, 0))
        V_final = np.zeros((m*p, 0))
        W_final = np.zeros((p*n, 0))
        
    best = {
        "U": U_final.tolist(), 
        "V": V_final.tolist(), 
        "W": W_final.tolist()
    }
    
    return best
# EVOLVE-BLOCK-END
