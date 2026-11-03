import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition
    # 1. 3x3x3 part (cols 0-2 of B) -> Rank 23
    # 2. 3x3x2 part (cols 3-4 of B) -> Rank 6
    # 3. 3x3x2 part (cols 5-6 of B) -> Rank 6
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    # U row index: i*m + j (i in 0..n-1, j in 0..m-1)
    # V row index: j*p + k (j in 0..m-1, k in 0..p-1)
    # W row index: k*n + i (k in 0..p-1, i in 0..n-1)
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        U.append(u_idx)
        V.append(v_idx)
        W.append(w_idx)
        
    # We will build the matrices as lists of rows, then convert to numpy
    # Or just build numpy arrays directly with dynamic size?
    # Better to count rank first or use lists and pad.
    
    terms = []
    
    # --- Part 1: 3x3x3 (k=0,1,2) ---
    # A is 3x3, B is 3x3 (cols 0-2)
    # Block split: S2={0,1}, S1={2}
    
    # 1. A_22 B_22 (2x2 * 2x2) -> 4 terms
    # A_22 is A[0:2, 0:2], B_22 is B[0:2, 0:2]
    # Terms: A[i,k] B[k,j] for i,j in 0,1 and k in 0,1
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                # U: i*m + k (A[i,k])
                # V: k*p + j (B[k,j]) -- Note: B is 3x7, so p=7. Here j is col index in 0..2.
                # W: j*n + i (C[i,j])
                terms.append((i*m + k, k*p + j, j*n + i))
                
    # 2. A_21 B_12 (2x1 * 1x2) -> 4 terms
    # A_21 is A[0:2, 2], B_12 is B[2, 0:2]
    # Result adds to C[0:2, 0:2]
    for i in [0, 1]:
        for j in [0, 1]:
            # Term: A[i, 2] * B[2, j]
            # U: i*m + 2
            # V: 2*p + j
            # W: j*n + i
            terms.append((i*m + 2, 2*p + j, j*n + i))
            
    # 3. A_22 B_21 (2x2 * 2x1) -> 4 terms
    # A_22 is A[0:2, 0:2], B_21 is B[0:2, 2]
    # Result adds to C[0:2, 2]
    for i in [0, 1]:
        for k in [0, 1]:
            # Term: A[i, k] * B[k, 2]
            # U: i*m + k
            # V: k*p + 2
            # W: 2*n + i
            terms.append((i*m + k, k*p + 2, 2*n + i))
            
    # 4. A_21 B_11 (2x1 * 1x1) -> 2 terms
    # A_21 is A[0:2, 2], B_11 is B[2, 2]
    # Result adds to C[0:2, 2]
    for i in [0, 1]:
        # Term: A[i, 2] * B[2, 2]
        terms.append((i*m + 2, 2*p + 2, 2*n + i))
        
    # 5. A_12 B_22 (1x2 * 2x2) -> 4 terms
    # A_12 is A[2, 0:2], B_22 is B[0:2, 0:2]
    # Result adds to C[2, 0:2]
    for j in [0, 1]:
        for k in [0, 1]:
            # Term: A[2, k] * B[k, j]
            terms.append((2*m + k, k*p + j, j*n + 2))
            
    # 6. A_11 B_12 (1x1 * 1x2) -> 2 terms
    # A_11 is A[2, 2], B_12 is B[2, 0:2]
    # Result adds to C[2, 0:2]
    for j in [0, 1]:
        # Term: A[2, 2] * B[2, j]
        terms.append((2*m + 2, 2*p + j, j*n + 2))
        
    # 7. A_12 B_21 (1x2 * 2x1) -> 2 terms
    # A_12 is A[2, 0:2], B_21 is B[0:2, 2]
    # Result adds to C[2, 2]
    for k in [0, 1]:
        # Term: A[2, k] * B[k, 2]
        terms.append((2*m + k, k*p + 2, 2*n + 2))
        
    # 8. A_11 B_11 (1x1 * 1x1) -> 1 term
    # Term: A[2, 2] * B[2, 2]
    terms.append((2*m + 2, 2*p + 2, 2*n + 2))
    
    # --- Part 2: 3x3x2 (k=0,1,2, cols 3-4 of B) ---
    # A is 3x3, B is 3x2 (cols 3,4)
    # Rank 6: Schoolbook for 2 cols
    for j in [3, 4]:
        for i in [0, 1, 2]:
            for k in [0, 1, 2]:
                terms.append((i*m + k, k*p + j, j*n + i))
                
    # --- Part 3: 3x3x2 (k=0,1,2, cols 5-6 of B) ---
    # A is 3x3, B is 3x2 (cols 5,6)
    # Rank 6: Schoolbook for 2 cols
    for j in [5, 6]:
        for i in [0, 1, 2]:
            for k in [0, 1, 2]:
                terms.append((i*m + k, k*p + j, j*n + i))
                
    rank = len(terms)
    
    U_mat = np.zeros((n*m, rank))
    V_mat = np.zeros((m*p, rank))
    W_mat = np.zeros((p*n, rank))
    
    for col, (u_idx, v_idx, w_idx) in enumerate(terms):
        U_mat[u_idx, col] = 1.0
        V_mat[v_idx, col] = 1.0
        W_mat[w_idx, col] = 1.0
        
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
