import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We expect rank 28
    rank = 28
    
    # Initialize U, V, W as lists of lists or numpy arrays?
    # The seed uses lists of lists. The return type is dict of lists.
    # But numpy might be easier for construction, then convert.
    # However, the seed returns lists. Let's stick to lists or convert at the end.
    # The prompt says "integer/half-integer entries".
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        # This is not quite right because a Strassen term involves multiple non-zeros in U, V, W.
        # We need a way to add a "pattern" to the current column.
        pass

    # Better approach:
    # Define the 7 Strassen patterns for local 2x2x2.
    # Each pattern is a list of (u_indices, u_values), (v_indices, v_values), (w_indices, w_values).
    
    # Local indices 0, 1
    # U indices: 0, 1 (for i=0,1 and j=0,1? No, local j is 0,1)
    # In the block, i_local \in {0,1}, j_local \in {0,1}, k_local \in {0,1}.
    
    strassen_patterns = [
        {
            'U': [(0, 1), (3, 1)], # i=0,j=0 -> idx 0; i=1,j=1 -> idx 3 (if m=2? No, m is global)
            # Wait, U index is i*m + j.
            # Inside the block, we are fixing the block position.
            # Let's define a helper that takes global offsets.
        }
    ]
    
    # Let's just hardcode the additions for each block type.
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # 7 terms
        # Term 1
        U[i_off + 0][col] += 1
        U[i_off + 1][col] += 1 # i=1, j=1? No.
        # Let's map local (i,j) to global index in U.
        # U index = i_global * m + j_global.
        # i_global = i_off + i_local
        # j_global = j_off + j_local
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00 (i=0,j=0), A11 (i=1,j=1)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        # V: B00 (j=0,k=0), B11 (j=1,k=1)
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        # W: C00 (k=0,i=0), C11 (k=1,i=1)
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1 # C10
        col += 1
        
        # Term 3: A00(B01-B11)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] -= 1
        W[(k_off+1)*n + (i_off+0)][col] += 1 # C01
        col += 1
        
        # Term 4: A11(B10-B00)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] -= 1
        W[(k_off+0)*n + (i_off+1)][col] += 1 # C10
        W[(k_off+0)*n + (i_off+0)][col] -= 1 # C00
        col += 1
        
        # Term 5: (A00+A01)B11
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+0)][col] += 1 # C01
        W[(k_off+0)*n + (i_off+0)][col] -= 1 # C00
        col += 1
        
        # Term 6: (A01-A10)(B00+B01)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+0)][col] -= 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1 # C11
        col += 1
        
        # Term 7: (A10-A00)(B11+B01)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] -= 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1 # C00
        W[(k_off+1)*n + (i_off+1)][col] -= 1 # C11
        col += 1

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m + j][col] += 1
        V[j*p + k][col] += 1
        W[k*n + i][col] += 1
        col += 1

    # 1. Strassen block for j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    
    # 2. Strassen block for j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2)
    
    # 3. Schoolbook for j in {0,1}, k=4
    for i in range(2):
        for j in range(2):
            add_schoolbook(i, j, 4)
            
    # 4. Schoolbook for j=2, k in {0,1,2,3,4}
    for i in range(2):
        for k in range(5):
            add_schoolbook(i, 2, k)
            
    # Convert to integers if they are whole numbers?
    # The prompt allows integer/half-integer.
    # The values are integers.
    
    return {"U": U, "V": V, "W": W}
