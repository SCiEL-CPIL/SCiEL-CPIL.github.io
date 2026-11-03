import numpy as np

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, val_u=1.0, val_v=1.0, val_w=1.0):
        nonlocal col
        U[i * m + j, col] = val_u
        V[j * p + k, col] = val_v
        W[k * n + i, col] = val_w
        col += 1

    # Strassen decomposition for 2x2 x 2x2
    # A is 2x2, B is 2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    def add_strassen_block(j_offset, k_offset):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # A11: i=0, j=0. A22: i=1, j=1.
        # B11: j=0, k=0. B22: j=1, k=1.
        # Term 1: U(0,0)=1, U(1,1)=1. V(0,0)=1, V(1,1)=1.
        # This is not a single term in the decomposition sum.
        # The decomposition is sum of rank-1 tensors.
        # Strassen expresses C as linear combo of M1..M7.
        # Each M is a product of linear combos of A and B.
        # So each M corresponds to 1 rank-1 term in the tensor decomposition?
        # Yes, if we define U, V, W appropriately.
        
        # M1: U = e_{0,0} + e_{1,1}, V = e_{0,0} + e_{1,1}
        # Wait, U is a vector.
        # U_vec = [1, 0, 0, 1, 0, 0] (indices 0,3 are 1)
        # V_vec = [1, 0, 0, 0, 1, 0] (indices 0,4 are 1? No, V index is j*p+k)
        # j=0, k=0 -> idx 0. j=1, k=1 -> idx 1*2+1=3? No, p is local 2 here?
        # In the block, p=2. But global p=7.
        # We are filling the global U, V, W.
        
        # M1:
        # A part: A[0, j_off] + A[1, j_off+1]
        # B part: B[j_off, k_off] + B[j_off+1, k_off+1]
        # C part: C[0, k_off] + C[1, k_off] + C[0, k_off+1] + C[1, k_off+1]
        
        # We add 7 terms.
        
        # 1. (A00+A11)(B00+B11)
        U[0*m + j_offset, col] = 1
        U[1*m + j_offset + 1, col] = 1
        V[j_offset*p + k_offset, col] = 1
        V[(j_offset+1)*p + k_offset + 1, col] = 1
        W[k_offset*n + 0, col] = 1
        W[(k_offset+1)*n + 0, col] = 1
        W[k_offset*n + 1, col] = 1
        W[(k_offset+1)*n + 1, col] = 1
        col += 1
        
        # 2. (A10+A11)B00
        U[1*m + j_offset, col] = 1
        U[1*m + j_offset + 1, col] = 1
        V[j_offset*p + k_offset, col] = 1
        W[k_offset*n + 1, col] = 1
        col += 1
        
        # 3. A00(B10-B11)
        U[0*m + j_offset, col] = 1
        V[(j_offset+1)*p + k_offset, col] = 1
        V[(j_offset+1)*p + k_offset + 1, col] = -1
        W[k_offset*n + 0, col] = 1
        col += 1
        
        # 4. A11(B01-B00)
        U[1*m + j_offset + 1, col] = 1
        V[j_offset*p + k_offset + 1, col] = 1
        V[j_offset*p + k_offset, col] = -1
        W[(k_offset+1)*n + 1, col] = 1
        col += 1
        
        # 5. (A00+A01)B11
        U[0*m + j_offset, col] = 1
        U[0*m + j_offset + 1, col] = 1
        V[(j_offset+1)*p + k_offset + 1, col] = 1
        W[(k_offset+1)*n + 0, col] = 1
        col += 1
        
        # 6. (A10-A00)(B00+B01)
        U[1*m + j_offset, col] = 1
        U[0*m + j_offset, col] = -1
        V[j_offset*p + k_offset, col] = 1
        V[j_offset*p + k_offset + 1, col] = 1
        W[k_offset*n + 1, col] = 1
        W[(k_offset+1)*n + 1, col] = 1
        col += 1
        
        # 7. (A01-A11)(B10+B11)
        U[0*m + j_offset + 1, col] = 1
        U[1*m + j_offset + 1, col] = -1
        V[(j_offset+1)*p + k_offset, col] = 1
        V[(j_offset+1)*p + k_offset + 1, col] = 1
        W[k_offset*n + 0, col] = 1
        W[(k_offset+1)*n + 0, col] = 1
        col += 1

    # Helper for 2x1 x 1x1 (Outer product part for j=2)
    # A[i, 2] * B[2, k]
    def add_outer_term(i, k):
        nonlocal col
        U[i*m + 2, col] = 1
        V[2*p + k, col] = 1
        W[k*n + i, col] = 1
        col += 1

    # Process 3 pairs of columns
    for pair_idx in range(3):
        j_off = 0
        k_off = pair_idx * 2
        
        # Strassen for j in {0,1}
        add_strassen_block(j_off, k_off)
        
        # Outer product for j=2
        for i in range(2):
            for k_rel in range(2):
                add_outer_term(i, k_off + k_rel)

    # Process last column k=6
    j_off = 0
    k_off = 6
    
    # Schoolbook for j in {0,1}
    for i in range(2):
        for j in range(2):
            add_term(i, j, k_off)
            
    # Schoolbook for j=2
    for i in range(2):
        add_term(i, 2, k_off)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
