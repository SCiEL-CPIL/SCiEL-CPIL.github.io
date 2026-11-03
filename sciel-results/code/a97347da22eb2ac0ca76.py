import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5  # A is n x m, B is m x p, C is n x p
    
    # We'll build U, V, W with rank R
    # U: (n*m, R) = (12, R)
    # V: (m*p, R) = (20, R)  
    # W: (p*n, R) = (15, R)
    
    # Strategy: Use Strassen for 2x2x2 blocks where possible
    # Split j (middle index, size 4) into J1={0,1} and J2={2,3}
    # For each j-group, split i into I1={0,1} and I2={2}
    # For I1 x J_group x k, split k into groups of 2 (plus remainder)
    
    # Let's build the decomposition term by term
    # Each term r has U[r], V[r], W[r] vectors
    
    terms = []  # list of (U_vec, V_vec, W_vec)
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a rank-1 term. u_vals: dict {(i,j): val}, v_vals: dict {(j,k): val}, w_vals: dict {(k,i): val}"""
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        for (i, j), val in u_vals.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_vec[k * n + i] = val
        terms.append((u_vec, v_vec, w_vec))
    
    # Strassen 2x2x2 decomposition
    # For A (2x2) * B (2x2) = C (2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our tensor formulation, each "M" term contributes to specific (i,j,k) triples
    # Actually, let me think about this differently.
    
    # The tensor T[i,j,k] = 1 for all i,j,k in the multiplication.
    # A rank-1 term (u, v, w) contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k].
    # We need sum of all terms = 1 for each (i,j,k).
    
    # For Strassen on a 2x2x2 block with indices i∈{ia,ia+1}, j∈{ja,ja+1}, k∈{ka,ka+1}:
    # The 8 schoolbook terms would be:
    # For each (i,j,k) in the 2x2x2 block, one term with u[i,j]=1, v[j,k]=1, w[k,i]=1
    
    # Strassen replaces these 8 terms with 7 terms where the products u[i,j]*v[j,k]*w[k,i]
    # sum to 1 for each (i,j,k) in the block.
    
    # Let me define Strassen terms for a generic 2x2x2 block at position (i0, j0, k0)
    # where i0 is the first row, j0 is the first middle index, k0 is the first col
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for the 2x2x2 block starting at (i0,j0,k0)"""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        # Contributes to C[i0,k0], C[i1,k1] (and cross terms via the structure)
        # Actually, in the tensor formulation, each term is a product of three vectors.
        # Let me define the 7 terms explicitly.
        
        # Term 1: M1
        u1 = {(i0,j0): 1, (i1,j1): 1}
        v1 = {(j0,k0): 1, (j1,k1): 1}
        w1 = {(k0,i0): 1, (k1,i1): 1}
        add_term(u1, v1, w1)
        
        # Term 2: M2 = (A[i1,j0]+A[i1,j1])B[j0,k0]
        u2 = {(i1,j0): 1, (i1,j1): 1}
        v2 = {(j0,k0): 1}
        w2 = {(k0,i0): 1, (k0,i1): 1}
        add_term(u2, v2, w2)
        
        # Term 3: M3 = A[i0,j0](B[j1,k0]-B[j1,k1])
        u3 = {(i0,j0): 1}
        v3 = {(j1,k0): 1, (j1,k1): -1}
        w3 = {(k0,i0): 1, (k1,i0): 1}
        add_term(u3, v3, w3)
        
        # Term 4: M4 = A[i1,j1](B[j0,k1]-B[j0,k0])
        u4 = {(i1,j1): 1}
        v4 = {(j0,k1): 1, (j0,k0): -1}
        w4 = {(k1,i1): 1, (k0,i1): 1}
        add_term(u4, v4, w4)
        
        # Term 5: M5 = (A[i0,j0]+A[i0,j1])B[j1,k1]
        u5 = {(i0,j0): 1, (i0,j1): 1}
        v5 = {(j1,k1): 1}
        w5 = {(k1,i0): 1, (k1,i1): 1}
        add_term(u5, v5, w5)
        
        # Term 6: M6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j1,k0])
        u6 = {(i1,j0): 1, (i0,j0): -1}
        v6 = {(j0,k0): 1, (j1,k0): 1}
        w6 = {(k0,i0): 1, (k0,i1): 1}
        add_term(u6, v6, w6)
        
        # Term 7: M7 = (A[i0,j1]-A[i1,j1])(B[j1,k1]+B[j0,k1])
        u7 = {(i0,j1): 1, (i1,j1): -1}
        v7 = {(j1,k1): 1, (j0,k1): 1}
        w7 = {(k1,i0): 1, (k1,i1): 1}
        add_term(u7, v7, w7)
    
    # Now let's build the full decomposition
    
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1} -> Strassen
    add_strassen_2x2x2(0, 0, 0)
    # Block 2: i∈{0,1}, j∈{0,1}, k∈{2,3} -> Strassen
    add_strassen_2x2x2(0, 0, 2)
    # Block 3: i∈{0,1}, j∈{0,1}, k∈{4,4} -> This is 2x2x1, use schoolbook (2 terms)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j): 1}, {(j,4): 1}, {(4,i): 1})
    
    # Block 4: i∈{0,1}, j∈{2,3}, k∈{0,1} -> Strassen
    add_strassen_2x2x2(0, 2, 0)
    # Block 5: i∈{0,1}, j∈{2,3}, k∈{2,3} -> Strassen
    add_strassen_2x2x2(0, 2, 2)
    # Block 6: i∈{0,1}, j∈{2,3}, k∈{4,4} -> schoolbook (2 terms)
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j): 1}, {(j,4): 1}, {(4,i): 1})
    
    # Block 7: i∈{2}, j∈{0,1}, k∈{0,1,2,3,4} -> schoolbook (2*5=10 terms)
    for j in [0, 1]:
        for k in [0, 1, 2, 3, 4]:
            add_term({(2,j): 1}, {(j,k): 1}, {(k,2): 1})
    
    # Block 8: i∈{2}, j∈{2,3}, k∈{0,1,2,3,4} -> schoolbook (2*5=10 terms)
    for j in [2, 3]:
        for k in [0, 1, 2, 3, 4]:
            add_term({(2,j): 1}, {(j,k): 1}, {(k,2): 1})
    
    R = len(terms)
    
    # Build U, V, W matrices
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    for r, (u_vec, v_vec, w_vec) in enumerate(terms):
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[idx][r] = val
    
    # Verify
    T = np.zeros((n, m, p))
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        R = n * m * p
        U = [[0.0] * R for _ in range(n * m)]
        V = [[0.0] * R for _ in range(m * p)]
        W = [[0.0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
