import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # We will build the decomposition block by block.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen 2x2x2 decomposition
    # A indices: (0,0), (0,1), (1,0), (1,1) -> local (i,j)
    # B indices: (0,0), (0,1), (1,0), (1,1) -> local (j,k)
    # C indices: (0,0), (0,1), (1,0), (1,1) -> local (i,k)
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to define the coefficients for U, V, W for each of the 7 terms.
    # U is for A, V is for B, W is for C.
    
    # Term 1: M1
    # A: a(0,0), d(1,1)
    # B: e(0,0), h(1,1)
    # C: (0,0) coef 1, (1,1) coef 1
    # We need to map local (i,j) to global U index.
    # Local (i,j) in 2x2 block corresponds to specific global indices.
    
    # Let's iterate through the blocks.
    
    # j in {0,1,2,3}
    # Split j into j_block 0 (j=0,1) and j_block 1 (j=2,3)
    
    for jb in range(2):
        j_offset = jb * 2
        # j_local in {0,1} maps to j_global = j_offset + j_local
        
        # k in {0..5}
        # Split k into 3 blocks of 2: k_block 0 (k=0,1), k_block 1 (k=2,3), k_block 2 (k=4,5)
        
        for kb in range(3):
            k_offset = kb * 2
            # k_local in {0,1} maps to k_global = k_offset + k_local
            
            # i in {0,1,2}
            # Split i into i_block 0 (i=0,1) and i_block 1 (i=2)
            
            # Block i_block 0: 2x2x2 Strassen
            # i_local in {0,1} maps to i_global = i_local (since offset 0)
            
            # Strassen terms
            # M1
            # U: (0,0), (1,1) -> global (0, j_offset+0), (1, j_offset+1)
            # V: (0,0), (1,1) -> global (j_offset+0, k_offset+0), (j_offset+1, k_offset+1)
            # W: (0,0), (1,1) -> global (k_offset+0, 0), (k_offset+1, 1)
            
            # We need to add 7 terms for this 2x2x2 block.
            
            # Term 1: M1
            # U[0*m + (j_offset+0)] += 1
            # U[1*m + (j_offset+1)] += 1
            # V[(j_offset+0)*p + (k_offset+0)] += 1
            # V[(j_offset+1)*p + (k_offset+1)] += 1
            # W[(k_offset+0)*n + 0] += 1
            # W[(k_offset+1)*n + 1] += 1
            
            # But we are building columns. Each column is a rank-1 term.
            # So for M1, we create a new column.
            
            # M1
            U[0*m + (j_offset+0)][col] = 1
            U[1*m + (j_offset+1)][col] = 1
            V[(j_offset+0)*p + (k_offset+0)][col] = 1
            V[(j_offset+1)*p + (k_offset+1)][col] = 1
            W[(k_offset+0)*n + 0][col] = 1
            W[(k_offset+1)*n + 1][col] = 1
            col += 1
            
            # M2: (c+d)e
            # A: c(1,0), d(1,1)
            # B: e(0,0)
            # C: (1,0) coef 1
            U[1*m + (j_offset+0)][col] = 1
            U[1*m + (j_offset+1)][col] = 1
            V[(j_offset+0)*p + (k_offset+0)][col] = 1
            W[(k_offset+0)*n + 1][col] = 1
            col += 1
            
            # M3: a(f-h)
            # A: a(0,0)
            # B: f(0,1), h(1,1) with -1
            # C: (0,1) coef 1
            U[0*m + (j_offset+0)][col] = 1
            V[(j_offset+0)*p + (k_offset+1)][col] = 1
            V[(j_offset+1)*p + (k_offset+1)][col] = -1
            W[(k_offset+1)*n + 0][col] = 1
            col += 1
            
            # M4: d(g-c)
            # A: d(1,1)
            # B: g(1,0), c(1,0)?? No, c is A. g is B.
            # B: g(1,0), -c? No, c is in A.
            # M4 = d * (g-c) is wrong. Strassen: M4 = d(g-c) is not standard.
            # Standard Strassen:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B11)
            # M4 = A22(B22-B12)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # Let's re-verify M4.
            # M4 = A22(B22-B12)
            # A: d(1,1)
            # B: h(1,1), f(0,1) with -1
            # C: (1,1) coef 1?
            # C22 = M1 - M2 + M3 + M6.
            # C21 = M2 + M4.
            # So M4 contributes to C21 (1,0) with coef 1.
            
            U[1*m + (j_offset+1)][col] = 1
            V[(j_offset+1)*p + (k_offset+1)][col] = 1
            V[(j_offset+0)*p + (k_offset+1)][col] = -1
            W[(k_offset+0)*n + 1][col] = 1
            col += 1
            
            # M5: (A11+A12)B22
            # A: a(0,0), b(0,1)
            # B: h(1,1)
            # C: (0,0) coef -1, (0,1) coef 1
            U[0*m + (j_offset+0)][col] = 1
            U[0*m + (j_offset+1)][col] = 1
            V[(j_offset+1)*p + (k_offset+1)][col] = 1
            W[(k_offset+0)*n + 0][col] = -1
            W[(k_offset+1)*n + 0][col] = 1
            col += 1
            
            # M6: (A21-A11)(B11+B12)
            # A: c(1,0), a(0,0) with -1
            # B: e(0,0), f(0,1)
            # C: (1,1) coef 1
            U[1*m + (j_offset+0)][col] = 1
            U[0*m + (j_offset+0)][col] = -1
            V[(j_offset+0)*p + (k_offset+0)][col] = 1
            V[(j_offset+0)*p + (k_offset+1)][col] = 1
            W[(k_offset+1)*n + 1][col] = 1
            col += 1
            
            # M7: (A12-A22)(B21+B22)
            # A: b(0,1), d(1,1) with -1
            # B: g(1,0), h(1,1)
            # C: (0,0) coef 1
            U[0*m + (j_offset+1)][col] = 1
            U[1*m + (j_offset+1)][col] = -1
            V[(j_offset+1)*p + (k_offset+0)][col] = 1
            V[(j_offset+1)*p + (k_offset+1)][col] = 1
            W[(k_offset+0)*n + 0][col] = 1
            col += 1
            
            # Block i_block 1: i=2
            # 1x2x2 Schoolbook
            # A: (2, j_local)
            # B: (j_local, k_local)
            # C: (2, k_local)
            
            # Term 1: j=0, k=0
            U[2*m + (j_offset+0)][col] = 1
            V[(j_offset+0)*p + (k_offset+0)][col] = 1
            W[(k_offset+0)*n + 2][col] = 1
            col += 1
            
            # Term 2: j=0, k=1
            U[2*m + (j_offset+0)][col] = 1
            V[(j_offset+0)*p + (k_offset+1)][col] = 1
            W[(k_offset+1)*n + 2][col] = 1
            col += 1
            
            # Term 3: j=1, k=0
            U[2*m + (j_offset+1)][col] = 1
            V[(j_offset+1)*p + (k_offset+0)][col] = 1
            W[(k_offset+0)*n + 2][col] = 1
            col += 1
            
            # Term 4: j=1, k=1
            U[2*m + (j_offset+1)][col] = 1
            V[(j_offset+1)*p + (k_offset+1)][col] = 1
            W[(k_offset+1)*n + 2][col] = 1
            col += 1

    if col != rank:
        print(f"Warning: used {col} columns, expected {rank}")
        
    return {"U": U, "V": V, "W": W}
