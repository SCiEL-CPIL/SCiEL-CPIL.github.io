import numpy as np

def construct():
    """
    Constructs a rank-22 exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Uses block splitting:
    - j=0 slice (2x1x4): 8 rank-1 terms (trivial)
    - j in {1,2}, k in {0,1} slice (2x2x2): 7 rank-1 terms (Strassen)
    - j in {1,2}, k in {2,3} slice (2x2x2): 7 rank-1 terms (Strassen)
    Total rank = 8 + 7 + 7 = 22.
    """
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    rank = 22
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u, v, w):
        nonlocal col
        U[u][col] += v[0] * 1.0  # Wait, u,v,w are indices? No, they are the values for the rank-1 term.
        # Actually, let's pass the index and the value.
        # Let's redefine: add_term(a, b, c, val)
        # U[a][col] = val, V[b][col] = val, W[c][col] = val? No.
        # A rank-1 term is U[:, r] * V[:, r] * W[:, r].
        # So U[a, r] = u_val, V[b, r] = v_val, W[c, r] = w_val.
        pass

    def set_term(a, b, c, val):
        nonlocal col
        U[a][col] = val
        V[b][col] = val
        W[c][col] = val
        col += 1

    def set_term_scaled(a, b, c, val):
        nonlocal col
        U[a][col] = val
        V[b][col] = 1.0
        W[c][col] = 1.0
        col += 1

    # The term contributes U[a,r]*V[b,r]*W[c,r] to T[a,b,c].
    # If we set U[a,r]=u, V[b,r]=v, W[c,r]=w, the contribution is u*v*w.
    # We can distribute the scaling arbitrarily among u,v,w as long as product is correct.
    # Let's just set U[a,r]=val, V[b,r]=1, W[c,r]=1. Then contribution is val.
    # But wait, a single rank-1 term affects ALL (a',b',c') where U[a',r]!=0, V[b',r]!=0, W[c',r]!=0.
    # So if we set U[a,r]=val, V[b,r]=1, W[c,r]=1, it ONLY affects (a,b,c) if U,V,W are zero elsewhere for this rank.
    # Yes, since we initialize to 0 and only set one index per matrix per rank.
    
    # 1. j=0 slice (8 terms)
    # T[3i, 4k, 2k+i] = 1 for i in {0,1}, k in {0,1,2,3}
    for i in range(n):
        for k in range(p):
            a = i * m + 0
            b = 0 * p + k
            c = k * n + i
            set_term_scaled(a, b, c, 1.0)

    # 2. Strassen for j in {1,2}, k in {0,1}
    # Submatrix A_sub (2x2): rows i in {0,1}, cols j in {1,2} -> A[3i+1], A[3i+2]
    # Submatrix B_sub (2x2): rows j in {1,2}, cols k in {0,1} -> B[1*4+k], B[2*4+k]
    # Output C_sub (2x2): rows k in {0,1}, cols i in {0,1} -> C[k*2+i]
    
    # Strassen 7 products:
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to decompose this into 7 rank-1 terms for the whole tensor.
    # Each Strassen product m_r corresponds to a rank-1 term in the sub-tensor.
    # But we must map it to the global indices.
    
    # Let's define the 7 terms explicitly.
    # Term 1: m1
    # U part: A00 + A11 -> U[3*0+1]=1, U[3*1+2]=1 (indices 1 and 5)
    # V part: B00 + B11 -> V[1*4+0]=1, V[2*4+1]=1 (indices 4 and 9)
    # W part: C00 + C01 + C10 + C11? No.
    # m1 contributes to C00 and C11.
    # So W[0*2+0]=1 (C00), W[1*2+1]=1 (C11).
    # Wait, the rank-1 term is U_vec * V_vec * W_vec.
    # The contribution to T[a,b,c] is U[a,r]*V[b,r]*W[c,r].
    # For m1:
    # We want T[1,4,c] and T[5,9,c] to have contributions?
    # No, the Strassen algorithm computes the sub-matrix product.
    # The sub-tensor T_sub[a',b',c'] for a' in {0,1}, b' in {0,1}, c' in {0,1} is 1 if a'=i', b'=j', c'=k' ...
    # Actually, the sub-tensor is the matrix multiplication tensor for 2x2x2.
    # T_sub[i', j', k'] = 1 if i'=i, j'=j, k'=k? No.
    # T_sub[i', j', k'] = 1 if the output C[i', j'] depends on A[i', k'] and B[k', j'].
    # So T_sub[i', k', j'] = 1?
    # Let's stick to the global indices.
    
    # Let's construct the 7 vectors for the j in {1,2}, k in {0,1} block.
    # A indices: 1, 2, 4, 5 (i=0,j=1; i=0,j=2; i=1,j=1; i=1,j=2)
    # B indices: 4, 5, 8, 9 (j=1,k=0; j=1,k=1; j=2,k=0; j=2,k=1)
    # C indices: 0, 1, 2, 3 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    
    # m1: (A00+A11)(B00+B11)
    # A00=1, A11=5, B00=4, B11=9
    # Contributes to C00 (0) and C11 (3)
    U[1][col] = 1; U[5][col] = 1
    V[4][col] = 1; V[9][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # m2: (A10+A11)B00
    # A10=4, A11=5, B00=4
    # Contributes to C10 (2)
    U[4][col] = 1; U[5][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # m3: A00(B01-B11)
    # A00=1, B01=5, B11=9
    # Contributes to C01 (1)
    U[1][col] = 1
    V[5][col] = 1; V[9][col] = -1
    W[1][col] = 1
    col += 1
    
    # m4: A11(B10-B00)
    # A11=5, B10=8, B00=4
    # Contributes to C00 (0) and C10 (2)
    U[5][col] = 1
    V[8][col] = 1; V[4][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # m5: (A00+A01)B11
    # A00=1, A01=2, B11=9
    # Contributes to C00 (0) with -1 and C01 (1) with +1
    U[1][col] = 1; U[2][col] = 1
    V[9][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # m6: (A10-A00)(B00+B01)
    # A10=4, A00=1, B00=4, B01=5
    # Contributes to C11 (3)
    U[4][col] = 1; U[1][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[3][col] = 1
    col += 1
    
    # m7: (A01-A11)(B10+B11)
    # A01=2, A11=5, B10=8, B11=9
    # Contributes to C00 (0) with +1 and C11 (3) with +1
    U[2][col] = 1; U[5][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # 3. Strassen for j in {1,2}, k in {2,3}
    # A indices: 1, 2, 4, 5
    # B indices: 6, 7, 10, 11 (j=1,k=2; j=1,k=3; j=2,k=2; j=2,k=3)
    # C indices: 4, 5, 6, 7 (k=2,i=0; k=2,i=1; k=3,i=0; k=3,i=1)
    
    # m1: (A00+A11)(B00+B11)
    # A00=1, A11=5, B00=6, B11=11
    # Contributes to C4 (4) and C7 (7)
    U[1][col] = 1; U[5][col] = 1
    V[6][col] = 1; V[11][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # m2: (A10+A11)B00
    # A10=4, A11=5, B00=6
    # Contributes to C6 (6)
    U[4][col] = 1; U[5][col] = 1
    V[6][col] = 1
    W[6][col] = 1
    col += 1
    
    # m3: A00(B01-B11)
    # A00=1, B01=7, B11=11
    # Contributes to C5 (5)
    U[1][col] = 1
    V[7][col] = 1; V[11][col] = -1
    W[5][col] = 1
    col += 1
    
    # m4: A11(B10-B00)
    # A11=5, B10=10, B00=6
    # Contributes to C4 (4) and C6 (6)
    U[5][col] = 1
    V[10][col] = 1; V[6][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # m5: (A00+A01)B11
    # A00=1, A01=2, B11=11
    # Contributes to C4 (4) with -1 and C5 (5) with +1
    U[1][col] = 1; U[2][col] = 1
    V[11][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # m6: (A10-A00)(B00+B01)
    # A10=4, A00=1, B00=6, B01=7
    # Contributes to C7 (7)
    U[4][col] = 1; U[1][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # m7: (A01-A11)(B10+B11)
    # A01=2, A11=5, B10=10, B11=11
    # Contributes to C4 (4) with +1 and C7 (7) with +1
    U[2][col] = 1; U[5][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # Verification (optional, for debugging)
    # valid_indices = []
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             a = i * m + j
    #             b = j * p + k
    #             c = k * n + i
    #             valid_indices.append((a, b, c))
            
    # for a, b, c in valid_indices:
    #     val = 0.0
    #     for r in range(rank):
    #         val += U[a][r] * V[b][r] * W[c][r]
    #     if abs(val - 1.0) > 1e-9:
    #         print(f"Error at {a},{b},{c}: {val}")
    
    return {"U": U, "V": V, "W": W}
