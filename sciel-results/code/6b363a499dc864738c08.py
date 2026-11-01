"""
Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.

Strategy:
- The schoolbook rank is 2*2*14 = 56.
- We know Strassen-style tricks can reduce rank for certain matrix sizes.
- For 2x2 * 2x14, we can view this as 7 independent 2x2 * 2x2 multiplications
  (since the second matrix is 2x14, we can split it into 7 blocks of 2x2).
  Each 2x2 * 2x2 can be done in rank 7 (Strassen) instead of rank 8.
  So total rank = 7 * 7 = 49.

The Strassen decomposition for 2x2 * 2x2:
  M1 = (A11 + A22)(B11 + B22)
  M2 = (A21 + A22) B11
  M3 = A11 (B12 - B22)
  M4 = A22 (B21 - B11)
  M5 = (A11 + A12) B22
  M6 = (A21 - A11) (B11 + B12)
  M7 = (A12 - A22) (B21 + B22)

  C11 = M1 + M4 - M5 + M7
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M2 + M3 + M6

For our 2x2 * 2x14 problem:
  A is 2x2, B is 2x14, C is 2x14.
  
  We split B into 7 blocks B^(t) of size 2x2 for t=0..6.
  C^(t) = A * B^(t) for each t.
  
  For each block, use Strassen with 7 multiplications.
  Total rank = 7 * 7 = 49.

Tensor convention:
  U has shape (n*m, rank) = (4, 49)
  V has shape (m*p, rank) = (28, 49)
  W has shape (p*n, rank) = (28, 49)
  
  The multiplication tensor T_{i,j,k} = delta_{i,k} (for A_{i,j} * B_{j,k} = C_{i,k})
  Wait, let me be careful with indices.
  
  Matrix multiplication: C_{i,k} = sum_j A_{i,j} * B_{j,k}
  
  In the tensor decomposition:
    T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    where U is indexed by (i,j) pairs, V by (j,k) pairs, W by (i,k) pairs.
    
    Actually, the standard convention:
    U[i*m+j, r] - coefficient for A_{i,j} in rank-r term
    V[j*p+k, r] - coefficient for B_{j,k} in rank-r term  
    W[k*n+i, r] - coefficient for C_{i,k} in rank-r term
    
    The tensor element T_{(i,j),(j,k),(k,i)} should equal 1 (delta for matrix mult).
    Wait, the tensor for matrix mult has T_{i,j,k} = 1 if the indices match for C_ik = sum_j A_ij B_jk.
    
    More precisely, the multiplication tensor has components:
    T_{(i,j),(j,k),(i,k)} = 1
    
    In the factorization: sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(i,k),r} = T_{(i,j),(j,k),(i,k)}
    
    So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Wait, that's not right either. Let me think again.
    
    The matrix multiplication tensor is defined such that:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    This can be written as: C = A * B, where the bilinear form is:
    <C, X> = <A, X * B^T> ... no.
    
    The standard way: The tensor T has T_{ijk} = 1 where i indexes rows of A and C,
    j indexes cols of A / rows of B, k indexes cols of B and C.
    
    T_{i,j,k} = 1 (it's just the structure tensor of matrix multiplication)
    
    The rank-r decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
    
    In our indexing:
    U is (n*m) x rank, indexed by row = i*m + j
    V is (m*p) x rank, indexed by row = j*p + k
    W is (p*n) x rank, indexed by row = k*n + i
    
    So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i,j,k.
    
    Wait, that gives T_{i,j,k} = 1 for ALL i,j,k, which is wrong. The matrix multiplication
    tensor should have T_{i,j,k} = 1 only when... no, actually for matrix multiplication,
    the tensor IS all ones in the sense that every combination (i,j,k) contributes to the
    product. Let me reconsider.
    
    Actually, the matrix multiplication tensor T satisfies:
    T(A,B) = A*B where (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    As a trilinear form: sum_{i,j,k} A_{i,j} B_{j,k} C'_{i,k} = <C, A*B> where C' is the "dual" of C.
    
    The tensor components are T_{i,j,k} = 1 for all i,j,k in range. Yes! Because:
    sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = <C, AB>
    
    So indeed T_{i,j,k} = 1 for all valid i, j, k.
    
    Therefore we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    For the Strassen-based approach with 7 blocks:
    Let me index the 14 columns of B as k = 0,1,...,13.
    Block t (t=0..6) covers columns k = 2*t and k = 2*t+1.
    
    For block t, the 2x2 submatrix B^(t) has entries:
    B^(t)_{0,0} = B_{0, 2t}
    B^(t)_{0,1} = B_{0, 2t+1}
    B^(t)_{1,0} = B_{1, 2t}
    B^(t)_{1,1} = B_{1, 2t+1}
    
    Strassen for block t uses 7 products:
    P1_t = (A_{0,0}+A_{1,1})(B^(t)_{0,0}+B^(t)_{1,1})
    P2_t = (A_{1,0}+A_{1,1}) B^(t)_{0,0}
    P3_t = A_{0,0}(B^(t)_{0,1}-B^(t)_{1,1})
    P4_t = A_{1,1}(B^(t)_{1,0}-B^(t)_{0,0})
    P5_t = (A_{0,0}+A_{0,1}) B^(t)_{1,1}
    P6_t = (A_{1,0}-A_{0,0})(B^(t)_{0,0}+B^(t)_{0,1})
    P7_t = (A_{0,1}-A_{1,1})(B^(t)_{1,0}+B^(t)_{1,1})
    
    Results:
    C^(t)_{0,0} = P1_t + P4_t - P5_t + P7_t
    C^(t)_{0,1} = P3_t + P5_t
    C^(t)_{1,0} = P2_t + P4_t
    C^(t)_{1,1} = P1_t - P2_t + P3_t + P6_t
    
    Now I need to express this as a rank-49 decomposition.
    Each product P_s_t (s=1..7, t=0..6) is one rank-1 term.
    
    For product P_s_t, I need to find coefficients a_{i,j}^{(s,t)}, b_{j,k}^{(s,t)}, c_{i,k}^{(s,t)}
    such that P_s_t = (sum_{i,j} a_{i,j} A_{i,j}) * (sum_{j,k} b_{j,k} B_{j,k})
    and the contribution to C_{i,k} is c_{i,k} * P_s_t.
    
    Let me work out each Strassen product:
    
    P1_t = (A_{00}+A_{11})(B_{0,2t}+B_{1,2t+1})
    a: A_{00}=1, A_{11}=1
    b: B_{0,2t}=1, B_{1,2t+1}=1
    c: C_{00}+=1, C_{01}+=0, C_{10}+=0, C_{11}+=1
    So c_{0,2t}=1, c_{1,2t+1}=1
    
    P2_t = (A_{10}+A_{11}) B_{0,2t}
    a: A_{10}=1, A_{11}=1
    b: B_{0,2t}=1
    c: C_{10}+=1, C_{11}+=-1
    So c_{1,2t}=1, c_{1,2t+1}=-1
    
    P3_t = A_{00}(B_{0,2t+1}-B_{1,2t+1})
    a: A_{00}=1
    b: B_{0,2t+1}=1, B_{1,2t+1}=-1
    c: C_{01}+=1, C_{11}+=1
    So c_{0,2t+1}=1, c_{1,2t+1}=1
    
    P4_t = A_{11}(B_{1,2t}-B_{0,2t})
    a: A_{11}=1
    b: B_{1,2t}=1, B_{0,2t}=-1
    c: C_{00}+=1, C_{10}+=1
    So c_{0,2t}=1, c_{1,2t}=1
    
    P5_t = (A_{00}+A_{01}) B_{1,2t+1}
    a: A_{00}=1, A_{01}=1
    b: B_{1,2t+1}=1
    c: C_{00}+=-1, C_{01}+=1
    So c_{0,2t}=-1, c_{0,2t+1}=1
    
    P6_t = (A_{10}-A_{00})(B_{0,2t}+B_{0,2t+1})
    a: A_{10}=1, A_{00}=-1
    b: B_{0,2t}=1, B_{0,2t+1}=1
    c: C_{11}+=1
    So c_{1,2t+1}=1
    
    P7_t = (A_{01}-A_{11})(B_{1,2t}+B_{1,2t+1})
    a: A_{01}=1, A_{11}=-1
    b: B_{1,2t}=1, B_{1,2t+1}=1
    c: C_{00}+=1
    So c_{0,2t}=1
    
    Let me verify with the Strassen formulas:
    C_{00} = P1 + P4 - P5 + P7
    Coeff of C_{00}: P1:1, P4:1, P5:-1, P7:1 -> 1+1-1+1=2? That's wrong, should be 1.
    
    Wait, I think I made an error. Let me re-derive Strassen carefully.
    
    Standard Strassen for 2x2:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    i = p1 + p4 - p5 + p7
    j = p3 + p5
    k = p2 + p4
    l = p1 - p2 + p3 + p6
    
    Let me verify i:
    p1 + p4 - p5 + p7 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    = ae + (ah-ah) + (de-de) + (dh-dh) + dg-dg + (-bh+bh) + bg
    = ae + bg
    
    Yes! That's correct. i = ae+bg.
    
    So my formulas are correct. Let me re-check the C coefficients.
    
    For block t, mapping:
    a=A_{00}, b=A_{01}, c=A_{10}, d=A_{11}
    e=B_{0,2t}, f=B_{0,2t+1}, g=B_{1,2t}, h=B_{1,2t+1}
    i=C_{0,2t}, j=C_{0,2t+1}, k=C_{1,2t}, l=C_{1,2t+1}
    
    p1 = (A00+A11)(B_{0,2t}+B_{1,2t+1})
    p2 = (A10+A11) B_{0,2t}
    p3 = A00(B_{0,2t+1}-B_{1,2t+1})
    p4 = A11(B_{1,2t}-B_{0,2t})
    p5 = (A00+A01) B_{1,2t+1}
    p6 = (A10-A00)(B_{0,2t}+B_{0,2t+1})
    p7 = (A01-A11)(B_{1,2t}+B_{1,2t+1})
    
    C_{0,2t} = p1 + p4 - p5 + p7
    C_{0,2t+1} = p3 + p5
    C_{1,2t} = p2 + p4
    C_{1,2t+1} = p1 - p2 + p3 + p6
    
    Now for the tensor decomposition, each p_s contributes to C with specific coefficients:
    
    p1: contributes +1 to C_{0,2t} and +1 to C_{1,2t+1}
    p2: contributes +1 to C_{1,2t} and -1 to C_{1,2t+1}
    p3: contributes +1 to C_{0,2t+1} and +1 to C_{1,2t+1}
    p4: contributes +1 to C_{0,2t} and +1 to C_{1,2t}
    p5: contributes -1 to C_{0,2t} and +1 to C_{0,2t+1}
    p6: contributes +1 to C_{1,2t+1}
    p7: contributes +1 to C_{0,2t}
    
    Let me build the U, V, W matrices.
    
    U has 4 rows (i*m+j for i=0,1 and j=0,1):
    Row 0: (i=0,j=0) -> A_{00}
    Row 1: (i=0,j=1) -> A_{01}
    Row 2: (i=1,j=0) -> A_{10}
    Row 3: (i=1,j=1) -> A_{11}
    
    V has 28 rows (j*p+k for j=0,1 and k=0..13):
    Row j*14+k
    
    W has 28 rows (k*n+i for k=0..13 and i=0,1):
    Row k*2+i
    
    For each (t, s) where t=0..6, s=1..7, we have one column r = t*7 + (s-1).
    
    U values for each (t,s):
    s=1: A00=1, A01=0, A10=0, A11=1
    s=2: A00=0, A01=0, A10=1, A11=1
    s=3: A00=1, A01=0, A10=0, A11=0
    s=4: A00=0, A01=0, A10=0, A11=1
    s=5: A00=1, A01=1, A10=0, A11=0
    s=6: A00=-1, A01=0, A10=1, A11=0
    s=7: A00=0, A01=1, A10=0, A11=-1
    
    V values for each (t,s):
    s=1: B_{0,2t}=1, B_{1,2t+1}=1
    s=2: B_{0,2t}=1
    s=3: B_{0,2t+1}=1, B_{1,2t+1}=-1
    s=4: B_{1,2t}=1, B_{0,2t}=-1
    s=5: B_{1,2t+1}=1
    s=6: B_{0,2t}=1, B_{0,2t+1}=1
    s=7: B_{1,2t}=1, B_{1,2t+1}=1
    
    W values for each (t,s):
    s=1: C_{0,2t}=1, C_{1,2t+1}=1
    s=2: C_{1,2t}=1, C_{1,2t+1}=-1
    s=3: C_{0,2t+1}=1, C_{1,2t+1}=1
    s=4: C_{0,2t}=1, C_{1,2t}=1
    s=5: C_{0,2t}=-1, C_{0,2t+1}=1
    s=6: C_{1,2t+1}=1
    s=7: C_{0,2t}=1
"""

import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # Strassen-based rank-49 decomposition
    # 7 blocks of 2x2, each using 7 Strassen multiplications
    num_blocks = p // 2  # 7
    strassen_per_block = 7
    rank = num_blocks * strassen_per_block  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # U rows: (i,j) -> i*m+j
    #   row 0: (0,0) A00
    #   row 1: (0,1) A01
    #   row 2: (1,0) A10
    #   row 3: (1,1) A11
    
    # V rows: (j,k) -> j*p+k
    #   j=0: rows 0-13 (B_{0,k})
    #   j=1: rows 14-27 (B_{1,k})
    
    # W rows: (k,i) -> k*n+i
    #   i=0: rows 0,2,4,...,26 (C_{0,k})
    #   i=1: rows 1,3,5,...,27 (C_{1,k})
    
    for t in range(num_blocks):
        k0 = 2 * t      # first column of block
        k1 = 2 * t + 1  # second column of block
        
        for s in range(strassen_per_block):
            col = t * strassen_per_block + s
            
            if s == 0:  # p1 = (A00+A11)(B_{0,k0}+B_{1,k1})
                U[0][col] = 1   # A00
                U[3][col] = 1   # A11
                V[0 * p + k0][col] = 1   # B_{0,k0}
                V[1 * p + k1][col] = 1   # B_{1,k1}
                W[k0 * n + 0][col] = 1   # C_{0,k0}
                W[k1 * n + 1][col] = 1   # C_{1,k1}
                
            elif s == 1:  # p2 = (A10+A11) * B_{0,k0}
                U[2][col] = 1   # A10
                U[3][col] = 1   # A11
                V[0 * p + k0][col] = 1   # B_{0,k0}
                W[k0 * n + 1][col] = 1   # C_{1,k0}
                W[k1 * n + 1][col] = -1  # C_{1,k1}
                
            elif s == 2:  # p3 = A00 * (B_{0,k1}-B_{1,k1})
                U[0][col] = 1   # A00
                V[0 * p + k1][col] = 1   # B_{0,k1}
                V[1 * p + k1][col] = -1  # B_{1,k1}
                W[k1 * n + 0][col] = 1   # C_{0,k1}
                W[k1 * n + 1][col] = 1   # C_{1,k1}
                
            elif s == 3:  # p4 = A11 * (B_{1,k0}-B_{0,k0})
                U[3][col] = 1   # A11
                V[1 * p + k0][col] = 1   # B_{1,k0}
                V[0 * p + k0][col] = -1  # B_{0,k0}
                W[k0 * n + 0][col] = 1   # C_{0,k0}
                W[k0 * n + 1][col] = 1   # C_{1,k0}
                
            elif s == 4:  # p5 = (A00+A01) * B_{1,k1}
                U[0][col] = 1   # A00
                U[1][col] = 1   # A01
                V[1 * p + k1][col] = 1   # B_{1,k1}
                W[k0 * n + 0][col] = -1  # C_{0,k0}
                W[k1 * n + 0][col] = 1   # C_{0,k1}
                
            elif s == 5:  # p6 = (A10-A00) * (B_{0,k0}+B_{0,k1})
                U[0][col] = -1  # A00
                U[2][col] = 1   # A10
                V[0 * p + k0][col] = 1   # B_{0,k0}
                V[0 * p + k1][col] = 1   # B_{0,k1}
                W[k1 * n + 1][col] = 1   # C_{1,k1}
                
            elif s == 6:  # p7 = (A01-A11) * (B_{1,k0}+B_{1,k1})
                U[1][col] = 1   # A01
                U[3][col] = -1  # A11
                V[1 * p + k0][col] = 1   # B_{1,k0}
                V[1 * p + k1][col] = 1   # B_{1,k1}
                W[k0 * n + 0][col] = 1   # C_{0,k0}
    
    # Verify correctness with a simple check
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        # This is a batched check. Let's verify a few random cases.
        np.random.seed(42)
        for _ in range(50):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
            if abs(val - 1.0) > 1e-9:
                # Fallback to schoolbook if verification fails
                rank_fallback = n * m * p
                U_fb = [[0] * rank_fallback for _ in range(n * m)]
                V_fb = [[0] * rank_fallback for _ in range(m * p)]
                W_fb = [[0] * rank_fallback for _ in range(p * n)]
                col = 0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            U_fb[ii*m+jj][col] = 1
                            V_fb[jj*p+kk][col] = 1
                            W_fb[kk*n+ii][col] = 1
                            col += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
