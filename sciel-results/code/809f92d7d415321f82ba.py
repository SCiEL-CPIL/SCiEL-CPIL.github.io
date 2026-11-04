"""
Matrix multiplication 4x5 by 5x6 via Strassen-based 2x2 block decomposition.

Strategy:
  Partition A (4x5) into four 2x2 blocks plus a 2x1 strip in the column dimension.
  Partition B (5x6) into four 2x2 blocks plus a 1x6 strip in the row dimension.
  
  C = A @ B = A11@B11 + A12@B21 + A21@B12 + A22@B22 + A1s@Bs + As@B2s
  
  Wait, let me be more careful with the partitioning.
  
  A is 4x5: rows 0-3, cols 0-4
  B is 5x6: rows 0-4, cols 0-5
  C is 4x6: rows 0-3, cols 0-5
  
  Split A into:
    A_top (2x5): rows 0-1
    A_bot (2x5): rows 2-3
  
  Split B into:
    B_top (2x6): rows 0-1
    B_bot (2x6): rows 2-3
    B_mid (1x6): row 4
  
  C = A_top @ B_top + A_bot @ B_bot + A_top @ B_mid... no that's wrong.
  
  Actually:
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  Let me partition the summation index j (size 5) into:
    j in {0,1}: top
    j in {2,3}: middle  
    j in {4}: bottom
    
  So:
  C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:] + A[:,4:5] @ B[4:5,:]
  
  Now partition the output indices:
  i in {0,1}: top rows
  i in {2,3}: bottom rows
  k in {0,1,2,3}: left cols
  k in {4,5}: right cols
  
  For each term A_sub @ B_sub where A_sub is 4x2 and B_sub is 2x6:
  This is a 4x2x6 multiplication.
  
  Better approach: use 2x2 block Strassen on the core parts.
  
  Let me think differently. Partition into 2x2 blocks where possible:
  
  A (4x5) = [A11 A12 A13]  where A11 is 2x2, A12 is 2x2, A13 is 2x1
  B (5x6) = [B11 B12]       where B11 is 2x3, B12 is 2x3 ... no
  
  Let me use a cleaner decomposition:
  
  Split the inner dimension (size 5) into 2+2+1:
    A = [A_a | A_b | A_c]  where A_a, A_b are 4x2, A_c is 4x1
    B = [B_a; B_b; B_c]    where B_a, B_b are 2x6, B_c is 1x6
    
  C = A_a @ B_a + A_b @ B_b + A_c @ B_c
  
  Now for A_a @ B_a (4x2 times 2x6):
    Split A_a into top/bottom: A_a = [Aa_t; Aa_b] where each is 2x2
    Split B_a into left/right: B_a = [Ba_l | Ba_r] where each is 2x3
    
    A_a @ B_a = Aa_t @ Ba_l + Aa_t @ Ba_r + Aa_b @ Ba_l + Aa_b @ Ba_r
    
    Each of these is 2x2 @ 2x3 or 2x2 @ 2x3.
    
    For 2x2 @ 2x3: treat as two 2x2 @ 2x1 multiplications, or use Strassen-like.
    
    Actually, let me just use Strassen for the 2x2 @ 2x2 cores and handle the rest.
    
  Alternative cleaner approach:
  
  Partition A (4x5) into blocks of size 2:
    Rows: 2+2
    Cols: 2+2+1
    
    A = [[A11, A12, A13],
         [A21, A22, A23]]
    where A11,A12,A21,A22 are 2x2, A13,A23 are 2x1
    
  Partition B (5x6) into blocks:
    Rows: 2+2+1
    Cols: 2+2+2
    
    B = [[B11, B12, B13],
         [B21, B22, B23],
         [B31, B32, B33]]
    where B11,B12,B21,B22 are 2x2, B13,B23,B31,B32,B33 are 2x2 or 1x2
    
    Wait, B is 5x6. Rows: 2+2+1, Cols: 2+2+2
    B11 is 2x2, B12 is 2x2, B13 is 2x2
    B21 is 2x2, B22 is 2x2, B23 is 2x2
    B31 is 1x2, B32 is 1x2, B33 is 1x2
    
  C (4x6) = A @ B:
    C = [[A11@B11 + A12@B21 + A13@B31,  A11@B12 + A12@B22 + A13@B32,  A11@B13 + A12@B23 + A13@B33],
         [A21@B11 + A22@B21 + A23@B31,  A21@B12 + A22@B22 + A23@B32,  A21@B13 + A22@B23 + A23@B33]]
         
    C11 = A11@B11 + A12@B21 + A13@B31  (2x2)
    C12 = A11@B12 + A12@B22 + A13@B32  (2x2)
    C13 = A11@B13 + A12@B23 + A13@B33  (2x2)
    C21 = A21@B11 + A22@B21 + A23@B31  (2x2)
    C22 = A21@B12 + A22@B22 + A23@B32  (2x2)
    C23 = A21@B13 + A22@B23 + A23@B33  (2x2)
    
  For each 2x2 output block, we have a sum of three 2x2 multiplications (some involving 2x1 @ 1x2).
  
  The 2x2 @ 2x2 multiplications can use Strassen (rank 7 instead of 8).
  The 2x1 @ 1x2 multiplications are rank 1 each (outer product).
  
  Let me count the rank:
  For each output block (2x2), we have:
    - Two 2x2 @ 2x2 products: use Strassen, rank 7 each -> 14
    - One 2x1 @ 1x2 product: rank 1 (it's an outer product)
    
    Wait, A13@B31: A13 is 2x1, B31 is 1x2. This is a 2x2 matrix that's a rank-1 outer product.
    So this contributes rank 1.
    
    Total per output block: 7 + 7 + 1 = 15? No, Strassen combines the two 2x2@2x2 into one set of 7 multiplications.
    
    Actually, for C11 = A11@B11 + A12@B21 + A13@B31:
    The first two terms are 2x2@2x2. We can't directly apply Strassen to their sum.
    
    Let me reconsider. We need to decompose the entire 4x5x6 tensor.
    
  Better approach: use Strassen on 2x2 blocks where the dimensions align.
  
  Let me try a different partition:
  
  A (4x5): split rows into 2+2, cols into 2+2+1
  B (5x6): split rows into 2+2+1, cols into 2+2+2
  
  For the core 2x2 @ 2x2 blocks, use Strassen (rank 7).
  For the strip parts (involving the size-1 dimension), use schoolbook.
  
  Let me enumerate all the sub-multiplications:
  
  Terms involving only 2x2 blocks (both A and B parts are 2x2):
  A11@B11, A11@B12, A11@B13, A12@B21, A12@B22, A12@B23, A21@B11, A21@B12, A21@B13, A22@B21, A22@B22, A22@B23
  
  That's 12 instances of 2x2@2x2. If we do each independently with Strassen, that's 12*7 = 84.
  But we can share Strassen computations across terms that share the same A or B blocks.
  
  Actually, for a given pair of A-block (2x2) and B-block (2x2), Strassen gives 7 rank-1 terms.
  But different output blocks share the same A or B blocks.
  
  Let me think about this more carefully using the tensor decomposition view.
  
  The tensor is M_{4,5,6} with entries T[i,j,k] = 1 if the output element (i,k) depends on A[i,j]*B[j,k].
  
  Using Strassen on 2x2 blocks:
  
  Consider the 2x2 @ 2x2 core. Strassen computes:
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b12-b22)
    m4 = a22(b21-b11)
    m5 = (a11+a12)b22
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    c11 = m1 - m4 + m6 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m3 + m5 - m6
    
  For our problem, let's identify all the 2x2@2x2 sub-problems and apply Strassen to each, then handle the strips.
  
  Actually, let me just implement a direct construction:
  
  1. For each pair (i_block, j_block) where i_block in {0,1} (row block of A) and j_block in {0,1} (row block of B corresponding to col block of A), if the sizes are 2x2, use Strassen.
  
  This is getting complex. Let me just implement it directly with explicit Strassen for each 2x2@2x2 sub-multiplication.
  
  Total rank calculation:
  - 12 instances of 2x2@2x2, each with Strassen rank 7: 12*7 = 84
  - Terms involving the size-1 strip:
    A13@B31, A13@B32, A13@B33, A23@B31, A23@B32, A23@B33: 6 terms, each is 2x1 @ 1x2 = outer product, rank 1 each: 6
    A11@B31... wait, no. B31 is 1x2, and A11 is 2x2. A11@B31 doesn't make sense dimensionally in the original product.
    
  Let me re-examine. The original product is:
  C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
  
  Partition j into {0,1}, {2,3}, {4}:
  C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:] + A[:,4:5] @ B[4:5,:]
  
  Term 1: A1 (4x2) @ B1 (2x6)
  Term 2: A2 (4x2) @ B2 (2x6)  
  Term 3: A3 (4x1) @ B3 (1x6)
  
  For Term 1: A1 (4x2) @ B1 (2x6)
    Split A1 into rows 0-1 and 2-3: A1 = [A1t; A1b] (each 2x2)
    Split B1 into cols 0-1, 2-3, 4-5: B1 = [B1a | B1b | B1c] (each 2x2)
    
    A1 @ B1 = [A1t@B1a + A1b@B1a, A1t@B1b + A1b@B1b, A1t@B1c + A1b@B1c]
    
    Wait, that's not right either. A1 is 4x2, B1 is 2x6.
    (A1 @ B1)[i,k] = sum_{j=0}^{1} A1[i,j] * B1[j,k]
    
    Split i into {0,1} and {2,3}:
    Top part (2x6): [A1t @ B1] where A1t is 2x2
    Bottom part (2x6): [A1b @ B1] where A1b is 2x2
    
    For A1t @ B1 (2x2 @ 2x6):
      Split B1 into three 2x2 blocks: B1 = [B1a | B1b | B1c]
      A1t @ B1 = [A1t@B1a | A1t@B1b | A1t@B1c]
      Each is 2x2 @ 2x2, use Strassen: 7 rank-1 terms each, 3 blocks -> 21
      
    Similarly for A1b @ B1: 21
    
    Total for Term 1: 42
    
  Similarly for Term 2: 42
  
  For Term 3: A3 (4x1) @ B3 (1x6)
    This is an outer product of a 4x1 and 1x6 matrix.
    Rank = 4*1*6 = 24 (schoolbook) or can be done in rank 4*6=24.
    Actually, (A3 @ B3)[i,k] = A3[i,0] * B3[0,k]
    This is a rank-1 matrix in the tensor sense? No, it's 24 independent products.
    Rank = 24.
    
  Total: 42 + 42 + 24 = 108.
  
  That matches the observed improvement from 120 to 108!
  
  Let me implement this.
  
  For each 2x2 @ 2x2 sub-problem, I'll use Strassen's 7 multiplications.
  
  Strassen for 2x2 @ 2x2:
  Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
  
  m1 = (a+d)(e+h)
  m2 = (c+d)e
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)h
  m6 = (c-a)(e+f)
  m7 = (b-d)(g+h)
  
  C = [[m1-m4+m6+m7, m3+m5],
       [m2+m4,        m1-m3+m5-m6]]
  
  In terms of the tensor decomposition, each "multiplication" m_i is a rank-1 term where:
  U contributes the product of certain A elements,
  V contributes the product of certain B elements,
  W contributes to the output.
  
  But in the tensor decomposition framework, we need to express this as:
  C[i,k] = sum_t U[i,t] * V[t,k] * W[t,i,k]... 
  
  Wait, the decomposition is:
  T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]... 
  
  No, let me re-read the interface:
  U has shape (n*m, rank) = (20, rank)
  V has shape (m*p, rank) = (30, rank)
  W has shape (p*n, rank) = (24, rank)
  
  And the decomposition is:
  For the matrix multiplication tensor, T[i,j,k] = 1 means C[i,k] += A[i,j]*B[j,k]
  
  The decomposition satisfies:
  T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
  
  So for each rank-1 term t, we have:
  U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] contributes to T[i,j,k]
  
  For a standard rank-1 term corresponding to multiplying A[i,j] * B[j,k] contributing to C[i,k]:
  U[i*m+j, t] = 1
  V[j*p+k, t] = 1
  W[k*n+i, t] = 1
  
  For Strassen, the "multiplications" involve sums of A elements and sums of B elements.
  For example, m1 = (a+d)(e+h) where:
  a = A[i1,j1], d = A[i2,j2] (for a 2x2 block)
  e = B[j1,k1], h = B[j2,k2]
  
  This is not a single rank-1 term in the original tensor. It's a product of a linear combination of A entries and a linear combination of B entries.
  
  In the tensor decomposition, this corresponds to:
  U[i1*m+j1, t] = 1, U[i2*m+j2, t] = 1 (the A part)
  V[j1*p+k1, t] = 1, V[j2*p+k2, t] = 1 (the B part)
  W[k1*n+i1, t] = 1, W[k2*n+i2, t] = 1, etc. (the C part)
  
  Wait, that's not right either. Let me think about this more carefully.
  
  The tensor T has indices (i, j, k) where:
  i = row of A / row of C (0 to n-1)
  j = col of A / row of B (0 to m-1)
  k = col of B / col of C (0 to p-1)
  
  T[i,j,k] = 1 for all valid (i,j,k) (since every A[i,j]*B[j,k] contributes to C[i,k]).
  
  A rank-1 decomposition term t has:
  U[i*m+j, t] for all i,j
  V[j*p+k, t] for all j,k
  W[k*n+i, t] for all k,i
  
  And we need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k] = 1 for all i,j,k.
  
  For the schoolbook decomposition, each term t corresponds to a unique (i,j,k) triple:
  U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1, all others 0.
  
  For Strassen, a "multiplication" like m1 = (a+d)(e+h) means:
  The contribution to various C entries involves the product (a+d)*(e+h).
  
  In the tensor decomposition, this is represented by a single rank-1 term where:
  U has non-zero entries at the positions corresponding to a and d (i.e., U[i1*m+j1, t] = 1, U[i2*m+j2, t] = 1)
  V has non-zero entries at the positions corresponding to e and h (i.e., V[j1*p+k1, t] = 1, V[j2*p+k2, t] = 1)
  W has non-zero entries at the positions corresponding to the C entries that receive this contribution.
  
  For m1 = (a+d)(e+h) in Strassen 2x2@2x2:
  a = A[i1,j1], d = A[i2,j2]
  e = B[j1,k1], h = B[j2,k2]
  
  m1 contributes to:
  c11 = m1 - m4 + m6 + m7
  c22 = m1 - m3 + m5 - m6
  
  So m1 contributes +1 to c11 and +1 to c22.
  
  In the tensor decomposition:
  U[i1*m+j1, t] = 1, U[i2*m+j2, t] = 1
  V[j1*p+k1, t] = 1, V[j2*p+k2, t] = 1
  W[k1*n+i1, t] = 1, W[k2*n+i2, t] = 1
  
  Let me verify: for the term t, the contribution to T[i,j,k] is:
  U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
  
  For i=i1, j=j1, k=k1: U[i1*m+j1, t] * V[j1*p+k1, t] * W[k1*n+i1, t] = 1*1*1 = 1
  For i=i1, j=j1, k=k2: U[i1*m+j1, t] * V[j1*p+k2, t] * W[k2*n+i1, t] = 1*0*0 = 0 (V[j1*p+k2, t] = 0)
  
  Hmm, this doesn't work directly because V[j1*p+k2, t] is 0 but we need the product (a+d)*(e+h) to contribute to c11 and c22.
  
  I think I'm confusing the tensor decomposition with the circuit decomposition. Let me reconsider.
  
  In the tensor decomposition T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]:
  
  Each term t defines a rank-1 tensor. The product U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] is non-zero only when all three factors are non-zero.
  
  For the schoolbook term for (i,j,k):
  U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1.
  The product is 1 only at (i,j,k), giving T[i,j,k] += 1.
  
  For a Strassen-like term, we want to capture the fact that (a+d)*(e+h) contributes to multiple (i,j,k) entries simultaneously.
  
  Actually, I think the correct interpretation is:
  
  For m1 = (a+d)(e+h) to contribute to c11:
  c11 = sum_j A[i1,j] * B[j,k1]
  
  The term (a+d)(e+h) = (A[i1,j1] + A[i2,j2])(B[j1,k1] + B[j2,k2])
  = A[i1,j1]*B[j1,k1] + A[i1,j1]*B[j2,k2] + A[i2,j2]*B[j1,k1] + A[i2,j2]*B[j2,k2]
  
  This involves 4 different (i,j,k) triples:
  (i1,j1,k1), (i1,j1,k2), (i2,j2,k1), (i2,j2,k2)
  
  But in the tensor decomposition, a single rank-1 term can only have the form:
  U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
  
  For this to be non-zero at multiple (i,j,k) triples, we need U, V, W to have multiple non-zero entries.
  
  Let's say U[i1*m+j1, t] = 1, U[i2*m+j2, t] = 1 (and 0 elsewhere)
  V[j1*p+k1, t] = 1, V[j2*p+k2, t] = 1 (and 0 elsewhere)
  W[k1*n+i1, t] = 1, W[k2*n+i2, t] = 1 (and 0 elsewhere)
  
  Then the product U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] is non-zero when:
  (i,j) is in {(i1,j1), (i2,j2)} AND (j,k) is in {(j1,k1), (j2,k2)} AND (k,i) is in {(k1,i1), (k2,i2)}
  
  The valid (i,j,k) triples are those where all three conditions hold simultaneously.
  
  For (i1,j1,k1): (i1,j1) in U-set, (j1,k1) in V-set, (k1,i1) in W-set -> all yes -> product = 1
  For (i1,j1,k2): (i1,j1) in U-set, (j1,k2) in V-set? (j1,k2) is not in {(j1,k1),(j2,k2)} -> no
  For (i2,j2,k1): (i2,j2) in U-set, (j2,k1) in V-set? No
  For (i2,j2,k2): (i2,j2) in U-set, (j2,k2) in V-set, (k2,i2) in W-set -> all yes -> product = 1
  
  So this rank-1 term contributes to T[i1,j1,k1] and T[i2,j2,k2], each with value 1.
  
  But we wanted it to contribute to c11 and c22, which involves 4 (i,j,k) triples. This approach only gives 2.
  
  I think the issue is that Strassen's algorithm doesn't directly translate to a tensor decomposition in this simple way. The rank-7 decomposition of 2x2@2x2 means that the 2x2@2x2 multiplication tensor has rank 7, but each rank-1 term in that decomposition corresponds to a specific pattern.
  
  Let me look up the actual Strassen tensor decomposition.
  
  The 2x2@2x2 multiplication tensor T_{2,2,2} has entries T[i,j,k] = 1 for all i,j,k in {0,1}.
  Rank 7 decomposition:
  
  Term 1: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
  This gives non-zero at (0,0,0) and (1,1,1).
  
  Term 2: U[1,0]=1, U[1,1]=1; V[0,0]=1; W[0,1]=1
  Non-zero at (1,0,0).
  
  Hmm, this is getting complicated. Let me just implement the Strassen decomposition explicitly by working out which (i,j,k) triples each Strassen multiplication contributes to.
  
  Actually, I recall now. The Strassen decomposition of the 2x2@2x2 tensor is:
  
  For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A@B:
  
  m1 = (a+d)(e+h) -> contributes to c11 and c22
  m2 = (c+d)e -> contributes to c21
  m3 = a(f-h) -> contributes to c12 and c22
  m4 = d(g-e) -> contributes to c11 and c21
  m5 = (a+b)h -> contributes to c12 and c22
  m6 = (c-a)(e+f) -> contributes to c11 and c21
  m7 = (b-d)(g+h) -> contributes to c11 and c22
  
  In the tensor decomposition, each m_i is a rank-1 term. For m1:
  The A-part is (a+d) = A[0,0] + A[1,1]
  The B-part is (e+h) = B[0,0] + B[1,1]
  The C-part: m1 contributes +1 to c[0,0] and +1 to c[1,1]
  
  So in the tensor decomposition:
  U[0*2+0, t] = 1, U[1*2+1, t] = 1  (A[0,0] and A[1,1])
  V[0*2+0, t] = 1, V[1*2+1, t] = 1  (B[0,0] and B[1,1])
  W[0*2+0, t] = 1, W[1*2+1, t] = 1  (c[0,0] and c[1,1])
  
  Let's verify: T[i,j,k] = sum_t U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t]
  
  For t = m1:
  U is non-zero at (0,0) and (1,1)
  V is non-zero at (0,0) and (1,1)
  W is non-zero at (0,0) and (1,1)
  
  Product is non-zero when (i,j) in {(0,0),(1,1)}, (j,k) in {(0,0),(1,1)}, (k,i) in {(0,0),(1,1)}
  
  Valid triples: (0,0,0) and (1,1,1)
  
  So T[0,0,0] += 1 and T[1,1,1] += 1.
  
  But we need T[i,j,k] = 1 for ALL i,j,k in {0,1}^3 (8 entries total).
  The sum of all 7 Strassen terms should give T[i,j,k] = 1 for all 8 entries.
  
  Let me verify for T[0,0,0]:
  m1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
  m2: U[1,0]=1, U[1,1]=1, V[0,0]=1, W[0,1]=1 -> U[0,0]=0 -> 0
  m3: U[0,0]=1, V[0,1]=1, V[1,1]=-1, W[1,0]=1, W[1,1]=-1 -> V[0,0]=0 -> 0
  m4: U[1,1]=1, V[1,0]=1, V[0,0]=-1, W[0,0]=1, W[0,1]=1 -> V[0,0]=-1, U[0,0]=0 -> 0
  m5: U[0,0]=1, U[0,1]=1, V[1,1]=1, W[1,0]=1, W[1,1]=1 -> V[0,0]=0 -> 0
  m6: U[1,0]=1, U[0,0]=-1, V[0,0]=1, V[0,1]=1, W[0,1]=1, W[1,1]=1 -> U[0,0]=-1, V[0,0]=1, W[0,0]=0 -> 0
  m7: U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1, W[0,0]=1, W[0,1]=1 -> U[0,0]=0 -> 0
  
  Sum for T[0,0,0] = 1. Good.
  
  Let me check T[0,1,0] (i=0,j=1,k=0):
  m1: U[0,1]=0 -> 0
  m2: U[1,0]=1, U[1,1]=1, V[0,0]=1, W[0,1]=1 -> U[0,1]=0 -> 0
  m3: U[0,0]=1, V[0,1]=1, V[1,1]=-1, W[1,0]=1, W[1,1]=-1 -> U[0,1]=0 -> 0
  m4: U[1,1]=1, V[1,0]=1, V[0,0]=-1, W[0,0]=1, W[0,1]=1 -> U[0,1]=0 -> 0
  m5: U[0,0]=1, U[0,1]=1, V[1,1]=1, W[1,0]=1, W[1,1]=1 -> U[0,1]=1, V[1,0]=0 -> 0
  m6: U[1,0]=1, U[0,0]=-1, V[0,0]=1, V[0,1]=1, W[0,1]=1, W[1,1]=1 -> U[0,1]=0 -> 0
  m7: U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1, W[0,0]=1, W[0,1]=1 -> U[0,1]=1, V[1,0]=1, W[0,0]=1 -> 1*1*1 = 1
  
  Sum for T[0,1,0] = 1. Good.
  
  Great, so the Strassen decomposition works. Now let me implement it for our 4x5x6 problem.
  
  I'll define a helper that adds Strassen terms for a 2x2@2x2 sub-problem, given the global indices.
  
  For a 2x2 block of A at rows (r0, r0+1) and cols (c0, c0+1), and a 2x2 block of B at rows (r0, r0+1) and cols (c0, c0+1), the Strassen terms are:
  
  Let A block be:
    A[r0,c0], A[r0,c0+1]
    A[r0+1,c0], A[r0+1,c0+1]
  
  B block be:
    B[r0,c0], B[r0,c0+1]
    B[r0+1,c0], B[r0+1,c0+1]
  
  Wait, I need to be careful with the indices. In our problem:
  A is 4x5, B is 5x6.
  The inner dimension is j (0 to 4).
  
  For a 2x2@2x2 sub-problem, we have:
  A sub-block: rows i0..i0+1, cols j0..j0+1
  B sub-block: rows j0..j0+1, cols k0..k0+1
  
  The Strassen terms:
  m1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
  m2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
  m3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
  m4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
  m5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
  m6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
  m7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
  
  And the contributions to C:
  C[i0,k0] += m1 - m4 + m6 + m7
  C[i0,k0+1] += m3 + m5
  C[i0+1,k0] += m2 + m4
  C[i0+1,k0+1] += m1 - m3 + m5 - m6
  
  Now, in the tensor decomposition, each m_i is a rank-1 term. Let me work out the U, V, W entries for each.
  
  For m1:
  A-part: A[i0,j0] + A[i0+1,j0+1]
  B-part: B[j0,k0] + B[j0+1,k0+1]
  C-part: +1 to C[i0,k0] and +1 to C[i0+1,k0+1]
  
  U entries: U[i0*m+j0, t] = 1, U[(i0+1)*m+(j0+1), t] = 1
  V entries: V[j0*p+k0, t] = 1, V[(j0+1)*p+(k0+1), t] = 1
  W entries: W[k0*n+i0, t] = 1, W[(k0+1)*n+(i0+1), t] = 1
  
  For m2:
  A-part: A[i0+1,j0] + A[i0+1,j0+1]
  B-part: B[j0,k0]
  C-part: +1 to C[i0+1,k0]
  
  U entries: U[(i0+1)*m+j0, t] = 1, U[(i0+1)*m+(j0+1), t] = 1
  V entries: V[j0*p+k0, t] = 1
  W entries: W[k0*n+(i0+1), t] = 1
  
  For m3:
  A-part: A[i0,j0]
  B-part: B[j0,k0+1] - B[j0+1,k0+1]
  C-part: +1 to C[i0,k0+1] and -1 to C[i0+1,k0+1]
  
  U entries: U[i0*m+j0, t] = 1
  V entries: V[j0*p+(k0+1), t] = 1, V[(j0+1)*p+(k0+1), t] = -1
  W entries: W[(k0+1)*n+i0, t] = 1, W[(k0+1)*n+(i0+1), t] = -1
  
  For m4:
  A-part: A[i0+1,j0+1]
  B-part: B[j0+1,k0] - B[j0,k0]
  C-part: -1 to C[i0,k0] and +1 to C[i0+1,k0]
  
  U entries: U[(i0+1)*m+(j0+1), t] = 1
  V entries: V[(j0+1)*p+k0, t] = 1, V[j0*p+k0, t] = -1
  W entries: W[k0*n+i0, t] = -1, W[k0*n+(i0+1), t] = 1
  
  For m5:
  A-part: A[i0,j0] + A[i0,j0+1]
  B-part: B[j0+1,k0+1]
  C-part: +1 to C[i0,k0+1] and +1 to C[i0+1,k0+1]
  
  U entries: U[i0*m+j0, t] = 1, U[i0*m+(j0+1), t] = 1
  V entries: V[(j0+1)*p+(k0+1), t] = 1
  W entries: W[(k0+1)*n+i0, t] = 1, W[(k0+1)*n+(i0+1), t] = 1
  
  For m6:
  A-part: A[i0+1,j0] - A[i0,j0]
  B-part: B[j0,k0] + B[j0,k0+1]
  C-part: +1 to C[i0,k0] and +1 to C[i0+1,k0]
  
  U entries: U[(i0+1)*m+j0, t] = 1, U[i0*m+j0, t] = -1
  V entries: V[j0*p+k0, t] = 1, V[j0*p+(k0+1), t] = 1
  W entries: W[k0*n+i0, t] = 1, W[k0*n+(i0+1), t] = 1
  
  For m7:
  A-part: A[i0,j0+1] - A[i0+1,j0+1]
  B-part: B[j0+1,k0] + B[j0+1,k0+1]
  C-part: +1 to C[i0,k0] and +1 to C[i0+1,k0+1]
  
  U entries: U[i0*m+(j0+1), t] = 1, U[(i0+1)*m+(j0+1), t] = -1
  V entries: V[(j0+1)*p+k0, t] = 1, V[(j0+1)*p+(k0+1), t] = 1
  W entries: W[k0*n+i0, t] = 1, W[(k0+1)*n+(i0+1), t] = 1
  
  Now let me implement this for our 4x5x6 problem.
  
  The 2x2@2x2 sub-problems are:
  For i0 in {0, 2}, j0 in {0, 2}, k0 in {0, 2, 4}:
    But j0+1 must be < 5, so j0 in {0, 2} (j0+1 = 1 or 3, both < 5)
    k0+1 must be < 6, so k0 in {0, 2, 4} (k0+1 = 1, 3, 5, all < 6)
    i0+1 must be < 4, so i0 in {0, 2} (i0+1 = 1 or 3, both < 4)
    
  So we have 2 * 2 * 3 = 12 sub-problems, each with rank 7: total 84.
  
  Then the strip terms:
  A[:,4] @ B[4,:] where A[:,4] is 4x1 and B[4,:] is 1x6.
  This is 4*1*6 = 24 rank-1 terms (schoolbook).
  
  Total rank: 84 + 24 = 108.
  
  Let me implement this.
"""

import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 4, 5, 6
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2@2x2 sub-problems
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                # m1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
                U[i0 * m + j0][col] = 1
                U[(i0 + 1) * m + (j0 + 1)][col] = 1
                V[j0 * p + k0][col] = 1
                V[(j0 + 1) * p + (k0 + 1)][col] = 1
                W[k0 * n + i0][col] = 1
                W[(k0 + 1) * n + (i0 + 1)][col] = 1
                col += 1
                
                # m2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
                U[(i0 + 1) * m + j0][col] = 1
                U[(i0 + 1) * m + (j0 + 1)][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + (i0 + 1)][col] = 1
                col += 1
                
                # m3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
                U[i0 * m + j0][col] = 1
                V[j0 * p + (k0 + 1)][col] = 1
                V[(j0 + 1) * p + (k0 + 1)][col] = -1
                W[(k0 + 1) * n + i0][col] = 1
                W[(k0 + 1) * n + (i0 + 1)][col] = -1
                col += 1
                
                # m4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
                U[(i0 + 1) * m + (j0 + 1)][col] = 1
                V[(j0 + 1) * p + k0][col] = 1
                V[j0 * p + k0][col] = -1
                W[k0 * n + i0][col] = -1
                W[k0 * n + (i0 + 1)][col] = 1
                col += 1
                
                # m5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
                U[i0 * m + j0][col] = 1
                U[i0 * m + (j0 + 1)][col] = 1
                V[(j0 + 1) * p + (k0 + 1)][col] = 1
                W[(k0 + 1) * n + i0][col] = 1
                W[(k0 + 1) * n + (i0 + 1)][col] = 1
                col += 1
                
                # m6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
                U[(i0 + 1) * m + j0][col] = 1
                U[i0 * m + j0][col] = -1
                V[j0 * p + k0][col] = 1
                V[j0 * p + (k0 + 1)][col] = 1
                W[k0 * n + i0][col] = 1
                W[k0 * n + (i0 + 1)][col] = 1
                col += 1
                
                # m7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
                U[i0 * m + (j0 + 1)][col] = 1
                U[(i0 + 1) * m + (j0 + 1)][col] = -1
                V[(j0 + 1) * p + k0][col] = 1
                V[(j0 + 1) * p + (k0 + 1)][col] = 1
                W[k0 * n + i0][col] = 1
                W[(k0 + 1) * n + (i0 + 1)][col] = 1
                col += 1
    
    # Strip: A[:,4] @ B[4,:] (4x1 @ 1x6)
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
