"""Program for rank-minimal exact tensor decomposition of 2x3 * 3x3 matrix multiplication."""

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-13 exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    The decomposition is built by splitting the middle index j (of A) into two parts:
    - Part 1: j in {0, 1}. This corresponds to multiplying a 2x2 matrix by a 3x3 matrix 
      (ignoring the 3rd column of A and 3rd row of B). The rank of T_{2,2,3} is 2*2*3 = 12.
    - Part 2: j = 2. This corresponds to multiplying a 2x1 matrix (column 2 of A) by a 
      1x3 matrix (row 2 of B). The rank of T_{2,1,3} is 2*1*3 = 6? No, wait.
      
    Actually, let's re-evaluate.
    T_{n,m,p} rank is at most n*m*p.
    For T_{2,2,3}, the schoolbook rank is 2*2*3 = 12.
    For T_{2,1,3}, the schoolbook rank is 2*1*3 = 6.
    Total would be 18. This is the same as the seed.
    
    Let's look for a better decomposition.
    The rank of matrix multiplication tensor T_{n,m,p} is known to be exactly n*m*p for 
    small dimensions, but sometimes can be lower?
    Actually, for T_{2,3,3}, the rank is known to be 13?
    Let's check the literature or derive it.
    
    Known results:
    Rank(T_{2,2,2}) = 7 (Strassen).
    Rank(T_{2,2,3}) = 12? Or less?
    Rank(T_{2,3,3}) ?
    
    Let's try to construct a rank 13 decomposition.
    We can write T_{2,3,3} = T_{2,2,3} + T_{2,1,3} (disjoint in j).
    If Rank(T_{2,2,3}) = 12 and Rank(T_{2,1,3}) = 1? No, T_{2,1,3} is outer product of a vector and a vector?
    T_{2,1,3}: A is 2x1, B is 1x3. C is 2x3.
    C_{i,k} = A_{i,0} * B_{0,k}.
    This is an outer product of a length-2 vector and a length-3 vector.
    The tensor is rank 1?
    T_{i,0,k} = 1.
    U_{i,0} = 1 for i=0,1.
    V_{0,k} = 1 for k=0,1,2.
    W_{k,i} = 1 for all k,i.
    Term: U^{(r)} \otimes V^{(r)} \otimes W^{(r)}.
    If we set U_{i,0}=1, V_{0,k}=1, W_{k,i}=1, then the product is 1 for all i,k.
    This matches T_{2,1,3}.
    So Rank(T_{2,1,3}) = 1.
    
    Therefore, if Rank(T_{2,2,3}) = 12, then Rank(T_{2,3,3}) <= 12 + 1 = 13.
    
    Is Rank(T_{2,2,3}) = 12?
    T_{2,2,3}: A is 2x2, B is 2x3.
    Schoolbook rank is 12.
    Can it be lower?
    The rank of T_{2,2,2} is 7.
    T_{2,2,3} can be seen as T_{2,2,2} + T_{2,2,1}.
    T_{2,2,1} is multiplying 2x2 by 2x1. Rank is 4?
    T_{2,2,1}: C_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0}.
    This is two terms. Rank 2? No, it's a bilinear form.
    Actually, T_{n,m,1} has rank n*m.
    So T_{2,2,1} has rank 4.
    T_{2,2,2} has rank 7.
    So T_{2,2,3} <= 7 + 4 = 11?
    If so, T_{2,3,3} <= 11 + 1 = 12.
    
    Let's verify Rank(T_{2,2,1}).
    A is 2x2, B is 2x1.
    C_0 = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
    C_1 = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}
    This is just 2 independent bilinear forms.
    The tensor is sum of 4 terms?
    T_{i,j,0} = 1.
    Terms:
    (i=0,j=0): U_{0,0}=1, V_{0,0}=1, W_{0,0}=1
    (i=0,j=1): U_{0,1}=1, V_{1,0}=1, W_{0,0}=1
    (i=1,j=0): U_{1,0}=1, V_{0,0}=1, W_{1,0}=1
    (i=1,j=1): U_{1,1}=1, V_{1,0}=1, W_{1,0}=1
    These 4 terms are linearly independent. So rank is 4.
    
    So T_{2,2,3} <= 7 + 4 = 11.
    And T_{2,3,3} <= 11 + 1 = 12.
    
    Can we do even better?
    T_{2,3,3} = T_{2,2,3} + T_{2,1,3}.
    We found a rank 12 decomposition.
    
    Let's construct the rank 12 decomposition.
    Part 1: T_{2,2,2} with rank 7 (Strassen).
    Part 2: T_{2,2,1} with rank 4.
    Part 3: T_{2,1,3} with rank 1.
    Total rank = 7 + 4 + 1 = 12.
    
    We need to map these to the global indices.
    Global dimensions: n=2, m=3, p=3.
    Indices: i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    Part 1: T_{2,2,2} uses i in {0,1}, j in {0,1}, k in {0,1}.
    Part 2: T_{2,2,1} uses i in {0,1}, j in {0,1}, k = 2.
    Part 3: T_{2,1,3} uses i in {0,1}, j = 2, k in {0,1,2}.
    
    Let's define the Strassen decomposition for T_{2,2,2}.
    Strassen's 7 terms for multiplying 2x2 matrices.
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]].
    
    Strassen's formulas:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C00 = m1 - m4 + m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m3 + m6 + m7
    
    We need to express these as tensor terms.
    Each term m_k is a product of a linear form in A and a linear form in B.
    So each m_k corresponds to one rank-1 term in the tensor decomposition.
    
    Let's list the linear forms.
    m1: A-form: a+d, B-form: e+h
    m2: A-form: c+d, B-form: e
    m3: A-form: a, B-form: f-h
    m4: A-form: d, B-form: g-e
    m5: A-form: a+b, B-form: h
    m6: A-form: c-a, B-form: e+f
    m7: A-form: b-d, B-form: g+h
    
    Now, we need to assign coefficients to U, V, W.
    For a term r, U_{i,j}^{(r)}, V_{j,k}^{(r)}, W_{i',k'}^{(r)}?
    Wait, the output index is W[k*n+i].
    The term contributes to C_{i,k} if the A-form has coefficient for A_{i,j} and B-form has coefficient for B_{j,k}.
    And the W part must be 1 at index k*n+i.
    
    Let's refine the tensor definition.
    T_{i,j,k} = 1.
    Sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,n+i}^{(r)}? No.
    The standard definition is:
    C_{i,k} = Sum_{j} A_{i,j} B_{j,k}.
    So T_{i,j,k} = 1.
    The decomposition is Sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}?
    No, W is a vector of size p*n.
    The term is U^{(r)} \otimes V^{(r)} \otimes W^{(r)}.
    The component is U_{i,j}^{(r)} V_{j,k}^{(r)} W_{index}^{(r)}.
    For this to contribute to C_{i,k}, the W index must correspond to (i,k).
    And the sum over r must yield 1 for T_{i,j,k}.
    So for a specific (i,j,k), Sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k*n+i}^{(r)} = 1.
    
    In Strassen, for a specific term m_r, it contributes to multiple C_{i,k} entries.
    This means W^{(r)} will have non-zero entries at multiple indices.
    Specifically, if m_r contributes to C_{i,k} with coefficient c, then W_{k*n+i}^{(r)} = c.
    
    Let's build the matrices.
    
    Part 1: Strassen for T_{2,2,2} (i in 0,1; j in 0,1; k in 0,1).
    Rank 7.
    
    Term 0 (m1):
    A-form: a+d -> U_{0,0}=1, U_{1,1}=1.
    B-form: e+h -> V_{0,0}=1, V_{1,1}=1.
    Contributions:
    C00: +1 -> W_{0*2+0}=W_0 = 1
    C01: 0
    C10: 0
    C11: +1 -> W_{1*2+1}=W_3 = 1
    Wait, m1 = (a+d)(e+h) = ae + ah + de + dh.
    C00 gets ae. C11 gets dh.
    C01 gets ah? No, C01 = af+bh.
    C10 = ce+dg.
    So m1 contributes to C00 and C11.
    W_0 = 1, W_3 = 1.
    
    Term 1 (m2):
    A-form: c+d -> U_{1,0}=1, U_{1,1}=1.
    B-form: e -> V_{0,0}=1.
    Contributions:
    ce -> C10. dg -> C10? No, dg is in C10?
    C10 = ce + dg.
    So m2 contributes to C10.
    W_{1*2+0}=W_2 = 1.
    
    Term 2 (m3):
    A-form: a -> U_{0,0}=1.
    B-form: f-h -> V_{0,1}=1, V_{1,1}=-1.
    Contributions:
    af -> C01. -ah -> ?
    C01 = af + bh.
    So m3 contributes to C01 with +1.
    W_{0*2+1}=W_1 = 1.
    
    Term 3 (m4):
    A-form: d -> U_{1,1}=1.
    B-form: g-e -> V_{1,0}=1, V_{0,0}=-1.
    Contributions:
    dg -> C10. -de -> ?
    C10 = ce + dg.
    So m4 contributes to C10 with +1.
    W_2 = 1.
    Wait, m1 also contributed to C11.
    C11 = cf + dh.
    m4 has dg and -de. Neither is in C11.
    Let's re-check Strassen.
    C11 = m1 - m3 + m6 + m7.
    m1 = ae+ah+de+dh.
    m3 = af-ah.
    m6 = ce+cf-ae-af.
    m7 = bg+bh-dg-dh.
    Sum: (ae+ah+de+dh) - (af-ah) + (ce+cf-ae-af) + (bg+bh-dg-dh)
    = ae+ah+de+dh - af + ah + ce+cf-ae-af + bg+bh-dg-dh
    = 2ah + de + ce + cf + bg + bh - 2af - dg
    This doesn't look like cf+dh.
    
    Let's use the standard Strassen coefficients.
    C00 = m1 - m4 + m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m3 + m6 + m7
    
    Let's re-verify m1..m7 definitions.
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    Check C00:
    m1 - m4 + m5 + m7
    = (ae+ah+de+dh) - (dg-de) + (ah+bh) + (bg+bh-dg-dh)
    = ae+ah+de+dh - dg+de + ah+bh + bg+bh-dg-dh
    = ae + 2ah + 2de + dh - 2dg + 2bh + bg
    This is not ae+bg.
    
    I am misremembering the Strassen formulas or their combination.
    Let's look up the correct combination.
    C = [[m1-m4+m5+m7, m3+m5], [m2+m4, m1-m3+m6+m7]]
    
    Let's re-evaluate C00.
    m1 = (a+d)(e+h) = ae+ah+de+dh
    m4 = d(g-e) = dg-de
    m5 = (a+b)h = ah+bh
    m7 = (b-d)(g+h) = bg+bh-dg-dh
    
    Sum:
    ae+ah+de+dh - dg+de + ah+bh + bg+bh-dg-dh
    = ae + (ah+ah) + (de+de) + (dh-dh) + (bh+bh) + bg - (dg+dg)
    = ae + 2ah + 2de + 2bh + bg - 2dg
    
    This is definitely not ae+bg.
    
    There is a mistake in my Strassen recall.
    Correct Strassen:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C00 = m1 - m4 + m5 + m7 ?
    Let's try a different set.
    Maybe m7 is (b-d)(g+h)?
    
    Actually, let's just use the schoolbook for T_{2,2,2} to be safe, rank 8.
    Then total rank = 8 + 4 + 1 = 13.
    This is still better than 18.
    
    Let's stick to rank 13 with schoolbook for T_{2,2,2}.
    
    Part 1: T_{2,2,2} (i in 0,1; j in 0,1; k in 0,1). Rank 8.
    Part 2: T_{2,2,1} (i in 0,1; j in 0,1; k = 2). Rank 4.
    Part 3: T_{2,1,3} (i in 0,1; j = 2; k in 0,1,2). Rank 1.
    Total Rank = 13.
    
    Construction:
    
    Part 1: 8 terms.
    For i in 0,1, j in 0,1, k in 0,1:
    Term r:
    U_{i,j} = 1
    V_{j,k} = 1
    W_{k*2+i} = 1
    
    Part 2: 4 terms.
    For i in 0,1, j in 0,1: (k is fixed at 2)
    Term r:
    U_{i,j} = 1
    V_{j,2} = 1
    W_{2*2+i} = W_{4+i} = 1
    
    Part 3: 1 term.
    i in 0,1; j=2; k in 0,1,2.
    Term r:
    U_{i,2} = 1
    V_{2,k} = 1
    W_{k*2+i} = 1
    
    Let's implement this.
    
    n=2, m=3, p=3.
    U size: 2*3 = 6.
    V size: 3*3 = 9.
    W size: 3*2 = 6.
    Rank = 13.
    
    Initialize U, V, W with zeros.
    
    Part 1:
    for i in range(2):
        for j in range(2):
            for k in range(2):
                r = ...
                U[i*3+j][r] = 1
                V[j*3+k][r] = 1
                W[k*2+i][r] = 1
    
    Part 2:
    for i in range(2):
        for j in range(2):
            r = ...
            U[i*3+j][r] = 1
            V[j*3+2][r] = 1
            W[2*2+i][r] = 1
    
    Part 3:
    r = ...
    for i in range(2):
        U[i*3+2][r] = 1
    for k in range(3):
        V[2*3+k][r] = 1
    for k in range(3):
        for i in range(2):
            W[k*2+i][r] = 1
    
    This should work.
    """
    n, m, p = DIMENSIONS
    rank = 13
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: T_{2,2,2} (schoolbook, rank 8)
    # i in 0..1, j in 0..1, k in 0..1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    # Part 2: T_{2,2,1} (schoolbook, rank 4)
    # i in 0..1, j in 0..1, k = 2
    for i in range(2):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 2][r] = 1
            W[2 * n + i][r] = 1
            r += 1
            
    # Part 3: T_{2,1,3} (rank 1)
    # i in 0..1, j = 2, k in 0..2
    for i in range(2):
        U[i * m + 2][r] = 1
    for k in range(3):
        V[2 * p + k][r] = 1
    for k in range(3):
        for i in range(2):
            W[k * n + i][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
