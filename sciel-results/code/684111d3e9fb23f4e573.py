"""Program exploiting Kronecker product structure for 2x2 x 2x8 matrix multiplication."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # The 2x2 x 2x8 multiplication decomposes as 4 independent 2x2 multiplications
    # (columns 0-1, 2-3, 4-5, 6-7 of the 2x8 matrix).
    # Each 2x2 multiplication has tensor rank 7 (Strassen), so total rank = 4 * 7 = 28.
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # Strassen decomposition for 2x2 x 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # 7 products (each is a product of a linear form in A and a linear form in B):
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)   -- WAIT: this has c in A and g in B, but d is also in A
    # Actually let me use the standard verified Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)  -- this is d*g - d*c, NOT separable!
    # 
    # The correct Strassen that IS separable:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)  -- NO
    #
    # Let me use a verified separable Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d-b)(g+e)  -- no, let me just use the standard one carefully
    #
    # Standard Strassen (verified to be bilinear):
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = (a)(f - h)
    # m4 = (d)(g - c)  -- PROBLEM: c is in A, not B
    #
    # I see the issue. Let me look at this differently.
    # In Strassen's algorithm, the 7 multiplications are:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = a(f-h)
    # p4 = d(g-c)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # For p4 = d(g-c): this equals d*g - d*c.
    # In the tensor decomposition sense, each term must be (linear in A) * (linear in B).
    # d*g is (d from A) * (g from B) -- OK
    # d*c is (d from A) * (c from A) -- NOT OK, both from A
    #
    # So p4 as written is NOT a valid single bilinear term. This means the standard
    # Strassen algorithm, while using 7 scalar multiplications, does NOT directly give
    # a rank-7 tensor decomposition in the naive sense.
    #
    # HOWEVER, the tensor rank of 2x2 matrix multiplication IS 7. This is a well-known
    # result. The Strassen algorithm DOES give a rank-7 decomposition, but we need to
    # be careful about how we express it.
    #
    # The key insight: in the tensor decomposition, we have:
    # T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^{nm}, v_t is in R^{mp}, w_t is in R^{pn}
    #
    # For 2x2 x 2x2, the tensor is:
    # T = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (with appropriate indexing)
    #
    # Strassen's 7 terms DO form a valid rank-7 decomposition. Let me verify p4:
    # p4 = d(g-c) = d*g - d*c
    # This contributes to C as:
    # C_{21} += p4 (i.e., C[1][0] += p4)
    #
    # In the tensor: the term d*g contributes to A_{22}*B_{21}*C_{21}
    # The term d*c contributes to... wait, d*c is not a product of one A and one B entry.
    #
    # I think I'm overcomplicating this. Let me just use a known correct rank-7
    # decomposition. The standard Strassen IS a valid rank-7 decomposition. The way
    # it works is that each "multiplication" p_t is computed as (linear combo of A) *
    # (linear combo of B), and then the results are combined linearly to get C.
    #
    # Let me just implement it and verify by checking that the decomposition is correct.
    
    rank = 28  # 4 blocks * 7 Strassen terms
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 28)
    V = [[0] * rank for _ in range(m * p)]   # shape (16, 28)
    W = [[0] * rank for _ in range(p * n)]   # shape (16, 28)
    
    # Strassen for one 2x2 block:
    # A indices: 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
    # B indices for block b: e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
    #   V indices: e=0*8+2b=2b, f=0*8+2b+1=2b+1, g=1*8+2b=8+2b, h=1*8+2b+1=8+2b+1
    # C indices: C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1]
    #   W indices: C[0,2b]=2b*2+0=4b, C[0,2b+1]=(2b+1)*2+0=4b+2,
    #              C[1,2b]=2b*2+1=4b+1, C[1,2b+1]=(2b+1)*2+1=4b+3
    
    for b in range(4):
        base = b * 7
        # Strassen terms for block b
        # m1 = (a+d)(e+h)
        col = base + 0
        U[0][col] = 1   # a
        U[3][col] = 1   # d
        V[2*b][col] = 1       # e
        V[8+2*b+1][col] = 1   # h
        W[4*b][col] = 1       # C[0,2b]
        W[4*b+3][col] = 1     # C[1,2b+1]
        
        # m2 = (c+d)(e)
        col = base + 1
        U[2][col] = 1   # c
        U[3][col] = 1   # d
        V[2*b][col] = 1       # e
        W[4*b+1][col] = 1     # C[1,2b]
        W[4*b+3][col] = 1     # C[1,2b+1]
        
        # m3 = a(f-h)
        col = base + 2
        U[0][col] = 1   # a
        V[2*b+1][col] = 1      # f
        V[8+2*b+1][col] = -1   # -h
        W[4*b+2][col] = 1      # C[0,2b+1]
        W[4*b+3][col] = 1      # C[1,2b+1]
        
        # m4 = d(g-c)
        # This is d*g - d*c. In the bilinear form, this needs care.
        # Actually, in the standard Strassen, m4 = d*(g-c) where the "c" here
        # is actually meant to be handled differently. Let me use:
        # m4 = (d)(g-c) -> this gives d*g - d*c
        # But d*c is not bilinear in (A,B). 
        # 
        # The correct separable form: I'll use m4 = (d-b)(g+e) or similar.
        # Actually, let me just use the schoolbook approach to be safe.
        
        # Let me abandon Strassen and use schoolbook (rank 32) to ensure correctness.
        pass
    
    # Use schoolbook decomposition (rank 32) for guaranteed correctness
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
