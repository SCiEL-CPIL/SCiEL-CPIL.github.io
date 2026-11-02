"""Matrix multiplication 2,2,13 using polynomial ring decomposition."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the (2,2,13) matrix multiplication tensor.
    
    Key idea: Treat the 2x13 matrix as a 2x2 matrix over the ring
    Q[x]/(x^13) where each entry is a polynomial of degree < 13.
    
    The (2,2) matrix multiplication tensor over a ring has rank 7
    (Strassen's algorithm). Each of the 7 multiplication terms involves
    polynomial multiplication in Q[x]/(x^13), which has rank 13.
    
    Total rank: 7 * 13 = 91
    """
    n, m, p = DIMENSIONS
    rank = 7 * p  # 91
    
    # U: shape (n*m, rank) = (4, 91)
    # V: shape (m*p, rank) = (26, 91)  
    # W: shape (p*n, rank) = (26, 91)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 matrix product:
    # M1 = (a+b)(e+f)  -> contributes to c11, c22
    # M2 = (a+b)h      -> contributes to c12
    # M3 = d(e-f)      -> contributes to c21
    # M4 = a+c         -> wait, let me use standard Strassen:
    #
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = (a+b)e
    # M4 = (c+d)(g+f)
    # M5 = (a-d)(e+f)
    # M6 = (b-c)(g+h)
    # M7 = (a+c)(e-g)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M7
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M5 + M6
    #
    # But we need this over polynomials. Each "multiplication" like (a+d)(e+h)
    # is a polynomial multiplication in Q[x]/(x^13).
    #
    # Let's define the 7 bilinear forms:
    # U coefficients (for A matrix, indexed by row i, col j):
    # V coefficients (for B matrix, indexed by row j, col k):
    # W coefficients (for C matrix, indexed by col k, row i):
    
    # Define the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict from (i,j) to coefficient
    # V_coeffs: dict from (j,k) to coefficient  
    # W_coeffs: dict from (k,i) to coefficient
    
    strassen_terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # Contributes: C00 += M1, C11 += M1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    strassen_terms.append((u1, v1, w1))
    
    # M2 = (A10 + A11)B10
    # Contributes: C10 += M2, C11 -= M2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(1,0): 1}
    w2 = {(0,1): 1, (1,1): -1}
    strassen_terms.append((u2, v2, w2))
    
    # M3 = (A00 + A01)B00
    # Contributes: C00 += M3, C01 += M3
    u3 = {(0,0): 1, (0,1): 1}
    v3 = {(0,0): 1}
    w3 = {(0,0): 1, (1,0): 1}
    strassen_terms.append((u3, v3, w3))
    
    # M4 = (A10 + A11)(B10 + B01)
    # Contributes: C10 -= M4, C01 -= M4
    u4 = {(1,0): 1, (1,1): 1}
    v4 = {(1,0): 1, (0,1): 1}
    w4 = {(0,1): -1, (1,0): -1}
    strassen_terms.append((u4, v4, w4))
    
    # M5 = (A00 - A11)(B00 + B01)
    # Contributes: C00 -= M5, C11 += M5
    u5 = {(0,0): 1, (1,1): -1}
    v5 = {(0,0): 1, (0,1): 1}
    w5 = {(0,0): -1, (1,1): 1}
    strassen_terms.append((u5, v5, w5))
    
    # M6 = (A01 - A10)(B11 + B00)
    # Contributes: C10 += M6, C11 += M6
    u6 = {(0,1): 1, (1,0): -1}
    v6 = {(1,1): 1, (0,0): 1}
    w6 = {(0,1): 1, (1,1): 1}
    strassen_terms.append((u6, v6, w6))
    
    # M7 = (A00 + A10)(B00 - B10)
    # Contributes: C00 -= M7, C01 += M7
    u7 = {(0,0): 1, (1,0): 1}
    v7 = {(0,0): 1, (1,0): -1}
    w7 = {(0,0): -1, (1,0): 1}
    strassen_terms.append((u7, v7, w7))
    
    # Now fill in U, V, W for each of the 7 terms.
    # Each term involves polynomial multiplication of degree < 13.
    # For each term s (0-6), we have 13 columns (one for each degree).
    
    for s, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
        for deg in range(p):
            col = s * p + deg
            
            # Fill U: for each (i,j) in u_dict, U[i*m+j][col] += coeff * (degree indicator)
            # The degree indicator for polynomial multiplication: 
            # When multiplying polynomials, the coefficient of x^deg in the product
            # involves all pairs (d1,d2) where d1+d2 = deg (mod 13).
            # But in the tensor decomposition, we need to express this as
            # sum of rank-1 terms. For each degree shift, we add a contribution.
            
            # Actually, let me think about this differently.
            # The polynomial multiplication (f * g)[deg] = sum_{a+b=deg mod 13} f[a]*g[b]
            # This can be written as sum over all (a,b) with a+b=deg mod 13 of f[a]*g[b]
            
            # So for each degree, we need to add contributions from all valid (a,b) pairs.
            # But that would make the rank much larger.
            
            # Alternative: Use the fact that polynomial multiplication in Q[x]/(x^13)
            # can be decomposed into 13 rank-1 terms using the cyclic convolution structure.
            
            # For each s and deg, we add one column. In that column:
            # - U gets the u_dict coefficients (constant across degrees)
            # - V gets the v_dict coefficients with a degree shift
            # - W gets the w_dict coefficients with the output degree
            
            # Specifically, for column (s, deg):
            # U[i*m+j] = u_dict.get((i,j), 0)
            # V[j*p+k] = v_dict.get((j,k), 0) if the "input degree" for V is k
            # W[k*n+i] = w_dict.get((k,i), 0) if the "output degree" is deg
            
            # But this doesn't capture the convolution properly.
            
            # Let me use a cleaner approach:
            # For polynomial multiplication, we can write:
            # (f*g)[d] = sum_{a=0}^{12} f[a] * g[(d-a) mod 13]
            # 
            # This is a sum of 13 rank-1 terms per degree, giving 13*13 = 169 per Strassen term.
            # That's too much.
            
            # Better: Recognize that the entire polynomial multiplication tensor
            # T_poly[d, a, b] = 1 if (a+b) mod 13 == d else 0
            # has rank 13 (it's a cyclic convolution).
            
            # The rank-1 decomposition of the cyclic convolution:
            # T_poly[d,a,b] = sum_{t=0}^{12} omega^{t(d-a-b)} where omega is a 13th root of unity
            # But we need rational/integer coefficients.
            
            # Over Q, the cyclic convolution of length 13 has rank 13, and we can use
            # the fact that it's a circulant matrix. The rank-1 decomposition over Q
            # requires 13 terms.
            
            # For our purposes, let's just use the direct decomposition:
            # For each (s, deg), we'll have multiple columns to handle the convolution.
            
            pass  # This approach is getting complicated. Let me simplify.
    
    # Simplified approach: Use the direct convolution decomposition.
    # For each Strassen term s, and for each output degree d, and each pair (a,b) with (a+b)%13==d,
    # we add a rank-1 contribution.
    
    # But that gives 7 * 13 * 13 = 1183 columns, which is worse than 91.
    
    # Let me go back to the original insight: the rank of polynomial multiplication
    # in Q[x]/(x^13) is exactly 13. So 7 * 13 = 91 should work.
    
    # The key is that we can decompose the convolution tensor into 13 rank-1 terms.
    # For a prime length 13, we can use:
    # T[d,a,b] = delta_{(a+b) mod 13, d}
    # 
    # One decomposition: for t = 0, ..., 12:
    # T_t[d,a,b] = 1 if a = t and b = (d - t) mod 13, else 0
    # But that's 13 * 13 = 169 terms, not 13.
    
    # Actually, the rank of the convolution tensor for length n is n.
    # The decomposition is:
    # T[d,a,b] = sum_{t=0}^{n-1} T_t[d,a,b]
    # where T_t[d,a,b] = 1 if a=t and b=(d-t) mod n, else 0
    # That's n^2 terms, which is wrong.
    
    # I'm overcomplicating this. Let me just use the schoolbook for the polynomial part
    # but only for the 7 Strassen terms.
    
    # Total rank: 7 * 13 * 13 = 1183? No, that's not right either.
    
    # Let me start fresh with a clean construction.
    
    # The multiplication tensor for (n,m,p) is T[i,j,k] = delta.
    # We want sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = T[i,j,k]
    
    # For (2,2,13), we can group the k index into 13 "degree" slots.
    # For each degree d (0-12), we have a 2x2 matrix multiplication problem
    # where the B matrix is a 2x2 matrix of polynomials evaluated at that degree.
    
    # But the polynomial multiplication couples all degrees.
    
    # Cleanest approach: 
    # 1. Use Strassen's 7-term decomposition for the 2x2 matrix structure.
    # 2. Each term involves multiplying two "vectors" of length 13 (the polynomial coefficients).
    # 3. Vector multiplication (dot product) of length 13 has rank 13.
    # 4. Total: 7 * 13 = 91.
    
    # Let me implement this properly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each Strassen term s and each degree t (0-12):
    # Column index = s * 13 + t
    # 
    # U[i*m+j, col] = coefficient of A[i,j] in the s-th Strassen form
    # V[j*p+k, col] = coefficient of B[j,k] in the s-th Strassen form, times delta(k == t)
    # W[k*n+i, col] = coefficient of C[i,k] in the s-th Strassen form, times delta(k == t)
    
    # Wait, this doesn't work because the polynomial multiplication mixes degrees.
    
    # Let me try a different decomposition of the polynomial multiplication.
    # For two polynomials f(x) and g(x) of degree < 13, the product h(x) = f(x)*g(x) mod x^13
    # has h[d] = sum_{a+b=d, a+b<13} f[a]*g[b]
    
    # This is a linear operation. The tensor for this operation has entries:
    # T_poly[d, a, b] = 1 if a+b=d and d<13, else 0
    # 
    # The rank of this tensor: it's the multiplication table of the truncated polynomial ring.
    # For degree < n, the rank is n (I believe).
    
    # A rank-1 decomposition: T_poly[d,a,b] = sum_{t=0}^{12} T_t[d,a,b]
    # where T_t[d,a,b] = 1 if a=t, b=d-t, and 0<=d-t<13, else 0
    # But that's still 13*13 = 169 non-zero entries in the decomposition.
    
    # I think the issue is that I'm confusing the tensor rank with the number of non-zero entries.
    
    # Let me just implement a valid decomposition, even if not minimal.
    # I'll use the schoolbook for the polynomial part combined with Strassen for the matrix part.
    
    # Total rank: 7 * 13 * 13 = 1183 (still better than 52? No, 52 < 1183)
    
    # OK, my initial approach was wrong. Let me go back to basics.
    
    # The schoolbook rank is 2*2*13 = 52.
    # To beat this, I need a rank < 52.
    
    # The Strassen approach for (2,2,p) gives rank 7*p = 91 for p=13, which is worse.
    
    # So for this specific case, the schoolbook might be optimal or close to it.
    
    # Let me just return the schoolbook decomposition.
    
    rank = n * m * p
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
