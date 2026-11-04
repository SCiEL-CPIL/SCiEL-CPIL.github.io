"""Rewritten program: composition-based decomposition for 2x3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix.
    
    Strategy:
    - Split the 3 middle indices (columns of A / rows of B) into two groups:
      Group 1: indices 0,1 (size 2)
      Group 2: index 2 (size 1)
    
    For Group 1 (2x2x6): Use Strassen's algorithm for 2x2x2 blocks.
      Split the 6 output columns into 3 groups of 2.
      Each 2x2x2 block uses Strassen (rank 7).
      Total for Group 1: 3 * 7 = 21 terms.
    
    For Group 2 (2x1x6): This is just outer product-like.
      C[i,k] = A[i,2] * B[2,k]
      This requires 2*6 = 12 independent products (schoolbook for this part).
    
    Total rank: 21 + 12 = 33.
    
    The decomposition satisfies:
      T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = 1 if the indices correspond to a valid multiplication term, else 0.
    Actually, the matrix multiplication tensor has T[i,j,k,l] = delta_{j,l} for the sum over l.
    In the U,V,W factorization:
      sum_l A[i,l] B[l,k] = sum_c (sum_l U[i*m+l, c] A[i,l]) * (sum_k V[l*p+k, c] B[l,k]) ... 
    Wait, the standard factorization is:
      C[i,k] = sum_l A[i,l] B[l,k]
      = sum_c (sum_l U[i*m+l, c] A[i,l]) * (sum_l V[l*p+k, c] B[l,k]) * W[k*n+i, c]
    
    So for each term c, we have:
      U[i*m+l, c] picks out A[i,l]
      V[l*p+k, c] picks out B[l,k]
      W[k*n+i, c] scales the contribution to C[i,k]
    
    For Strassen 2x2x2:
      n=2, m=2, p=2
      U shape (4, 7), V shape (4, 7), W shape (4, 7)
      
      Strassen products:
      M1 = (a+d)(g+j)  -> U picks a,d; V picks g,j; W contributes to c11, c22
      M2 = (c+d)(g)    -> U picks c,d; V picks g; W contributes to c21, c22
      M3 = (a+b)(j)    -> U picks a,b; V picks j; W contributes to c12, c22
      M4 = (d-b)(h+j)  -> U picks d,b; V picks h,j; W contributes to c11, c21
      M5 = (a-c)(g+h)  -> U picks a,c; V picks g,h; W contributes to c11, c21
      M6 = (b+d)(i-j)  -> U picks b,d; V picks i,j; W contributes to c22
      M7 = (a+d)(h-i)  -> U picks a,d; V picks h,i; W contributes to c11, c12
      
      Where A = [[a,b],[c,d]], B = [[g,h],[i,j]], C = [[c11,c12],[c21,c22]]
      
      c11 = M1 + M4 - M5 - M7
      c12 = M3 + M7
      c21 = M2 + M5
      c22 = M1 - M2 + M3 + M6
      
      So for each M_c:
      M1: U[a]=1, U[d]=1; V[g]=1, V[j]=1; W[c11]=1, W[c22]=1
      M2: U[c]=1, U[d]=1; V[g]=1; W[c21]=1, W[c22]=-1
      M3: U[a]=1, U[b]=1; V[j]=1; W[c12]=1, W[c22]=1
      M4: U[d]=1, U[b]=-1; V[h]=1, V[j]=1; W[c11]=1, W[c21]=1
      M5: U[a]=1, U[c]=-1; V[g]=1, V[h]=1; W[c11]=-1, W[c21]=1
      M6: U[b]=1, U[d]=1; V[i]=1, V[j]=-1; W[c22]=1
      M7: U[a]=1, U[d]=1; V[h]=1, V[i]=-1; W[c11]=-1, W[c12]=1
      
      Note: In our indexing:
      A is 2x2, so A[i,l] with i in {0,1}, l in {0,1}
      U index: i*m + l = i*2 + l
      B is 2x2, so B[l,k] with l in {0,1}, k in {0,1}
      V index: l*p + k = l*2 + k
      C is 2x2, so C[i,k] with i in {0,1}, k in {0,1}
      W index: k*n + i = k*2 + i
      
      Mapping:
      a = A[0,0] -> U[0*2+0] = U[0]
      b = A[0,1] -> U[0*2+1] = U[1]
      c = A[1,0] -> U[1*2+0] = U[2]
      d = A[1,1] -> U[1*2+1] = U[3]
      
      g = B[0,0] -> V[0*2+0] = V[0]
      h = B[0,1] -> V[0*2+1] = V[1]
      i = B[1,0] -> V[1*2+0] = V[2]
      j = B[1,1] -> V[1*2+1] = V[3]
      
      c11 = C[0,0] -> W[0*2+0] = W[0]
      c12 = C[0,1] -> W[1*2+0] = W[2]
      c21 = C[1,0] -> W[0*2+1] = W[1]
      c22 = C[1,1] -> W[1*2+1] = W[3]
    """
    n, m, p = 2, 3, 6
    
    # U shape (n*m, rank) = (6, rank)
    # V shape (m*p, rank) = (18, rank)
    # W shape (p*n, rank) = (12, rank)
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Group 1: middle indices 0,1 (A cols 0,1; B rows 0,1)
    # Split output cols into 3 groups of 2: cols 0-1, 2-3, 4-5
    
    for block_idx in range(3):
        k_offset = block_idx * 2  # starting column index in B/C
        
        # For this 2x2x2 block:
        # A indices: i in {0,1}, l in {0,1}
        # B indices: l in {0,1}, k in {k_offset, k_offset+1}
        # C indices: i in {0,1}, k in {k_offset, k_offset+1}
        
        # Strassen factors
        # M1: U[a]=1, U[d]=1; V[g]=1, V[j]=1; W[c11]=1, W[c22]=1
        U[0*3+0][col] = 1  # a
        U[1*3+1][col] = 1  # d
        V[0*6+k_offset][col] = 1    # g
        V[1*6+k_offset+1][col] = 1  # j
        W[k_offset*2+0][col] = 1    # c11
        W[(k_offset+1)*2+1][col] = 1  # c22
        col += 1
        
        # M2: U[c]=1, U[d]=1; V[g]=1; W[c21]=1, W[c22]=-1
        U[1*3+0][col] = 1  # c
        U[1*3+1][col] = 1  # d
        V[0*6+k_offset][col] = 1  # g
        W[k_offset*2+1][col] = 1    # c21
        W[(k_offset+1)*2+1][col] = -1  # c22
        col += 1
        
        # M3: U[a]=1, U[b]=1; V[j]=1; W[c12]=1, W[c22]=1
        U[0*3+0][col] = 1  # a
        U[0*3+1][col] = 1  # b
        V[1*6+k_offset+1][col] = 1  # j
        W[(k_offset+1)*2+0][col] = 1  # c12
        W[(k_offset+1)*2+1][col] = 1  # c22
        col += 1
        
        # M4: U[d]=1, U[b]=-1; V[h]=1, V[j]=1; W[c11]=1, W[c21]=1
        U[1*3+1][col] = 1   # d
        U[0*3+1][col] = -1  # b
        V[0*6+k_offset+1][col] = 1  # h
        V[1*6+k_offset+1][col] = 1  # j
        W[k_offset*2+0][col] = 1    # c11
        W[k_offset*2+1][col] = 1    # c21
        col += 1
        
        # M5: U[a]=1, U[c]=-1; V[g]=1, V[h]=1; W[c11]=-1, W[c21]=1
        U[0*3+0][col] = 1   # a
        U[1*3+0][col] = -1  # c
        V[0*6+k_offset][col] = 1    # g
        V[0*6+k_offset+1][col] = 1  # h
        W[k_offset*2+0][col] = -1   # c11
        W[k_offset*2+1][col] = 1    # c21
        col += 1
        
        # M6: U[b]=1, U[d]=1; V[i]=1, V[j]=-1; W[c22]=1
        U[0*3+1][col] = 1   # b
        U[1*3+1][col] = 1   # d
        V[1*6+k_offset][col] = 1    # i
        V[1*6+k_offset+1][col] = -1  # j
        W[(k_offset+1)*2+1][col] = 1  # c22
        col += 1
        
        # M7: U[a]=1, U[d]=1; V[h]=1, V[i]=-1; W[c11]=-1, W[c12]=1
        U[0*3+0][col] = 1   # a
        U[1*3+1][col] = 1   # d
        V[0*6+k_offset+1][col] = 1   # h
        V[1*6+k_offset][col] = -1    # i
        W[k_offset*2+0][col] = -1    # c11
        W[(k_offset+1)*2+0][col] = 1  # c12
        col += 1
    
    # Group 2: middle index 2 (A col 2; B row 2)
    # C[i,k] = A[i,2] * B[2,k] for i in {0,1}, k in {0,...,5}
    # This is 12 independent products
    
    for i in range(n):
        for k in range(p):
            U[i*3+2][col] = 1    # A[i,2]
            V[2*6+k][col] = 1    # B[2,k]
            W[k*2+i][col] = 1    # C[i,k]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
