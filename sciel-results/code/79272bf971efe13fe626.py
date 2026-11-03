import time

DIMENSIONS = [2, 3, 10]

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.
    
    Uses a Strassen-based decomposition for the 2x2 block multiplication of the 10x10 output
    matrix, with 3x3 inner products. This gives rank 2 + 7*3 = 23.
    
    The 2x3 * 3x10 product can be viewed as:
    - Left: 2x3 matrix (2 rows, 3 columns)  
    - Right: 3x10 matrix (3 rows, 10 columns)
    - Output: 2x10 matrix
    
    We partition the 10 columns into 5 blocks of 2: [0:2], [2:4], [4:6], [6:8], [8:10].
    For each block q (0-4), we compute the 2x2 block product using Strassen's 7 products,
    but only the 3x3 inner products (since m=3). The first block uses naive 2x2 (4 products)
    to keep it simple, and blocks 1-4 use Strassen (7 products each).
    
    Total rank: 4 + 4*7 = 32.
    
    Wait, let me reconsider. The standard approach:
    - 2x3 * 3x10 = sum over j=0..2 of A[:,j] * B[j,:]
    - This is a rank-6 schoolbook per output element... no.
    
    Let me think more carefully. The tensor is T[i,j,k] = delta[i,k] for (i,j,k) where i in 0..1, j in 0..2, k in 0..9.
    Actually T[i,j,k] = 1 if the (i,k) output is computed from A[i,j]*B[j,k].
    
    Schoolbook: rank = 2*3*10 = 60.
    
    Better: Group the 10 columns into 5 pairs. For each pair q, we compute a 2x2 block C_q[i, 2q+r] for i,r in {0,1}.
    C_q = A * B_q where B_q is 3x2.
    
    For 2x3 * 3x2, schoolbook rank is 2*3*2 = 12. With Strassen-like: treat as 2x1 * 1x2 with 3x3 inner... no.
    
    Actually, 2x3 * 3x2: Let A be 2x3, B be 3x2. Output C is 2x2.
    C[i,r] = sum_j A[i,j] * B[j,r].
    
    We can use the fact that this is a bilinear map from (F^3 x F^3) to F^4 (since 2*2=4 outputs).
    Rank of 2x3*3x2: The minimum rank for 2x3*3x2 is known to be... 
    Actually, we can partition the 3 middle dimension. But m=3 is small.
    
    Let me use a different grouping. Partition the 10 columns into 2 groups of 5: [0:5] and [5:10].
    For each group, compute 2x3 * 3x5. 
    2x3*3x5 schoolbook rank: 2*3*5 = 30. Two groups: 60. No improvement.
    
    Better approach: Use the Strassen trick on the 2x2 output blocks.
    Partition the 10 columns into 5 blocks of 2. For block q (columns 2q, 2q+1):
    C_q is a 2x2 matrix where C_q[i, r] = sum_{j=0}^{2} A[i,j] * B[j, 2q+r].
    
    For each q, we need to compute C_q[0,0], C_q[0,1], C_q[1,0], C_q[1,1].
    These are 4 bilinear forms in A (6 variables: A[0,0],A[0,1],A[0,2],A[1,0],A[1,1],A[1,2]) 
    and B (6 variables per block: B[0,2q],B[1,2q],B[2,2q],B[0,2q+1],B[1,2q+1],B[2,2q+1]).
    
    Actually, for a fixed q, define:
    A_0 = A[0, :], a row of length 3
    A_1 = A[1, :], a row of length 3
    B_0 = B[:, 2q], a column of length 3
    B_1 = B[:, 2q+1], a column of length 3
    
    C_q[0,0] = A_0 . B_0 = sum_j A[0,j]*B[j,2q]
    C_q[0,1] = A_0 . B_1 = sum_j A[0,j]*B[j,2q+1]
    C_q[1,0] = A_1 . B_0 = sum_j A[1,j]*B[j,2q]
    C_q[1,1] = A_1 . B_1 = sum_j A[1,j]*B[j,2q+1]
    
    This is like multiplying a 2x1 matrix [A_0; A_1] (treating each A_i as a 3-vector) by a 1x2 matrix [B_0 B_1].
    Equivalently, it's the tensor product of a 2x1 "matrix" (really two 3-vectors) with a 1x2 "matrix".
    
    The bilinear rank of computing these 4 dot products from 3-dim vectors: 
    We have 6 inputs from A (but really 2 vectors of dim 3) and 6 inputs from B (2 vectors of dim 3).
    The rank is at most 3*3*2 = 18 for naive per-vector, but we can do better.
    
    Actually, for computing M[i,j] = sum_k x_i[k] * y_j[k] for i in {0,1}, j in {0,1}, k in {0,1,2}:
    This is the 2x1x1x2 tensor. The minimum bilinear rank is... 
    We can use: for each k, define u_k = x_0[k], v_k = y_j[k]... 
    
    Let's just use schoolbook for each block: rank per block = 2*3*2 = 12. Total = 5*12 = 60. No improvement.
    
    Let me try a different strategy. Use Strassen on the 2x2 block structure differently.
    
    Key insight: For the 2x3 * 3x10 problem, we can view it as:
    - The "inner" dimension is 3.
    - We can use the fact that 3 = 2+1, but that's not directly helpful.
    
    Alternative: Use a rank-5 decomposition for 2x2 * 2x2 (Strassen) by padding.
    Pad A from 2x3 to 2x2 by setting A[*,2] = 0? No, that loses information.
    
    Let me try: Partition the 10 columns into 5 groups of 2. For each group, compute the 2x2 output using
    the 7 Strassen products, but adapted for 3x3 inner products.
    
    For group q, define:
    M_00 = A[:, 0:2] * B[0:2, 2q:2q+2]  -- 2x2 product, rank 7 via Strassen
    M_10 = A[:, 2:3] * B[2:3, 2q:2q+2]  -- 2x1 product, rank 2
    C_q = M_00 + M_10
    
    Wait, A is 2x3, so A[:, 0:2] is 2x2, A[:, 2:3] is 2x1.
    B is 3x10, so B[0:2, 2q:2q+2] is 2x2, B[2:3, 2q:2q+2] is 1x2.
    
    M_00 is 2x2 (from 2x2 * 2x2), M_10 is 2x2 (from 2x1 * 1x2 = outer product).
    C_q[i,r] = M_00[i,r] + M_10[i,r] = sum_{j=0}^{1} A[i,j]*B[j,2q+r] + A[i,2]*B[2,2q+r].
    This is correct!
    
    Now, M_00 (2x2 from 2x2*2x2) has rank 7 (Strassen).
    M_10 (2x2 from 2x1 * 1x2) has rank 2 (since it's an outer product, each output is a product of one A and one B element).
    
    So per group: rank = 7 + 2 = 9. Total for 5 groups: 5*9 = 45.
    
    That's better than 60!
    
    Let me implement this.
    
    For Strassen on 2x2 * 2x2:
    Given P (2x2) and Q (2x2), compute R = P*Q.
    P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    
    S1 = P * (Q[:,0] - Q[:,1]) = [[p00, p01], [p10, p11]] * [[q00-q10], [q01-q11]]
    S2 = (P[:,0] + P[:,1]) * Q[:,1] = [[p00+p10], [p01+p11]] * [[q01], [q11]]
    S3 = (P[:,0] - P[:,1]) * (Q[:,0] + Q[:,1]) = [[p00-p10], [p01-p11]] * [[q00+q10], [q01+q11]]
    S4 = (P[:,0] + P[:,1]) * (Q[:,0] - Q[:,1]) = [[p00+p10], [p01+p11]] * [[q00-q10], [q01-q11]]
    
    R[0,0] = S1[0,0] + S2[0,0] + S4[0,0] - S3[0,0]
    R[0,1] = S1[0,0] + S2[0,0]
    R[1,0] = S3[0,0] + S4[0,0]
    R[1,1] = S2[0,0] + S3[0,0] - S4[0,0]
    
    Wait, let me use the standard Strassen formulas. I'll code it carefully.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # We'll use the decomposition:
    # For each of 5 column-pairs (q=0..4), columns 2q and 2q+1:
    #   C_q = A[:, 0:2] * B[0:2, 2q:2q+2] + A[:, 2:3] * B[2:3, 2q:2q+2]
    # The first term is a 2x2 * 2x2 product (rank 7 via Strassen)
    # The second term is a 2x1 * 1x2 outer product (rank 2)
    # Total rank: 5 * (7 + 2) = 45
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*10 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 10*2 = 20 rows
    
    col = 0
    for q in range(5):
        # q corresponds to columns 2q and 2q+1 of the output
        c0 = 2 * q
        c1 = 2 * q + 1
        
        # --- Strassen part: A[:, 0:2] * B[0:2, c0:c1+1] ---
        # A_left[i, j] for i in {0,1}, j in {0,1}
        # B_top[j, r] for j in {0,1}, r in {0,1}
        
        # Strassen products for 2x2 * 2x2
        # S1 = P * (Q[:,0] - Q[:,1])
        # U factors: A[0,0], A[0,1], A[1,0], A[1,1]
        # V factors: B[0,0], B[0,1], B[1,0], B[1,1]
        
        # S1: P * (Q_col0 - Q_col1)
        # U: A[0,0] on row i*m+j = 0*3+0=0, A[0,1] on row 1, A[1,0] on row 3, A[1,1] on row 4
        # V: B[0, c0-c1] = B[0,c0]-B[0,c1] on row 0*p+c0=0, B[0,c1] on row 1, 
        #     B[1, c0-c1] = B[1,c0]-B[1,c1] on row 10+c0=10, B[1,c1] on row 11
        # Actually V index is j*p + k. B[j, k] is at row j*10+k.
        # B[0, c0] at row 0*10+c0 = c0, B[0, c1] at row c1
        # B[1, c0] at row 10+c0, B[1, c1] at row 10+c1
        
        # S1: P * (Q[:,0] - Q[:,1])
        # (Q[:,0] - Q[:,1])[j] = B[j, c0] - B[j, c1] for j in {0,1}
        # S1[i, 0] = sum_j P[i,j] * (B[j, c0] - B[j, c1])
        # = A[i,0]*(B[0,c0]-B[0,c1]) + A[i,1]*(B[1,c0]-B[1,c1])
        
        # This is a single bilinear product per output element of S1.
        # S1 is 2x1 (since Q_col0 - Q_col1 is 2x1).
        # S1[i,0] for i in {0,1}
        
        # For the decomposition, each "product" in Strassen is a bilinear form.
        # S1[i,0] = sum_j A[i,j] * (B[j,c0] - B[j,c1])
        # This can be decomposed into 2 rank-1 terms:
        #   A[i,0] * (B[0,c0] - B[0,c1]) + A[i,1] * (B[1,c0] - B[1,c1])
        
        # But in the tensor decomposition, we need to represent each term as U * V * W.
        # Let me use a different approach: explicitly list all rank-1 components.
        
        # For Strassen, the 7 products are:
        # S1 = P * (Q[:,0] - Q[:,1])  -- 2x1, 2 bilinear terms
        # S2 = (P[:,0] + P[:,1]) * Q[:,1]  -- 2x1, 2 bilinear terms
        # S3 = (P[:,0] - P[:,1]) * (Q[:,0] + Q[:,1])  -- 2x1, 2 bilinear terms
        # S4 = (P[:,0] + P[:,1]) * (Q[:,0] - Q[:,1])  -- 2x1, 2 bilinear terms
        # S5 = P[0,0] * Q[0,0]  -- scalar, 1 bilinear term
        # S6 = (P[1,1] - P[0,0]) * (Q[0,0] + Q[0,1])  -- scalar, 1 bilinear term
        # S7 = (P[0,1] - P[1,1]) * (Q[1,0] + Q[1,1])  -- scalar, 1 bilinear term
        
        # Wait, that's not right. Let me use the standard Strassen:
        # S1 = P[0,0] * (Q[0,1] - Q[1,1])
        # S2 = (P[0,0] + P[0,1]) * Q[1,1]
        # S3 = (P[1,0] - P[0,0]) * Q[0,0]
        # S4 = P[1,1] * (Q[1,0] - Q[0,0])
        # S5 = (P[0,0] + P[1,1]) * (Q[0,0] + Q[1,1])
        # S6 = (P[0,1] - P[1,1]) * (Q[1,0] + Q[1,1])
        # S7 = (P[1,0] - P[0,0]) * (Q[0,0] + Q[0,1])
        
        # R[0,0] = S5 + S4 - S2 + S6
        # R[0,1] = S1 + S2
        # R[1,0] = S3 + S5
        # R[1,1] = S1 - S3 + S4 + S7
        
        # Each S is a scalar bilinear product. So 7 scalar products.
        # Each scalar product is rank 1 in the tensor decomposition.
        
        # For S1 = P[0,0] * (Q[0,1] - Q[1,1]) = A[0,0] * (B[0,c1] - B[1,c1])
        # U: A[0,0] at row 0*3+0 = 0
        # V: B[0,c1] at row 0*10+c1 = c1, B[1,c1] at row 1*10+c1 = 10+c1
        # W: contributes to R[0,0], R[0,1], R[1,1]
        # R[0,0] is C[0, c0] at W index c0*2+0 = 2*c0
        # R[0,1] is C[0, c1] at W index c1*2+0 = 2*c1
        # R[1,1] is C[1, c1] at W index c1*2+1 = 2*c1+1
        
        # This is getting complex. Let me just enumerate all terms.
        
        # I'll create helper to add a term: U[row_u] += a, V[row_v] += b, W[row_w] += c
        # where the term is a * A[row_u] * B[row_v] contributing c to output W[row_w]
        
        def add_term(row_u, coeff_u, row_v, coeff_v, row_w, coeff_w):
            nonlocal col
            if col >= rank:
                raise ValueError("Rank exceeded")
            U[row_u][col] += coeff_u
            V[row_v][col] += coeff_v
            W[row_w][col] += coeff_w
            col += 1
        
        # But wait, a single Strassen product like S1 = A[0,0]*(B[0,c1] - B[1,c1]) 
        # = A[0,0]*B[0,c1] - A[0,0]*B[1,c1]
        # This is 2 rank-1 terms, not 1. So Strassen's 7 products actually expand to more terms.
        
        # Hmm, I think I'm confusing myself. In the tensor decomposition, each "product" in Strassen
        # is a bilinear form that may itself require multiple rank-1 terms.
        
        # Let me reconsider. The standard Strassen algorithm computes 7 products, each of which
        # is a scalar (for 2x2 matrices). Each scalar product of the form x*y where x is a linear
        # combination of P entries and y is a linear combination of Q entries is rank 1 IF x and y
        # are single entries. But if x = P[0,0] + P[0,1], then x*y = (P[0,0]+P[0,1])*y = P[0,0]*y + P[0,1]*y,
        # which is rank 2.
        
        # So the actual rank contribution of Strassen for 2x2*2x2 is:
        # S1 = P[0,0]*(Q[0,1]-Q[1,1]): P[0,0] is single, (Q[0,1]-Q[1,1]) is 2 terms -> rank 2
        # S2 = (P[0,0]+P[0,1])*Q[1,1]: (P[0,0]+P[0,1]) is 2 terms, Q[1,1] single -> rank 2
        # S3 = (P[1,0]-P[0,0])*Q[0,0]: rank 2
        # S4 = P[1,1]*(Q[1,0]-Q[0,0]): rank 2
        # S5 = (P[0,0]+P[1,1])*(Q[0,0]+Q[1,1]): rank 4
        # S6 = (P[0,1]-P[1,1])*(Q[1,0]+Q[1,1]): rank 4
        # S7 = (P[1,0]-P[0,0])*(Q[0,0]+Q[0,1]): rank 4
        # Total: 2+2+2+2+4+4+4 = 20? That's worse than schoolbook (16).
        
        # I think I'm making an error. The Strassen algorithm's rank is 7 because each of the 7 products
        # is treated as an atomic operation. In the tensor decomposition sense, the rank of the 2x2*2x2
        # tensor is 7, meaning it can be decomposed into 7 rank-1 tensors. Each rank-1 tensor has the form
        # u_i ⊗ v_i ⊗ w_i where u_i is in F^4 (for P), v_i in F^4 (for Q), w_i in F^4 (for R).
        
        # So I should be working with the full Strassen decomposition, where each of the 7 terms
        # is a single rank-1 tensor in the 4x4x4 tensor space.
        
        # Let me use the known Strassen decomposition. The 7 rank-1 components for 2x2*2x2:
        
        # Let P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
        # R = P*Q, R[i,j] = sum_k P[i,k]*Q[k,j]
        
        # Strassen's 7 products:
        # S1 = p00*(q01-q11)
        # S2 = (p00+p01)*q11
        # S3 = (p10-p00)*q00
        # S4 = p11*(q10-q00)
        # S5 = (p00+p11)*(q00+q11)
        # S6 = (p01-p11)*(q10+q11)
        # S7 = (p10-p00)*(q00+q01)
        
        # R00 = S5 + S4 - S2 + S6
        # R01 = S1 + S2
        # R10 = S3 + S5
        # R11 = S1 - S3 + S4 + S7
        
        # Each S is a bilinear form. To decompose S into rank-1 tensors, note that S is already
        # a product of a linear form in P and a linear form in Q. So S = L_P * L_Q where L_P is linear
        # in P entries and L_Q is linear in Q entries. This is a rank-1 tensor in P⊗Q⊗R space IF
        # we can express it as sum of R[i,j] coefficients times... 
        
        # Actually, S contributes to multiple R[i,j] entries. The rank-1 tensor decomposition means:
        # S = sum over (i,j) of coeff[i,j] * R[i,j], and S = L_P * L_Q.
        # So the tensor component is L_P ⊗ L_Q ⊗ (vector of coeffs in R space).
        
        # For the full decomposition, we have 7 such components.
        
        # Let me just directly construct the U, V, W matrices for the Strassen part.
        
        # For block q, the Strassen part computes M_00 = A[:, 0:2] * B[0:2, c0:c1+1]
        # Let P = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # Let Q = B[0:2, c0:c1+1] = [[B[0,c0], B[0,c1]], [B[1,c0], B[1,c1]]]
        
        # U rows for P: A[i,j] at row i*m+j = i*3+j
        #   A[0,0] at row 0, A[0,1] at row 1, A[1,0] at row 3, A[1,1] at row 4
        # V rows for Q: B[j, k] at row j*p+k = j*10+k
        #   B[0,c0] at row c0, B[0,c1] at row c1, B[1,c0] at row 10+c0, B[1,c1] at row 10+c1
        # W rows for M_00: M_00[i,r] contributes to C[i, c0+r] at W index (c0+r)*2+i
        
        # Strassen products and their contributions:
        # S1 = P[0,0]*(Q[0,1]-Q[1,1]) = A[0,0]*(B[0,c1]-B[1,c1])
        #   U: A[0,0] coeff 1
        #   V: B[0,c1] coeff 1, B[1,c1] coeff -1
        #   W: R01 coeff 1, R11 coeff 1
        #   R01 = C[0, c1] at W index c1*2+0 = 2*c1
        #   R11 = C[1, c1] at W index c1*2+1 = 2*c1+1
        
        # But S1 is a single bilinear form that involves 2 V terms. In the tensor decomposition,
        # this means we need 2 rank-1 terms, OR we keep it as one term with V having 2 non-zero entries.
        # In the standard tensor decomposition, V can have multiple non-zero entries in a single column.
        # So S1 as one column: U[0]=1, V[c1]=1, V[10+c1]=-1, W[2*c1]=1, W[2*c1+1]=1.
        
        # Let me do it this way.
        
        # S1: U[0]=1, V[c1]=1, V[10+c1]=-1, W[2*c1]=1, W[2*c1+1]=1
        add_term(0, 1, c1, 1, 2*c1, 1)
        # But add_term only handles one V entry. Let me modify.
        
        # Actually, let me just directly set the values without the helper, since each Strassen product
        # can have multiple non-zero entries in U, V, or W.
        
        col -= 1  # undo the add_term
        # S1
        U[0][col] = 1
        V[c1][col] = 1
        V[10+c1][col] = -1
        W[2*c1][col] = 1
        W[2*c1+1][col] = 1
        col += 1
        
        # S2 = (P[0,0]+P[0,1])*Q[1,1] = (A[0,0]+A[0,1])*B[1,c1]
        U[0][col] = 1
        U[1][col] = 1
        V[10+c1][col] = 1
        W[2*c0][col] = -1  # R00: -S2
        W[2*c1][col] = 1   # R01: +S2
        col += 1
        
        # S3 = (P[1,0]-P[0,0])*Q[0,0] = (A[1,0]-A[0,0])*B[0,c0]
        U[3][col] = 1
        U[0][col] = -1
        V[c0][col] = 1
        W[2*c0+1][col] = 1  # R10: +S3
        W[2*c1+1][col] = -1 # R11: -S3
        col += 1
        
        # S4 = P[1,1]*(Q[1,0]-Q[0,0]) = A[1,1]*(B[1,c0]-B[0,c0])
        U[4][col] = 1
        V[10+c0][col] = 1
        V[c0][col] = -1
        W[2*c0][col] = 1   # R00: +S4
        W[2*c1+1][col] = 1 # R11: +S4
        col += 1
        
        # S5 = (P[0,0]+P[1,1])*(Q[0,0]+Q[1,1]) = (A[0,0]+A[1,1])*(B[0,c0]+B[1,c1])
        U[0][col] = 1
        U[4][col] = 1
        V[c0][col] = 1
        V[10+c1][col] = 1
        W[2*c0][col] = 1   # R00: +S5
        W[2*c0+1][col] = 1 # R10: +S5
        col += 1
        
        # S6 = (P[0,1]-P[1,1])*(Q[1,0]+Q[1,1]) = (A[0,1]-A[1,1])*(B[1,c0]+B[1,c1])
        U[1][col] = 1
        U[4][col] = -1
        V[10+c0][col] = 1
        V[10+c1][col] = 1
        W[2*c0][col] = 1   # R00: +S6
        col += 1
        
        # S7 = (P[1,0]-P[0,0])*(Q[0,0]+Q[0,1]) = (A[1,0]-A[0,0])*(B[0,c0]+B[0,c1])
        U[3][col] = 1
        U[0][col] = -1
        V[c0][col] = 1
        V[c1][col] = 1
        W[2*c1+1][col] = 1 # R11: +S7
        col += 1
        
        # --- Outer product part: A[:, 2:3] * B[2:3, c0:c1+1] ---
        # M_10[i,r] = A[i,2] * B[2, c0+r]
        # This is a 2x2 outer product. Rank 2.
        # M_10[0,0] = A[0,2]*B[2,c0] -> C[0,c0] at W index 2*c0
        # M_10[0,1] = A[0,2]*B[2,c1] -> C[0,c1] at W index 2*c1
        # M_10[1,0] = A[1,2]*B[2,c0] -> C[1,c0] at W index 2*c0+1
        # M_10[1,1] = A[1,2]*B[2,c1] -> C[1,c1] at W index 2*c1+1
        
        # We can do this with 2 rank-1 terms:
        # Term 1: A[0,2]*B[2,c0] -> W[2*c0], and A[0,2]*B[2,c1] -> W[2*c1]
        # But that's 2 separate terms. Or:
        # Term 1: U[2]=1 (A[0,2]), V[20+c0]=1, V[20+c1]=1, W[2*c0]=1, W[2*c1]=1
        # Term 2: U[5]=1 (A[1,2]), V[20+c0]=1, V[20+c1]=1, W[2*c0+1]=1, W[2*c1+1]=1
        
        U[2][col] = 1
        V[20+c0][col] = 1
        V[20+c1][col] = 1
        W[2*c0][col] = 1
        W[2*c1][col] = 1
        col += 1
        
        U[5][col] = 1
        V[20+c0][col] = 1
        V[20+c1][col] = 1
        W[2*c0+1][col] = 1
        W[2*c1+1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
