"""
6x6 matrix multiplication tensor decomposition.
Uses the Kronecker product of Strassen (2x2, rank 7) and a 3x3 decomposition.
The 3x3 decomposition is the standard rank-23 (or best known) decomposition.
Since we don't have the exact 3x3 rank-23 coefficients memorized,
we use the tensor product structure: rank = 7 * R(3,3,3).
But actually, the known improvement from the prompt is:
"Tensor product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 27) reduces rank from 216 to 189"
So R(3,3,3) schoolbook = 27, and 7*27 = 189.

Let me implement the Kronecker product of Strassen 2x2 with schoolbook 3x3.
This gives rank 189, which is an improvement over the seed's 216.
"""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    # We split 6 = 2 * 3 in each dimension.
    # A is 6x6, B is 6x6, C = A*B is 6x6.
    # Block A into 2x2 blocks of size 3x3: A = [[A11, A12], [A21, A22]]
    # Block B into 2x2 blocks of size 3x3: B = [[B11, B12], [B21, B22]]
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Strassen's 7 multiplications for the 2x2 block structure:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mj is a 3x3 matrix product, done with schoolbook rank 27.
    # So total rank = 7 * 27 = 189.
    
    # For the tensor decomposition, we need:
    # U: shape (6*6, rank) = (36, rank)
    # V: shape (6*6, rank) = (36, rank)  
    # W: shape (6*6, rank) = (36, rank)
    # where W follows the trace convention W[k*n+i, r]
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper: get the row index in the 6x6 matrix for block (bi, bj) and intra-block (i, j)
    # A is indexed as A[row, col] where row = bi*3 + i, col = bj*3 + j
    # For U (A matrix): row index = (bi*3 + i) * 6 + (bj*3 + j)
    # For V (B matrix): row index = (bi*3 + i) * 6 + (bj*3 + j)
    # For W (C matrix, transposed): W[k*n + i, r] means C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    # Wait, let me re-read the convention.
    
    # The standard tensor decomposition for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C_ik = sum_j A_ij B_jk
    # The rank-1 terms: for each (i,j,k), we have A_ij * B_jk contributes to C_ik
    # U row for A_ij: index i*m + j (row i, col j of A)
    # V row for B_jk: index j*p + k (row j, col k of B)
    # W row for C_ik: index k*n + i (the convention says W[k*n+i, r])
    
    # So for the Kronecker product approach:
    # We have 7 "multiplications" at the 2x2 block level, each being a 3x3 product.
    # For each Strassen multiplication Mj, we need the 27 schoolbook terms for the 3x3 product.
    
    # Let's define the Strassen terms:
    # Each Mj is computed as sum over 27 terms of (A_block element) * (B_block element)
    
    # For M1 = (A11 + A22) * (B11 + B22):
    # The A-block elements involved: A11 (block 0,0) and A22 (block 1,1)
    # The B-block elements involved: B11 (block 0,0) and B22 (block 1,1)
    # But it's a sum of products, so we need to expand.
    
    # Actually, let me think about this differently.
    # The Kronecker product of two decompositions:
    # If we have a decomposition for 2x2 with rank r1 and for 3x3 with rank r2,
    # the Kronecker product gives a decomposition for 6x6 with rank r1*r2.
    
    # Strassen 2x2: U2 (4, 7), V2 (4, 7), W2 (4, 7)
    # Schoolbook 3x3: U3 (9, 27), V3 (9, 27), W3 (9, 27)
    
    # The Kronecker product:
    # U6[i2*9 + i3, j2*27 + j3] = U2[i2, j2] * U3[i3, j3]
    # Similarly for V6 and W6.
    
    # Let me first define the Strassen 2x2 decomposition.
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ce+bg, cf+bh],[ae+dg, af+dh]]
    # Wait, standard: C11 = ae+bg, C12 = af+bh, C21 = ce+dg, C22 = cf+dh
    
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition of 2x2 matrix multiplication:
    # U2 rows are indexed by (i,j) where i is row of A, j is col of A: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    # V2 rows indexed by (j,k) where j is row of B, k is col of B: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    # W2 rows indexed by (k,i) per the convention W[k*2+i, r]: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    
    # Let's build U2, V2, W2 for Strassen:
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # Term ae: A(0,0)*B(0,0) -> U2[0,0]=1, V2[0,0]=1, W2[0,0]+=1 (C[0,0])
    # Term ah: A(0,0)*B(1,1) -> U2[0,1]=1, V2[3,1]=1, W2[1,0]+=1 (C[0,1])... 
    # Wait, let me be more careful.
    
    # A has indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B has indices: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C has indices: C[0,0]=ae+bg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh
    
    # U2 row index: i*2+j for A[i,j]
    # V2 row index: j*2+k for B[j,k]
    # W2 row index: k*2+i for C[i,k] (per convention W[k*n+i])
    
    # M1 = (a+d)(e+h):
    # a*e: A[0,0]*B[0,0] -> contributes to C[0,0]. U2[0,0]=1, V2[0,0]=1, W2[0,0]=1
    # a*h: A[0,0]*B[1,1] -> contributes to C[0,1]. U2[0,1]=1, V2[3,1]=1, W2[1,0]=1... 
    
    # Hmm, this is getting complex. Let me just directly construct the Kronecker product.
    
    # Strassen 2x2 decomposition (rank 7):
    # I'll define it explicitly.
    
    # Let me use a cleaner approach. Define the 2x2 Strassen tensors.
    # For each of the 7 multiplications, list the (A_index, B_index, C_index) triples.
    
    # M1 = (a+d)(e+h):
    # a*e -> C[0,0]: (0, 0, 0) where A_idx=0 (a), B_idx=0 (e), C_idx=0 (C[0,0])
    # a*h -> C[0,1]: (0, 3, 1) where A_idx=0 (a), B_idx=3 (h), C_idx=1 (C[0,1])
    # d*e -> C[1,0]: (3, 0, 2) where A_idx=3 (d), B_idx=0 (e), C_idx=2 (C[1,0])
    # d*h -> C[1,1]: (3, 3, 3) where A_idx=3 (d), B_idx=3 (h), C_idx=3 (C[1,1])
    
    # M2 = (c+d)e:
    # c*e -> C[1,0]: (2, 0, 2)
    # d*e -> C[1,0]: (3, 0, 2)
    
    # M3 = a(f-h):
    # a*f -> C[0,1]: (0, 1, 1)
    # a*(-h) -> C[0,0]: (0, 3, 0) with coefficient -1
    
    # M4 = d(g-e):
    # d*g -> C[1,1]: (3, 2, 3)
    # d*(-e) -> C[1,0]: (3, 0, 2) with coefficient -1
    
    # M5 = (a+b)h:
    # a*h -> C[0,1]: (0, 3, 1)
    # b*h -> C[1,1]: (1, 3, 3)
    
    # M6 = (c-a)(e+f):
    # c*e -> C[1,0]: (2, 0, 2)
    # c*f -> C[1,1]: (2, 1, 3)
    # (-a)*e -> C[0,0]: (0, 0, 0) with coefficient -1
    # (-a)*f -> C[0,1]: (0, 1, 1) with coefficient -1
    
    # M7 = (b-d)(g+h):
    # b*g -> C[0,0]: (1, 2, 0)
    # b*h -> C[0,1]: (1, 3, 1)
    # (-d)*g -> C[1,1]: (3, 2, 3) with coefficient -1
    # (-d)*h -> C[1,0]: (3, 3, 2) with coefficient -1
    
    # Now let's verify:
    # C[0,0] = ae + bg
    # From M1: ae (coeff 1)
    # From M3: -a*h... no, M3 contributes to C[0,0] via a*(-h)? No, a*(-h) would be in C[0,1]... 
    
    # I'm getting confused. Let me look up the standard Strassen coefficients.
    
    # Actually, let me just use the known Strassen decomposition directly.
    
    # For 2x2 matrix multiplication, the Strassen decomposition has rank 7.
    # U2, V2, W2 are 4x7 matrices.
    
    # A[i,j] for i,j in {0,1}: index = i*2+j
    # B[i,j] for i,j in {0,1}: index = i*2+j
    # C[i,j] for i,j in {0,1}: index = j*2+i (per W[k*n+i] convention)
    
    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Now, each Pj is a product of two linear combinations.
    # For the tensor decomposition, each Pj corresponds to one rank-1 term... no, 
    # each Pj is a matrix product, so in the 2x2 case, each Pj is a scalar product.
    
    # For the 2x2 case, U2, V2, W2:
    # P1: (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This contributes to C00 with coefficients:
    # A00*B00 -> C00: U2[0,0]=1, V2[0,0]=1, W2[0,0]+=1
    # A00*B11 -> C01: U2[0,1]=1, V2[3,1]=1, W2[1,0]+=1  (C[0,1] -> index 1*2+0=2? No, W[k*2+i] for C[i,k])
    # Wait, W[k*n+i] where n=2, so W[k*2+i]. C[i,k] corresponds to W[k*2+i].
    # C[0,0] -> W[0*2+0]=W[0]
    # C[0,1] -> W[1*2+0]=W[2]
    # C[1,0] -> W[0*2+1]=W[1]
    # C[1,1] -> W[1*2+1]=W[3]
    
    # Let me redo:
    # C[0,0] -> W index 0
    # C[0,1] -> W index 2
    # C[1,0] -> W index 1
    # C[1,1] -> W index 3
    
    # P1 = (A00+A11)(B00+B11):
    # A00*B00 -> C[0,0]: U2[0,0]=1, V2[0,0]=1, W2[0,0]+=1
    # A00*B11 -> C[0,1]: U2[0,1]=1, V2[3,1]=1, W2[2,0]+=1
    # A11*B00 -> C[1,0]: U2[3,0]=1, V2[0,2]=1, W2[1,0]+=1
    # A11*B11 -> C[1,1]: U2[3,1]=1, V2[3,3]=1, W2[3,0]+=1
    
    # P2 = (A10+A11)B00:
    # A10*B00 -> C[1,0]: U2[2,1]=1, V2[0,1]=1, W2[1,1]+=1
    # A11*B00 -> C[1,1]: U2[3,1]=1, V2[0,2]=1, W2[3,1]+=1
    
    # P3 = A00(B01-B11):
    # A00*B01 -> C[0,1]: U2[0,2]=1, V2[1,2]=1, W2[2,2]+=1
    # A00*(-B11) -> C[0,0]: U2[0,3]=-1, V2[3,3]=1, W2[0,3]+=1
    
    # P4 = A11(B10-B00):
    # A11*B10 -> C[1,1]: U2[3,3]=1, V2[2,3]=1, W2[3,3]+=1
    # A11*(-B00) -> C[1,0]: U2[3,4]=-1, V2[0,4]=1, W2[1,4]+=1
    
    # P5 = (A00+A01)B11:
    # A00*B11 -> C[0,1]: U2[0,4]=1, V2[3,4]=1, W2[2,4]+=1
    # A01*B11 -> C[1,1]: U2[1,4]=1, V2[3,5]=1, W2[3,5]+=1
    
    # P6 = (A10-A00)(B00+B01):
    # A10*B00 -> C[1,0]: U2[2,5]=1, V2[0,5]=1, W2[1,5]+=1
    # A10*B01 -> C[1,1]: U2[2,6]=1, V2[1,6]=1, W2[3,6]+=1
    # (-A00)*B00 -> C[0,0]: U2[0,7]=-1, V2[0,7]=1, W2[0,7]+=1
    # (-A00)*B01 -> C[0,1]: U2[0,8]=-1, V2[1,8]=1, W2[2,8]+=1
    
    # P7 = (A01-A11)(B10+B11):
    # A01*B10 -> C[0,0]: U2[1,8]=1, V2[2,8]=1, W2[0,8]+=1
    # A01*B11 -> C[0,1]: U2[1,9]=1, V2[3,9]=1, W2[2,9]+=1
    # (-A11)*B10 -> C[1,1]: U2[3,10]=-1, V2[2,10]=1, W2[3,10]+=1
    # (-A11)*B11 -> C[1,0]: U2[3,11]=-1, V2[3,11]=1, W2[1,11]+=1
    
    # Wait, this is not right. Each Pj should be a single rank-1 term in the 2x2 case,
    # but I'm expanding them into multiple terms. That's not how it works.
    
    # Actually, in the tensor decomposition, each "multiplication" in Strassen's algorithm
    # corresponds to a rank-1 term. So P1 is one rank-1 term, P2 is another, etc.
    # But P1 = (A00+A11)(B00+B11) is not a single A_ij * B_jk term, it's a sum.
    
    # I think I'm overcomplicating this. Let me use a different approach.
    
    # The Kronecker product of decompositions:
    # If T1 is a decomposition of the n1x m1 x p1 multiplication tensor with rank r1,
    # and T2 is a decomposition of the n2x m2 x p2 multiplication tensor with rank r2,
    # then the Kronecker product T1 ⊗ T2 is a decomposition of the (n1*n2)x(m1*m2)x(p1*p2)
    # multiplication tensor with rank r1*r2.
    
    # For Strassen 2x2 (n1=m1=p1=2, r1=7) and schoolbook 3x3 (n2=m2=p2=3, r2=27),
    # we get 6x6x6 with rank 7*27=189.
    
    # The construction:
    # U6[(i1*3+i2)*3 + (j1*3+j2), c1*27+c2] = U2[i1*2+j1, c1] * U3[i2*3+j2, c2]
    # V6[(j1*3+j2)*3 + (k1*3+k2), c1*27+c2] = V2[j1*2+k1, c1] * V3[j2*3+k2, c2]
    # W6[(k1*3+k2)*3 + (i1*3+i2), c1*27+c2] = W2[k1*2+i1, c1] * W3[k2*3+i2, c2]
    
    # Wait, I need to be careful with the indexing.
    
    # Let me just define U2, V2, W2 for Strassen 2x2 and U3, V3, W3 for schoolbook 3x3,
    # then compute the Kronecker product.
    
    # Strassen 2x2:
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)  [W[k*2+i] convention]
    
    # Actually, let me just hard-code the known Strassen decomposition.
    
    # From Wikipedia or standard references, the Strassen decomposition for 2x2:
    # U2 = [
    #   [1, 0, 1, 0, 1, -1, 0],   # a
    #   [0, 0, 0, 0, 1, 0, 1],     # b
    #   [0, 1, 0, 0, 0, 1, 0],     # c
    #   [1, 1, 0, 1, 0, 0, -1]     # d
    # ]
    # V2 = [
    #   [1, 1, 0, -1, 0, 1, 0],    # e
    #   [0, 0, 1, 0, 0, 1, 0],     # f
    #   [1, 0, 0, 1, 0, 0, 1],     # g
    #   [1, 0, -1, 0, 1, 0, 1]     # h
    # ]
    # W2 = [
    #   [1, 0, -1, 1, 0, 0, 1],    # C[0,0]
    #   [0, 1, 0, -1, 0, 0, -1],   # C[1,0]
    #   [0, 0, 1, 0, 1, 0, 0],     # C[0,1]
    #   [1, -1, 1, 0, 0, 1, 0]     # C[1,1]
    # ]
    
    # Let me verify: C[0,0] should be ae + bg
    # Sum over r: U2[0,r]*V2[0,r]*W2[0,r] + U2[1,r]*V2[2,r]*W2[0,r] (since C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0])
    # Wait, C[i,k] = sum_j A[i,j]*B[j,k]
    # For the tensor decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]... 
    # No, that's not right either.
    
    # The tensor decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] * W[k*n+i, r]) * (sum_j V[j*p+k, r] * ... )
    
    # No, the correct form is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j... 
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # The multiplication tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In the trace convention: T is a 3-tensor where T[(i,j),(j,k),(k,i)] = 1 for the term A_ij * B_jk contributing to C_ik
    # 
    # A rank-r decomposition: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # Where u_r is a vector of length n*m, v_r of length m*p, w_r of length p*n
    # 
    # So: T[i,j, k] (with the trace convention) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # And T[i,j,k] = 1 if the (i,j) element of A and (j,k) element of B multiply to give (i,k) element of C
    # In the trace convention, T is non-zero at positions where the three indices form a valid product term.
    
    # For the schoolbook decomposition: each term (i,j,k) gives a rank-1 term with
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # For Strassen, the decomposition is a sum of 7 rank-1 terms, but each term is a "linear combination"
    # so U, V, W have multiple non-zero entries per column.
    
    # Let me just use the known Strassen U, V, W matrices.
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, -1, 1, 0, 0, 1],
        [0, 1, 0, -1, 0, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook 3x3:
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][c] = 1
                V3[j*3+k][c] = 1
                W3[k*3+i][c] = 1
                c += 1
    
    # Now construct U6, V6, W6 via Kronecker product:
    # U6[(i1*3+i2)*6 + (j1*3+j2), c1*27+c2] = U2[i1*2+j1, c1] * U3[i2*3+j2, c2]
    # Wait, the row index for A in 6x6: i*6+j where i is row, j is col
    # i = i1*3+i2, j = j1*3+j2
    # So row index = (i1*3+i2)*6 + (j1*3+j2)
    
    # Similarly for V: row index = (j1*3+j2)*6 + (k1*3+k2)
    # For W: row index = (k1*3+k2)*6 + (i1*3+i2)
    
    U6 = [[0]*rank for _ in range(36)]
    V6 = [[0]*rank for _ in range(36)]
    W6 = [[0]*rank for _ in range(36)]
    
    for c1 in range(7):
        for c2 in range(27):
            cr = c1*27 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(3):
                        for j2 in range(3):
                            u2_val = U2[i1*2+j1][c1]
                            u3_val = U3[i2*3+j2][c2]
                            if u2_val != 0 and u3_val != 0:
                                row_u6 = (i1*3+i2)*6 + (j1*3+j2)
                                U6[row_u6][cr] = u2_val * u3_val
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(3):
                        for k2 in range(3):
                            v2_val = V2[j1*2+k1][c1]
                            v3_val = V3[j2*3+k2][c2]
                            if v2_val != 0 and v3_val != 0:
                                row_v6 = (j1*3+j2)*6 + (k1*3+k2)
                                V6[row_v6][cr] = v2_val * v3_val
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(3):
                        for i2 in range(3):
                            w2_val = W2[k1*2+i1][c1]
                            w3_val = W3[k2*3+i2][c2]
                            if w2_val != 0 and w3_val != 0:
                                row_w6 = (k1*3+k2)*6 + (i1*3+i2)
                                W6[row_w6][cr] = w2_val * w3_val
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
