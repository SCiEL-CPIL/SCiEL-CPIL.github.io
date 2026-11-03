import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Block decomposition:
    # A is 3x4: split rows into [2, 1], cols into [2, 2]
    # B is 4x6: split rows into [2, 2], cols into [2, 2, 2]
    # C is 3x6: split rows into [2, 1], cols into [2, 2, 2]
    #
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]  (two 2x2*2x2)
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]  (two 2x2*2x2)
    # C[0:2, 4:6] = A[0:2, 0:2]*B[0:2, 4:6] + A[0:2, 2:4]*B[2:4, 4:6]  (two 2x2*2x2)
    # C[2:3, 0:2] = A[2:3, 0:2]*B[0:2, 0:2] + A[2:3, 2:4]*B[2:4, 0:2]  (two 1x2*2x2)
    # C[2:3, 2:4] = A[2:3, 0:2]*B[0:2, 2:4] + A[2:3, 2:4]*B[2:4, 2:4]  (two 1x2*2x2)
    # C[2:3, 4:6] = A[2:3, 0:2]*B[0:2, 4:6] + A[2:3, 2:4]*B[2:4, 4:6]  (two 1x2*2x2)
    #
    # 6 blocks of 2x2*2x2 -> 6 * 7 = 42 (Strassen)
    # 6 blocks of 1x2*2x2 -> 6 * 4 = 24 (direct: sum over 2 inner dims, 2 output cols)
    # Total: 66
    #
    # Wait, let me reconsider. A 1x2 * 2x2 matrix product:
    # (a, b) * [[c, d], [e, f]] = [ac+be, ad+bf]
    # This is a bilinear map from 4 inputs to 2 outputs.
    # The tensor rank of (1,2,2) matrix multiplication:
    # We need U (1*2 x r), V (2*2 x r), W (2*1 x r)
    # U has 2 entries per column, V has 4 entries per column, W has 2 entries per column.
    # For each output (col k=0,1) and input pair, we need:
    # C[0,0] = a*c + b*e
    # C[0,1] = a*d + b*f
    #
    # This is equivalent to: for output col 0: sum over j of A[0,j]*B[j,0]
    # For output col 1: sum over j of A[0,j]*B[j,1]
    #
    # The bilinear complexity of (1,2,2) matmul:
    # We can write it with rank 4:
    # c0: U[0]=1,U[1]=0; V[0,0]=1,V[0,1]=0,V[1,0]=0,V[1,1]=0; W[0]=1,W[1]=0  -> a*c
    # c1: U[0]=1,U[1]=0; V[0,0]=0,V[0,1]=1,V[1,0]=0,V[1,1]=0; W[0]=0,W[1]=1  -> a*d
    # c2: U[0]=0,U[1]=1; V[0,0]=0,V[0,1]=0,V[1,0]=1,V[1,1]=0; W[0]=1,W[1]=0  -> b*e
    # c3: U[0]=0,U[1]=1; V[0,0]=0,V[0,1]=0,V[1,0]=0,V[1,1]=1; W[0]=0,W[1]=1  -> b*f
    #
    # Can we do it in rank 3? The (1,2,2) tensor... Let's think.
    # Actually, the matrix multiplication tensor M_{1,2,2} has rank 4 (it's the standard result).
    # So 6 * 4 = 24 for the mixed blocks.
    #
    # Total: 42 + 24 = 66. That's better than 68!
    #
    # Actually wait - let me re-examine. Can we share Strassen terms?
    # No, because the blocks involve different submatrices of A and B.
    #
    # Let me also consider: can we split differently?
    # A is 3x4, B is 4x6.
    # Alternative: split A cols into [1,3], B rows into [1,3]:
    # Then we get 1x1*1x1 blocks (rank 1 each) and 1x3*3x2 etc.
    # That seems worse.
    #
    # Let's go with: A rows [2,1], A cols [2,2]; B rows [2,2], B cols [2,2,2]
    # 6 Strassen blocks (rank 7 each) + 6 direct blocks (rank 4 each) = 42 + 24 = 66

    # Let me verify: can we do the 1x2 * 2x2 in rank 3?
    # The tensor M_{1,2,2}: T[i,j,k] = 1 if i=0, and (j,k) corresponds to the product.
    # Actually, M_{1,2,2}[a_idx, b_idx, c_idx] where:
    # a_idx ranges over (row, col) of A = (0,0),(0,1) -> 2 values
    # b_idx ranges over (row, col) of B = (0,0),(0,1),(1,0),(1,1) -> 4 values
    # c_idx ranges over (row, col) of C = (0,0),(0,1) -> 2 values
    #
    # T[a_idx, b_idx, c_idx] = 1 iff the (a_idx row, a_idx col) of A times (b_idx row, b_idx col) of B contributes to (c_idx row, c_idx col) of C.
    #
    # A is 1x2: A[0,0]=a, A[0,1]=b
    # B is 2x2: B[0,0]=c, B[0,1]=d, B[1,0]=e, B[1,1]=f
    # C is 1x2: C[0,0]=ac+be, C[0,1]=ad+bf
    #
    # T[0, (0,0), (0,0)] = 1  (a*c)
    # T[0, (1,0), (0,0)] = 1  (b*e)
    # T[0, (0,1), (0,1)] = 1  (a*d)
    # T[0, (1,1), (0,1)] = 1  (b*f)
    # All other entries are 0.
    #
    # This is a sum of 4 disjoint "trivial" tensors, so rank <= 4.
    # Is rank = 4? Each term involves a distinct (a_idx, b_idx, c_idx) triple,
    # and they're all linearly independent, so yes rank = 4.
    #
    # So total rank = 6*7 + 6*4 = 42 + 24 = 66.

    # Now let's build it.
    # Strassen for 2x2*2x2:
    # Given A2 (2x2), B2 (2x2), compute C2 = A2*B2 with 7 products.
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    # For the full problem, I'll build the U, V, W matrices directly.
    # 
    # Indexing:
    # U: (n*m) x rank = 12 x 66, indexed by (i, j) where i in [0,3), j in [0,4)
    #   U[i*m + j, c]
    # V: (m*p) x rank = 24 x 66, indexed by (j, k) where j in [0,4), k in [0,6)
    #   V[j*p + k, c]
    # W: (p*n) x rank = 18 x 66, indexed by (k, i) where k in [0,6), i in [0,3)
    #   W[k*n + i, c]
    #
    # For the decomposition to work:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    #
    # This means: for each (i,j,k), the "coefficient" of A[i,j]*B[j,k] in C[i,k] must be 1.
    #
    # Let me think about this more carefully. The standard tensor decomposition of matrix mult:
    # T[i,j,k] = 1 for all i,j,k (the "identity" tensor for matmul)
    # We want: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    #
    # For the schoolbook: each (i,j,k) gets its own term with U=V=W=1.
    #
    # For Strassen on a 2x2 block:
    # Consider rows i in {i0, i0+1}, cols j in {j0, j0+1}, cols k in {k0, k0+1}.
    # The 8 entries of T for this block form the 2x2x2 matmul tensor.
    # Strassen decomposes it with 7 terms.
    #
    # For term M1 = (a00+a11)*(b00+b11):
    #   U contributions: U[i0,j0] += 1, U[i0+1,j0+1] += 1
    #   V contributions: V[j0,k0] += 1, V[j0+1,k0+1] += 1
    #   W contributions: W[k0,i0] += 1, W[k0+1,i0] += 1, W[k0,i0+1] += 1, W[k0+1,i0+1] += 1
    #   Wait, that's not right. Let me re-derive.
    #
    # For M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1]):
    # This expands to: A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j0+1,k0+1] + A[i0+1,j0+1]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0+1]
    #
    # In the tensor decomposition, this single term c contributes:
    # U[i0,j0,c]=1, U[i0+1,j0+1,c]=1
    # V[j0,k0,c]=1, V[j0+1,k0+1,c]=1
    # W[k0,i0,c]=1, W[k0+1,i0,c]=1, W[k0,i0+1,c]=1, W[k0+1,i0+1,c]=1
    #
    # Then the product U*V*W for this term gives:
    # For (i0,j0,k0): U[i0,j0]=1, V[j0,k0]=1, W[k0,i0]=1 -> 1
    # For (i0,j0,k0+1): U[i0,j0]=1, V[j0,k0+1]=0, ... -> 0
    # For (i0,j0+1,k0): U[i0,j0+1]=0, ... -> 0
    # For (i0+1,j0,k0): U[i0+1,j0]=0, ... -> 0
    # For (i0+1,j0+1,k0): U[i0+1,j0+1]=1, V[j0+1,k0]=0, ... -> 0
    # For (i0+1,j0,k0+1): U[i0+1,j0]=0, ... -> 0
    # For (i0,j0+1,k0+1): U[i0,j0+1]=0, ... -> 0
    # For (i0+1,j0+1,k0+1): U[i0+1,j0+1]=1, V[j0+1,k0+1]=1, W[k0+1,i0+1]=1 -> 1
    #
    # So M1 contributes 1 to T[i0,j0,k0] and T[i0+1,j0+1,k0+1].
    #
    # Now let's verify the full Strassen reconstruction:
    # T[i0,j0,k0] should be 1 (coefficient of A[i0,j0]*B[j0,k0] in C[i0,k0]):
    #   M1: +1, M4: +1, M5: -1, M7: +1 -> 1+1-1+1 = 2? That's wrong!
    #
    # I think I'm confusing the output combination with the tensor decomposition.
    # In Strassen, the 7 products are intermediate values, and the final C entries are
    # linear combinations. But in the tensor decomposition, each "term" c is a rank-1
    # tensor, and the sum over c gives T directly.
    #
    # Let me re-approach. The matrix multiplication tensor M_{2,2,2} has entries:
    # M[i,j,k] = 1 for all i,j,k in {0,1}.
    #
    # A rank-1 tensor is U[i]*V[j]*W[k]. The rank of M_{2,2,2} is 7 (Strassen).
    #
    # One standard decomposition:
    # c=0: U=[1,0;0,1], V=[1,0;0,1], W=[1,0;0,1]  -> diagonal: contributes to (0,0,0) and (1,1,1)
    # c=1: U=[0,1;0,0], V=[1,0;0,0], W=[0,1;0,0]  -> contributes to (1,0,1)
    # c=2: U=[1,0;0,0], V=[0,1;0,0], W=[0,1;0,0]  -> contributes to (0,1,1)
    # c=3: U=[0,0;1,0], V=[0,0;1,0], W=[0,1;0,0]  -> contributes to (1,1,1)
    # ... this is getting complicated. Let me use a known explicit Strassen decomposition.
    #
    # Actually, let me just use the known Strassen 7-term decomposition of M_{2,2,2}.
    # The 7 rank-1 tensors (U_c, V_c, W_c) for c=0..6:
    #
    # I'll use the formulation where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M3
    #
    # In tensor decomposition form, each M_c corresponds to a rank-1 term.
    # M1 = (A00+A11)(B00+B11):
    #   U1 = [1,0; 0,1] (coefficients of A00, A11)
    #   V1 = [1,0; 0,1] (coefficients of B00, B11)
    #   W1 = [1,1; 1,1] (appears in C00, C01, C10, C11 with coeff +1 each)
    #
    # Wait, that's not right either. Let me think again.
    #
    # The tensor T[i,j,k] = 1 for all i,j,k. We want T = sum_c U_c ⊗ V_c ⊗ W_c.
    # Where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i] * V_c[j] * W_c[k].
    #
    # For M1 = (A00+A11)(B00+B11), this is the product of:
    #   (U1 · A_vec) * (V1 · B_vec) where U1=[1,0,0,1] (for A00,A01,A10,A11) and V1=[1,0,0,1]
    #   And it contributes to all 4 output entries C00,C01,C10,C11.
    #
    # So as a rank-1 tensor: U1[i,j] = 1 if (i,j) in {(0,0),(1,1)}, V1[j,k] = 1 if (j,k) in {(0,0),(1,1)}, W1[k,i] = 1 for all (k,i).
    #
    # Let me verify: (U1⊗V1⊗W1)[i,j,k] = U1[i,j]*V1[j,k]*W1[k,i]
    # For (0,0,0): U1[0,0]=1, V1[0,0]=1, W1[0,0]=1 -> 1
    # For (0,0,1): U1[0,0]=1, V1[0,1]=0, W1[1,0]=1 -> 0
    # For (0,1,0): U1[0,1]=0, ... -> 0
    # For (1,1,1): U1[1,1]=1, V1[1,1]=1, W1[1,1]=1 -> 1
    # For (1,1,0): U1[1,1]=1, V1[1,0]=0, ... -> 0
    # For (0,1,1): U1[0,1]=0, ... -> 0
    # For (1,0,0): U1[1,0]=0, ... -> 0
    # For (1,0,1): U1[1,0]=0, ... -> 0
    #
    # So M1 contributes 1 to T[0,0,0] and T[1,1,1]. Good.
    #
    # Now I need the full set of 7 terms. Let me look up or derive them.
    #
    # Actually, let me just directly construct the decomposition using a different approach.
    # I'll use the fact that for 2x2x2 matmul, the rank is 7, and use a known explicit decomposition.
    #
    # Known decomposition of M_{2,2,2} (from Strassen, 1969):
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C00 = p1 - p5 + p4 + p7
    # C01 = p5 + p3
    # C10 = p2 + p4
    # C11 = p1 - p2 + p6 + p3
    #
    # Now, as tensor decomposition terms (each is a rank-1 tensor U_c ⊗ V_c ⊗ W_c):
    # The input vector for A is [a, b, c, d] (flattened row-major: A[0,0], A[0,1], A[1,0], A[1,1])
    # The input vector for B is [e, f, g, h] (flattened row-major: B[0,0], B[0,1], B[1,0], B[1,1])
    # The output vector for C is [C00, C01, C10, C11] (flattened row-major)
    #
    # p1 = (a+d)(e+h):
    #   U_p1 = [1, 0, 0, 1]  (coeffs of a, b, c, d)
    #   V_p1 = [1, 0, 0, 1]  (coeffs of e, f, g, h)
    #   W_p1 = [1, 1, 1, 1]  (coeffs of C00, C01, C10, C11)
    #
    # p2 = (c+d)e:
    #   U_p2 = [0, 0, 1, 1]
    #   V_p2 = [1, 0, 0, 0]
    #   W_p2 = [0, 0, 1, 0]  (appears in C10 with coeff +1, C11 with coeff -1)
    #
    # p3 = a(f-h):
    #   U_p3 = [1, 0, 0, 0]
    #   V_p3 = [0, 1, 0, -1]
    #   W_p3 = [0, 1, 0, 1]  (appears in C01 with +1, C11 with +1)
    #
    # p4 = d(g-e):
    #   U_p4 = [0, 0, 0, 1]
    #   V_p4 = [0, 0, 1, -1]  wait: g-e means B[1,0] - B[0,0] = g - e
    #   V_p4 = [-1, 0, 1, 0]  (coeffs of e, f, g, h: -1 for e, 0 for f, 1 for g, 0 for h)
    #   W_p4 = [1, 0, 1, 0]  (appears in C00 with +1, C10 with +1)
    #
    # p5 = (a+b)h:
    #   U_p5 = [1, 1, 0, 0]
    #   V_p5 = [0, 0, 0, 1]
    #   W_p5 = [-1, 1, 0, 0]  (appears in C00 with -1, C01 with +1)
    #
    # p6 = (c-a)(e+f):
    #   U_p6 = [-1, 0, 1, 0]
    #   V_p6 = [1, 1, 0, 0]
    #   W_p6 = [0, 0, 0, 1]  (appears in C11 with +1)
    #
    # p7 = (b-d)(g+h):
    #   U_p7 = [0, 1, 0, -1]
    #   V_p7 = [0, 0, 1, 1]
    #   W_p7 = [1, 0, 0, 0]  (appears in C00 with +1)
    #
    # Let me verify C00 = p1 - p5 + p4 + p7:
    # p1: W[0]=1
    # p5: W[0]=-1
    # p4: W[0]=1
    # p7: W[0]=1
    # Sum: 1-1+1+1 = 2? That should be 1!
    #
    # Hmm, I think I'm making an error. Let me re-check.
    # C00 = ae + bg
    # p1 = (a+d)(e+h) = ae + ah + de + dh
    # p5 = (a+b)h = ah + bh
    # p4 = d(g-e) = dg - de
    # p7 = (b-d)(g+h) = bg + bh - dg - dh
    #
    # p1 - p5 + p4 + p7 = (ae+ah+de+dh) - (ah+bh) + (dg-de) + (bg+bh-dg-dh)
    # = ae + ah + de + dh - ah - bh + dg - de + bg + bh - dg - dh
    # = ae + bg
    # ✓
    #
    # Now in the tensor decomposition, the coefficient of term c in output entry r is W_c[r].
    # So for C00 (output index 0):
    # W_p1[0] = 1, W_p2[0] = 0, W_p3[0] = 0, W_p4[0] = 1, W_p5[0] = -1, W_p6[0] = 0, W_p7[0] = 1
    # Sum = 1 + 0 + 0 + 1 + (-1) + 0 + 1 = 2?
    #
    # But we need the SUM over all terms of U_c[i,j]*V_c[j,k]*W_c[k,i] to equal 1 for the specific (i,j,k) that corresponds to C00.
    #
    # Oh wait, I see my confusion. The W vector in the tensor decomposition is indexed by (k,i), not just the output row. And the U is indexed by (i,j), V by (j,k).
    #
    # Let me redo this properly.
    #
    # For the 2x2x2 block with local indices (i,j,k) in {0,1}:
    # U_c is a 2x2 matrix (indexed by i,j), V_c is a 2x2 matrix (indexed by j,k), W_c is a 2x2 matrix (indexed by k,i).
    #
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k.
    #
    # p1 = (a+d)(e+h) where a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]:
    # U_p1[0,0] = 1, U_p1[1,1] = 1, all others 0.
    # V_p1[0,0] = 1, V_p1[1,1] = 1, all others 0.
    # W_p1[k,i]: p1 appears in C00, C01, C10, C11 all with coefficient +1.
    #   C00 = C[0,0]: k=0, i=0 -> W_p1[0,0] = 1
    #   C01 = C[0,1]: k=1, i=0 -> W_p1[1,0] = 1
    #   C10 = C[1,0]: k=0, i=1 -> W_p1[0,1] = 1
    #   C11 = C[1,1]: k=1, i=1 -> W_p1[1,1] = 1
    #
    # So W_p1 = [[1,1],[1,1]].
    #
    # Let's verify contribution of p1 to T[0,0,0]:
    # U_p1[0,0]*V_p1[0,0]*W_p1[0,0] = 1*1*1 = 1
    # T[0,1,0]: U_p1[0,1]*V_p1[1,0]*W_p1[0,0] = 0*0*1 = 0
    # T[1,0,0]: U_p1[1,0]*V_p1[0,0]*W_p1[0,1] = 0*1*1 = 0
    # T[1,1,0]: U_p1[1,1]*V_p1[1,0]*W_p1[0,1] = 1*0*1 = 0
    # T[0,0,1]: U_p1[0,0]*V_p1[0,1]*W_p1[1,0] = 1*0*1 = 0
    # T[0,1,1]: U_p1[0,1]*V_p1[1,1]*W_p1[1,0] = 0*1*1 = 0
    # T[1,0,1]: U_p1[1,0]*V_p1[0,1]*W_p1[1,1] = 0*0*1 = 0
    # T[1,1,1]: U_p1[1,1]*V_p1[1,1]*W_p1[1,1] = 1*1*1 = 1
    #
    # So p1 contributes 1 to T[0,0,0] and T[1,1,1]. Good.
    #
    # Now let me write all 7 terms properly:
    #
    # p1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]):
    #   U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,1],[1,1]]
    #
    # p2 = (A[1,0]+A[1,1])B[0,0]:
    #   U = [[0,0],[1,1]], V = [[1,0],[0,0]], 
    #   p2 appears in C10 (+1) and C11 (-1):
    #   C10: k=0,i=1 -> W[0,1]=1; C11: k=1,i=1 -> W[1,1]=-1
    #   W = [[0,1],[0,-1]]
    #
    # p3 = A[0,0](B[0,1]-B[1,1]):
    #   U = [[1,0],[0,0]], V = [[0,1],[0,-1]],
    #   p3 appears in C01 (+1) and C11 (+1):
    #   C01: k=1,i=0 -> W[1,0]=1; C11: k=1,i=1 -> W[1,1]=1
    #   W = [[0,0],[1,1]]
    #
    # p4 = A[1,1](B[1,0]-B[0,0]):
    #   U = [[0,0],[0,1]], V = [[-1,0],[1,0]],
    #   p4 appears in C00 (+1) and C10 (+1):
    #   C00: k=0,i=0 -> W[0,0]=1; C10: k=0,i=1 -> W[0,1]=1
    #   W = [[1,1],[0,0]]
    #
    # p5 = (A[0,0]+A[0,1])B[1,1]:
    #   U = [[1,1],[0,0]], V = [[0,0],[0,1]],
    #   p5 appears in C00 (-1) and C01 (+1):
    #   C00: k=0,i=0 -> W[0,0]=-1; C01: k=1,i=0 -> W[1,0]=1
    #   W = [[-1,0],[1,0]]
    #
    # p6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1]):
    #   U = [[-1,0],[1,0]], V = [[1,1],[0,0]],
    #   p6 appears in C11 (+1):
    #   C11: k=1,i=1 -> W[1,1]=1
    #   W = [[0,0],[0,1]]
    #
    # p7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1]):
    #   U = [[0,1],[0,-1]], V = [[0,0],[1,1]],
    #   p7 appears in C00 (+1):
    #   C00: k=0,i=0 -> W[0,0]=1
    #   W = [[1,0],[0,0]]
    #
    # Let me verify T[0,0,0] = sum of contributions:
    # p1: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # p2: U[0,0]*V[0,0]*W[0,0] = 0*1*0 = 0
    # p3: U[0,0]*V[0,0]*W[0,0] = 1*0*0 = 0
    # p4: U[0,0]*V[0,0]*W[0,0] = 0*(-1)*1 = 0
    # p5: U[0,0]*V[0,0]*W[0,0] = 1*0*(-1) = 0
    # p6: U[0,0]*V[0,0]*W[0,0] = -1*1*0 = 0
    # p7: U[0,0]*V[0,0]*W[0,0] = 0*0*1 = 0
    # Sum = 1 ✓
    #
    # T[0,0,1] (i=0,j=0,k=1):
    # p1: U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0
    # p2: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0
    # p3: U[0,0]*V[0,1]*W[1,0] = 1*1*1 = 1
    # p4: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0
    # p5: U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0
    # p6: U[0,0]*V[0,1]*W[1,0] = -1*1*0 = 0
    # p7: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0
    # Sum = 1 ✓
    #
    # T[0,1,0] (i=0,j=1,k=0):
    # p1: U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
    # p2: U[0,1]*V[1,0]*W[0,0] = 0*0*0 = 0
    # p3: U[0,1]*V[1,0]*W[0,0] = 0*0*0 = 0
    # p4: U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0
    # p5: U[0,1]*V[1,0]*W[0,0] = 1*0*(-1) = 0
    # p6: U[0,1]*V[1,0]*W[0,0] = 0*0*0 = 0
    # p7: U[0,1]*V[1,0]*W[0,0] = 1*1*1 = 1
    # Sum = 1 ✓
    #
    # T[0,1,1] (i=0,j=1,k=1):
    # p1: U[0,1]*V[1,1]*W[1,0] = 0*1*1 = 0
    # p2: U[0,1]*V[1,1]*W[1,0] = 0*0*0 = 0
    # p3: U[0,1]*V[1,1]*W[1,0] = 0*(-1)*1 = 0
    # p4: U[0,1]*V[1,1]*W[1,0] = 0*0*0 = 0
    # p5: U[0,1]*V[1,1]*W[1,0] = 1*1*1 = 1
    # p6: U[0,1]*V[1,1]*W[1,0] = 0*0*0 = 0
    # p7: U[0,1]*V[1,1]*W[1,0] = 1*1*0 = 0
    # Sum = 1 ✓
    #
    # T[1,0,0] (i=1,j=0,k=0):
    # p1: U[1,0]*V[0,0]*W[0,1] = 0*1*1 = 0
    # p2: U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1
    # p3: U[1,0]*V[0,0]*W[0,1] = 0*0*0 = 0
    # p4: U[1,0]*V[0,0]*W[0,1] = 0*(-1)*1 = 0
    # p5: U[1,0]*V[0,0]*W[0,1] = 0*0*0 = 0
    # p6: U[1,0]*V[0,0]*W[0,1] = 1*1*0 = 0
    # p7: U[1,0]*V[0,0]*W[0,1] = 0*0*0 = 0
    # Sum = 1 ✓
    #
    # T[1,0,1] (i=1,j=0,k=1):
    # p1: U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # p2: U[1,0]*V[0,1]*W[1,1] = 1*0*(-1) = 0
    # p3: U[1,0]*V[0,1]*W[1,1] = 0*1*1 = 0
    # p4: U[1,0]*V[0,1]*W[1,1] = 0*0*0 = 0
    # p5: U[1,0]*V[0,1]*W[1,1] = 0*0*0 = 0
    # p6: U[1,0]*V[0,1]*W[1,1] = 1*1*1 = 1
    # p7: U[1,0]*V[0,1]*W[1,1] = 0*0*0 = 0
    # Sum = 1 ✓
    #
    # T[1,1,0] (i=1,j=1,k=0):
    # p1: U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0
    # p2: U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0
    # p3: U[1,1]*V[1,0]*W[0,1] = 0*0*0 = 0
    # p4: U[1,1]*V[1,0]*W[0,1] = 1*1*1 = 1
    # p5: U[1,1]*V[1,0]*W[0,1] = 0*0*0 = 0
    # p6: U[1,1]*V[1,0]*W[0,1] = 0*0*0 = 0
    # p7: U[1,1]*V[1,0]*W[0,1] = -1*1*0 = 0
    # Sum = 1 ✓
    #
    # T[1,1,1] (i=1,j=1,k=1):
    # p1: U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    # p2: U[1,1]*V[1,1]*W[1,1] = 1*0*(-1) = 0
    # p3: U[1,1]*V[1,1]*W[1,1] = 0*(-1)*1 = 0
    # p4: U[1,1]*V[1,1]*W[1,1] = 1*0*0 = 0
    # p5: U[1,1]*V[1,1]*W[1,1] = 0*1*0 = 0
    # p6: U[1,1]*V[1,1]*W[1,1] = 0*0*1 = 0
    # p7: U[1,1]*V[1,1]*W[1,1] = -1*1*0 = 0
    # Sum = 1 ✓
    #
    # All 8 entries verified! The Strassen decomposition is correct.

    # Now for the 1x2 * 2x2 block (rank 4):
    # A is 1x2: A[0,0]=a, A[0,1]=b
    # B is 2x2: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C is 1x2: C[0,0]=ae+bg, C[0,1]=af+bh
    #
    # T[i,j,k] where i in {0}, j in {0,1}, k in {0,1}:
    # T[0,0,0] = 1 (a*e)
    # T[0,1,0] = 1 (b*g)
    # T[0,0,1] = 1 (a*f)
    # T[0,1,1] = 1 (b*h)
    #
    # Rank 4 decomposition (4 trivial terms):
    # c0: U=[1,0], V=[1,0;0,0], W=[1,0]  -> T[0,0,0]
    # c1: U=[0,1], V=[0,0;1,0], W=[1,0]  -> T[0,1,0]
    # c2: U=[1,0], V=[0,1;0,0], W=[0,1]  -> T[0,0,1]
    # c3: U=[0,1], V=[0,0;0,1], W=[0,1]  -> T[0,1,1]
    #
    # Verify T[0,0,0]: c0: 1*1*1=1, others 0. Sum=1 ✓
    # T[0,1,0]: c1: 1*1*1=1, others 0. Sum=1 ✓
    # T[0,0,1]: c2: 1*1*1=1, others 0. Sum=1 ✓
    # T[0,1,1]: c3: 1*1*1=1, others 0. Sum=1 ✓

    # Now build the full decomposition:
    # A is 3x4: rows [0,1,2], cols [0,1,2,3]
    # B is 4x6: rows [0,1,2,3], cols [0,1,2,3,4,5]
    # C is 3x6: rows [0,1,2], cols [0,1,2,3,4,5]
    #
    # Block structure:
    # A rows: [0,1] (top 2), [2] (bottom 1)
    # A cols: [0,1] (left 2), [2,3] (right 2)
    # B rows: [0,1] (top 2), [2,3] (bottom 2)
    # B cols: [0,1], [2,3], [4,5]
    #
    # 6 Strassen blocks (2x2 * 2x2):
    # (A_rows=[0,1], A_cols=[0,1], B_rows=[0,1], B_cols=[0,1])
    # (A_rows=[0,1], A_cols=[0,1], B_rows=[0,1], B_cols=[2,3])
    # (A_rows=[0,1], A_cols=[0,1], B_rows=[0,1], B_cols=[4,5])
    # (A_rows=[0,1], A_cols=[2,3], B_rows=[2,3], B_cols=[0,1])
    # (A_rows=[0,1], A_cols=[2,3], B_rows=[2,3], B_cols=[2,3])
    # (A_rows=[0,1], A_cols=[2,3], B_rows=[2,3], B_cols=[4,5])
    #
    # 6 direct blocks (1x2 * 2x2):
    # (A_rows=[2], A_cols=[0,1], B_rows=[0,1], B_cols=[0,1])
    # (A_rows=[2], A_cols=[0,1], B_rows=[0,1], B_cols=[2,3])
    # (A_rows=[2], A_cols=[0,1], B_rows=[0,1], B_cols=[4,5])
    # (A_rows=[2], A_cols=[2,3], B_rows=[2,3], B_cols=[0,1])
    # (A_rows=[2], A_cols=[2,3], B_rows=[2,3], B_cols=[2,3])
    # (A_rows=[2], A_cols=[2,3], B_rows=[2,3], B_cols=[4,5])
    #
    # Total rank: 6*7 + 6*4 = 42 + 24 = 66

    total_rank = 6 * 7 + 6 * 4  # 66
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 66
    
    # Strassen terms (local 2x2 matrices):
    # Each term is (U_local 2x2, V_local 2x2, W_local 2x2)
    strassen_terms = [
        # p1
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,1],[1,1]]),
        # p2
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        # p3
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        # p4
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        # p5
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        # p6
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # p7
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Direct 1x2*2x2 terms:
    # U_local is 1x2, V_local is 2x2, W_local is 2x1
    direct_terms = [
        # c0: a*e
        ([1,0], [[1,0],[0,0]], [1,0]),
        # c1: b*g
        ([0,1], [[0,0],[1,0]], [1,0]),
        # c2: a*f
        ([1,0], [[0,1],[0,0]], [0,1]),
        # c3: b*h
        ([0,1], [[0,0],[0,1]], [0,1]),
    ]
    
    col = 0
    
    # Add 6 Strassen blocks
    strassen_blocks = [
        # (a_row_start, a_col_start, b_row_start, b_col_start)
        (0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2),  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 0, 0, 4),  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
        (0, 2, 2, 0),  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 2, 2, 2),  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        (0, 2, 2, 4),  # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    ]
    
    for (ai, aj, bi, bk) in strassen_blocks:
        for (U_loc, V_loc, W_loc) in strassen_terms:
            # U_loc[i_local, j_local] -> U[ai + i_local, aj + j_local, col]
            for li in range(2):
                for lj in range(2):
                    U[(ai + li) * m + (aj + lj)][col] += U_loc[li][lj]
            # V_loc[j_local, k_local] -> V[bi + j_local, bk + k_local, col]
            for lj in range(2):
                for lk in range(2):
                    V[(bi + lj) * p + (bk + lk)][col] += V_loc[lj][lk]
            # W_loc[k_local, i_local] -> W[bk + k_local, ai + i_local, col]
            for lk in range(2):
                for li in range(2):
                    W[(bk + lk) * n + (ai + li)][col] += W_loc[lk][li]
            col += 1
    
    # Add 6 direct blocks
    direct_blocks = [
        # (a_row, a_col_start, b_row_start, b_col_start)
        (2, 0, 0, 0),  # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
        (2, 0, 0, 2),  # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
        (2, 0, 0, 4),  # A[2,0:2] * B[0:2,4:6] -> C[2,4:6]
        (2, 2, 2, 0),  # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
        (2, 2, 2, 2),  # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
        (2, 2, 2, 4),  # A[2,2:4] * B[2:4,4:6] -> C[2,4:6]
    ]
    
    for (ai, aj, bi, bk) in direct_blocks:
        for (U_loc, V_loc, W_loc) in direct_terms:
            # U_loc[j_local] -> U[ai, aj + j_local, col]
            for lj in range(2):
                U[ai * m + (aj + lj)][col] += U_loc[lj]
            # V_loc[j_local, k_local] -> V[bi + j_local, bk + k_local, col]
            for lj in range(2):
                for lk in range(2):
                    V[(bi + lj) * p + (bk + lk)][col] += V_loc[lj][lk]
            # W_loc[k_local] -> W[bk + k_local, ai, col]
            for lk in range(2):
                W[(bk + lk) * n + ai][col] += W_loc[lk]
            col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Structured block decomposition using Strassen (rank 7) for 2x2 blocks
    # and direct rank-4 decompositions for 1x2 * 2x2 blocks.
    # Total rank: 6*7 + 6*4 = 66
    
    total_rank = 6 * 7 + 6 * 4  # 66
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 66
    
    # Strassen terms for 2x2 * 2x2 (verified above)
    strassen_terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,1],[1,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Direct terms for 1x2 * 2x2 (rank 4)
    direct_terms = [
        ([1,0], [[1,0],[0,0]], [1,0]),
        ([0,1], [[0,0],[1,0]], [1,0]),
        ([1,0], [[0,1],[0,0]], [0,1]),
        ([0,1], [[0,0],[0,1]], [0,1]),
    ]
    
    col = 0
    
    # 6 Strassen blocks: A[0:2, *] * B[*] -> C[0:2, *]
    strassen_blocks = [
        (0, 0, 0, 0),
        (0, 0, 0, 2),
        (0, 0, 0, 4),
        (0, 2, 2, 0),
        (0, 2, 2, 2),
        (0, 2, 2, 4),
    ]
    
    for (ai, aj, bi, bk) in strassen_blocks:
        for (U_loc, V_loc, W_loc) in strassen_terms:
            for li in range(2):
                for lj in range(2):
                    U[(ai + li) * m + (aj + lj)][col] += U_loc[li][lj]
            for lj in range(2):
                for lk in range(2):
                    V[(bi + lj) * p + (bk + lk)][col] += V_loc[lj][lk]
            for lk in range(2):
                for li in range(2):
                    W[(bk + lk) * n + (ai + li)][col] += W_loc[lk][li]
            col += 1
    
    # 6 direct blocks: A[2, *] * B[*] -> C[2, *]
    direct_blocks = [
        (2, 0, 0, 0),
        (2, 0, 0, 2),
        (2, 0, 0, 4),
        (2, 2, 2, 0),
        (2, 2, 2, 2),
        (2, 2, 2, 4),
    ]
    
    for (ai, aj, bi, bk) in direct_blocks:
        for (U_loc, V_loc, W_loc) in direct_terms:
            for lj in range(2):
                U[ai * m + (aj + lj)][col] += U_loc[lj]
            for lj in range(2):
                for lk in range(2):
                    V[(bi + lj) * p + (bk + lk)][col] += V_loc[lj][lk]
            for lk in range(2):
                W[(bk + lk) * n + ai][col] += W_loc[lk]
            col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify the decomposition
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook (should not happen)
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
