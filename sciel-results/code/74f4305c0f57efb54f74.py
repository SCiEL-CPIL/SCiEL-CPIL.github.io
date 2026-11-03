import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Partition indices
    # n=3: pairs in i
    i_pairs = [(1, 2)]  # {1,2}
    i_singles = [0]
    
    # m=5: pairs in j/k
    jk_pairs = [(0, 1), (2, 3)]
    jk_singles = [4]
    
    # p=7: pairs in l
    l_pairs = [(0, 1), (2, 3), (4, 5)]
    l_singles = [6]
    
    # Collect all rank-1 terms as (U_idx, V_idx, W_idx, coeff)
    # U index: i*m + j
    # V index: j*p + k  (but j==k in the tensor support)
    # W index: k*n + i  (but k==l in the tensor support)
    # Actually: U[i*m+j], V[j*p+l], W[l*n+i]
    
    terms = []  # list of (u_idx, v_idx, w_idx, coeff)
    
    # Strassen blocks
    for (i1, i2) in i_pairs:
        for (j1, j2) in jk_pairs:
            for (l1, l2) in l_pairs:
                # 7 Strassen terms for this 2x2x2 block
                # The 7 multiplications are:
                # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,l1]+B[j2,l2])
                # M2 = (A[i1,j1]+A[i1,j2]) * (B[j2,l1]-B[j2,l2])  -- wait, I need to be careful
                
                # Standard Strassen for 2x2:
                # C11 = M1 - M2 + M3 - M4 + M7
                # C12 = M5 + M4
                # C21 = M6 + M3
                # C22 = M1 - M5 + M2 + M6
                # where:
                # M1 = (a11+a22)(b11+b22)
                # M2 = (a21+a22)b11
                # M3 = a11(b21-b22)
                # M4 = a22(b11-b12)
                # M5 = (a11+a12)b22
                # M6 = (a12-a21)(b11+b21)
                # M7 = (a21-a11)(b12+b22)
                
                # In our context:
                # a11 = A[i1,j1], a12 = A[i1,j2]
                # a21 = A[i2,j1], a22 = A[i2,j2]
                # b11 = B[j1,l1], b12 = B[j1,l2]
                # b21 = B[j2,l1], b22 = B[j2,l2]
                # c11 = C[i1,l1], c12 = C[i1,l2]
                # c21 = C[i2,l1], c22 = C[i2,l2]
                
                # M1: U has +1 at (i1,j1) and +1 at (i2,j2)
                #     V has +1 at (j1,l1) and +1 at (j2,l2)
                #     W has +1 at (l1,i1), +1 at (l1,i2), +1 at (l2,i1), +1 at (l2,i2)
                #     Wait, W index is l*n + i. So W[l1*n+i1], W[l1*n+i2], W[l2*n+i1], W[l2*n+i2]
                #     But M1 contributes to all of c11, c12, c21, c22 with coefficient +1
                #     So W should have +1 at all four positions? No...
                
                # Actually, each M_k is a scalar. The contribution to C is:
                # C = linear combination of M_k's
                # Each M_k = (linear form in A) * (linear form in B)
                # The tensor term is: (coeff in A) * (coeff in B) * (coeff in C)
                # For M1: A-part = A[i1,j1] + A[i2,j2], B-part = B[j1,l1] + B[j2,l2]
                # C-part: M1 contributes +1 to C[i1,l1], +1 to C[i1,l2], +1 to C[i2,l1], +1 to C[i2,l2]
                # Wait no. M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
                # This is NOT equal to any single C entry. It's a combination.
                # The point is that the 7 M_k's are used to compute the 4 C entries.
                
                # So the tensor decomposition term for M1 is:
                # U: +1 at (i1,j1), +1 at (i2,j2)
                # V: +1 at (j1,l1), +1 at (j2,l2)  -- wait, V index is j*p+k where j is row of B, k is col of B
                #     B[j1,l1] means row j1, col l1 of B. So V index = j1*p + l1
                #     B[j2,l2] means row j2, col l2 of B. So V index = j2*p + l2
                # W: The M1 term contributes to C[i1,l1], C[i1,l2], C[i2,l1], C[i2,l2]
                #     But with what coefficients? M1 appears in:
                #     C11 = M1 - M2 + M3 - M4 + M7 → coeff +1 for C[i1,l1]
                #     C12 = M5 + M4 → M1 not present
                #     C21 = M6 + M3 → M1 not present  
                #     C22 = M1 - M5 + M2 + M6 → coeff +1 for C[i2,l2]
                #     
                #     So M1 contributes +1 to C[i1,l1] and +1 to C[i2,l2]
                #     W index for C[i1,l1] is l1*n + i1
                #     W index for C[i2,l2] is l2*n + i2
                #     So W: +1 at (l1*n+i1), +1 at (l2*n+i2)
                
                # Let me redo this more carefully.
                
                # The 7 Strassen products and their contributions to C:
                # M1 = (a11+a22)(b11+b22): contributes to C11 (+1), C22 (+1)
                # M2 = (a21+a22)b11: contributes to C11 (-1), C22 (+1)
                # M3 = a11(b21-b22): contributes to C11 (+1), C21 (+1)
                # M4 = a22(b11-b12): contributes to C11 (-1), C12 (+1)
                # M5 = (a11+a12)b22: contributes to C12 (+1), C22 (-1)
                # M6 = (a12-a21)(b11+b21): contributes to C21 (+1), C22 (+1)
                # M7 = (a21-a11)(b12+b22): contributes to C11 (+1)
                
                # Let me verify:
                # C11 = M1 - M2 + M3 - M4 + M7
                # = (a11+a22)(b11+b22) - (a21+a22)b11 + a11(b21-b22) - a22(b11-b12) + (a21-a11)(b12+b22)
                # = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a21*b11 - a22*b11 + a11*b21 - a11*b22 - a22*b11 + a22*b12 + a21*b12 + a21*b22 - a11*b12 - a11*b22
                # = a11*b11 + (a11*b22 - a11*b22 - a11*b22) + (a22*b11 - a22*b11 - a22*b11) + a22*b22 - a21*b11 + a11*b21 + a22*b12 + a21*b12 + a21*b22 - a11*b12
                # This is getting messy. Let me just trust the standard Strassen formulas.
                
                # I'll use the standard Strassen decomposition and map it to our tensor.
                
                # For the block with A-indices (i1,i2)×(j1,j2), B-indices (j1,j2)×(l1,l2), C-indices (i1,i2)×(l1,l2):
                
                # U indices: (i*m+j) for i in {i1,i2}, j in {j1,j2}
                # V indices: (j*p+l) for j in {j1,j2}, l in {l1,l2}
                # W indices: (l*n+i) for l in {l1,l2}, i in {i1,i2}
                
                def u_idx(i, j): return i * m + j
                def v_idx(j, l): return j * p + l
                def w_idx(l, i): return l * n + i
                
                # M1
                terms.append((u_idx(i1,j1), v_idx(j1,l1), w_idx(l1,i1), 1))
                terms.append((u_idx(i2,j2), v_idx(j1,l1), w_idx(l1,i1), 1))
                terms.append((u_idx(i1,j1), v_idx(j2,l2), w_idx(l1,i1), 1))
                terms.append((u_idx(i2,j2), v_idx(j2,l2), w_idx(l1,i1), 1))
                terms.append((u_idx(i1,j1), v_idx(j1,l1), w_idx(l2,i2), 1))
                terms.append((u_idx(i2,j2), v_idx(j1,l1), w_idx(l2,i2), 1))
                terms.append((u_idx(i1,j1), v_idx(j2,l2), w_idx(l2,i2), 1))
                terms.append((u_idx(i2,j2), v_idx(j2,l2), w_idx(l2,i2), 1))
                
                # This is getting complicated. Let me use a different approach.
                
                # Actually, each rank-1 term in the tensor decomposition is of the form:
                # u_vec ⊗ v_vec ⊗ w_vec
                # where u_vec is in R^{n*m}, v_vec is in R^{m*p}, w_vec is in R^{p*n}
                # and the tensor entry T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # For Strassen, each of the 7 products M_k gives a rank-1 term where:
                # U has the coefficients of the linear form in A
                # V has the coefficients of the linear form in B
                # W has the coefficients of the linear form in C (how M_k contributes to each C entry)
                
                # So for M1 = (a11+a22)(b11+b22):
                # U: +1 at (i1,j1), +1 at (i2,j2)
                # V: +1 at (j1,l1), +1 at (j2,l2)
                # W: +1 at (l1,i1), +1 at (l2,i2)  [since M1 contributes +1 to C[i1,l1] and +1 to C[i2,l2]]
                
                # For M2 = (a21+a22)b11:
                # U: +1 at (i2,j1), +1 at (i2,j2)
                # V: +1 at (j1,l1)
                # W: -1 at (l1,i1), +1 at (l2,i2)  [M2 contributes -1 to C[i1,l1] and +1 to C[i2,l2]]
                
                # etc.
                
                # Let me define the 7 Strassen terms properly.
                # I'll use a list of (U_coeffs, V_coeffs, W_coeffs) where each is a dict from index to coeff.
                
                strassen_terms = []
                
                # M1
                strassen_terms.append((
                    {(i1,j1):1, (i2,j2):1},
                    {(j1,l1):1, (j2,l2):1},
                    {(l1,i1):1, (l2,i2):1}
                ))
                # M2
                strassen_terms.append((
                    {(i2,j1):1, (i2,j2):1},
                    {(j1,l1):1},
                    {(l1,i1):-1, (l2,i2):1}
                ))
                # M3
                strassen_terms.append((
                    {(i1,j1):1},
                    {(j2,l1):1, (j2,l2):-1},
                    {(l1,i1):1, (l1,i2):1}
                ))
                # M4
                strassen_terms.append((
                    {(i2,j2):1},
                    {(j1,l1):1, (j1,l2):-1},
                    {(l1,i1):-1, (l2,i1):1}
                ))
                # M5
                strassen_terms.append((
                    {(i1,j1):1, (i1,j2):1},
                    {(j2,l2):1},
                    {(l2,i1):1, (l2,i2):-1}
                ))
                # M6
                strassen_terms.append((
                    {(i1,j2):1, (i2,j1):-1},
                    {(j1,l1):1, (j2,l1):1},
                    {(l1,i2):1, (l2,i2):1}
                ))
                # M7
                strassen_terms.append((
                    {(i2,j1):1, (i1,j1):-1},
                    {(j1,l2):1, (j2,l2):1},
                    {(l1,i1):1}
                ))
                
                for (u_coeff, v_coeff, w_coeff) in strassen_terms:
                    # This is not a single rank-1 term; it's a sum of rank-1 terms
                    # because U, V, W each have multiple non-zero entries
                    # We need to expand this into actual rank-1 terms
                    # OR, we can keep it as is if the decomposition allows it
                    
                    # Actually, a rank-1 term in the tensor T[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]
                    # has U[a,r], V[b,r], W[c,r] all being scalars for a fixed r.
                    # The "term" above has U with 2 non-zeros, V with 2 non-zeros, W with 2 non-zeros.
                    # This represents a sum of 2*2*2 = 8 rank-1 terms, which defeats the purpose!
                    
                    # I think I'm confusing the bilinear form decomposition with the tensor decomposition.
                    # Let me reconsider.
                    pass
    
    # I'm making this too complicated. Let me look up the correct way to do this.
    # 
    # The key insight: in the tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r,
    # each term u_r ⊗ v_r ⊗ w_r has u_r, v_r, w_r as vectors.
    # For the Strassen algorithm, each of the 7 "multiplications" corresponds to ONE rank-1 term
    # where u_r, v_r, w_r can have multiple non-zero entries.
    # 
    # So for M1 = (a11+a22)(b11+b22):
    # u_r has +1 at positions corresponding to a11 and a22
    # v_r has +1 at positions corresponding to b11 and b22
    # w_r has +1 at positions corresponding to the C entries that M1 contributes to
    # 
    # This IS a valid rank-1 term in the tensor decomposition! The tensor entry
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For r corresponding to M1:
    # T[a,b,c] includes U[a,r]*V[b,r]*W[c,r]
    # If a corresponds to a11, b corresponds to b11, c corresponds to C[i1,l1]:
    # U[a11,r]=1, V[b11,r]=1, W[C[i1,l1],r]=1, so contribution = 1*1*1 = 1 ✓
    # If a corresponds to a11, b corresponds to b22, c corresponds to C[i1,l1]:
    # U[a11,r]=1, V[b22,r]=1, W[C[i1,l1],r]=1, so contribution = 1*1*1 = 1
    # But we want T[a11, b22, C[i1,l1]] to be... let's see.
    # T[a,b,c] = 1 if the indices are consistent for matrix multiplication.
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T = 1 if j=j' and k=k' and i=i'
    # For a=a11 (i1,j1), b=b22 (j2,l2), c=C[i1,l1] (l1,i1):
    # j1 != j2, so T = 0. Good, we don't want a contribution here.
    # But U[a11,r]=1, V[b22,r]=1, W[C[i1,l1],r]=1 gives 1*1*1=1 ≠ 0. PROBLEM!
    
    # I see the issue. The Strassen "multiplication" M1 = (a11+a22)(b11+b22) is a scalar,
    # and it contributes to multiple C entries. But in the tensor decomposition,
    # each rank-1 term contributes to ALL (a,b,c) entries where U[a,r]*V[b,r]*W[c,r] ≠ 0.
    # 
    # So I need to be more careful. Let me reconsider.
    # 
    # Actually, I think the correct interpretation is:
    # The 7 Strassen products are 7 rank-1 terms in the tensor decomposition.
    # For M1: u_r = e_{a11} + e_{a22}, v_r = e_{b11} + e_{b22}, w_r = e_{c11} + e_{c22}
    # (where c11 = C[i1,l1], c22 = C[i2,l2])
    # 
    # The contribution to T[a,b,c] from this term is u_r[a] * v_r[b] * w_r[c].
    # For a=a11, b=b11, c=c11: 1*1*1 = 1
    # For a=a11, b=b11, c=c22: 1*1*1 = 1
    # For a=a11, b=b22, c=c11: 1*1*1 = 1
    # For a=a11, b=b22, c=c22: 1*1*1 = 1
    # For a=a22, b=b11, c=c11: 1*1*1 = 1
    # For a=a22, b=b11, c=c22: 1*1*1 = 1
    # For a=a22, b=b22, c=c11: 1*1*1 = 1
    # For a=a22, b=b22, c=c22: 1*1*1 = 1
    # 
    # But the true tensor has T=1 only when the indices are consistent:
    # T[a11,b11,c11]: a11=(i1,j1), b11=(j1,l1), c11=(l1,i1). j1=j1, l1=l1, i1=i1. T=1 ✓
    # T[a11,b11,c22]: a11=(i1,j1), b11=(j1,l1), c22=(l2,i2). j1=j1, but l1≠l2. T=0 ✗
    # 
    # So the Strassen term M1 as I've defined it gives wrong contributions!
    # 
    # I think the issue is that I'm not correctly mapping the Strassen algorithm to the tensor decomposition.
    # 
    # Let me think about this differently. The Strassen algorithm computes:
    # C = A*B using 7 scalar multiplications.
    # Each scalar multiplication M_k = (linear form in A) * (linear form in B)
    # Then C = linear combination of the M_k's.
    # 
    # In the tensor decomposition, we want T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[a,b,c] = sum_r u_r[a] v_r[b] w_r[c]
    # 
    # For the Strassen algorithm on a 2x2 block:
    # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,l1] + B[j2,l2])
    # This M1 contributes to C[i1,l1] with coefficient +1 and to C[i2,l2] with coefficient +1.
    # 
    # So the rank-1 term corresponding to M1 should be:
    # u_r = e_{(i1,j1)} + e_{(i2,j2)}
    # v_r = e_{(j1,l1)} + e_{(j2,l2)}
    # w_r = e_{(l1,i1)} + e_{(l2,i2)}
    # 
    # But as I showed, this gives wrong contributions to T.
    # 
    # I think the resolution is that the Strassen algorithm is a decomposition of the BILINEAR MAP,
    # not directly of the tensor. The tensor T encodes the bilinear map (A,B) -> C.
    # 
    # Let me look at this from a different angle. Maybe I should just implement the 99-rank solution
    # that was measured to work, using a different formulation.
    
    # Let me try a completely different approach: just use the schoolbook for all entries,
    # but for the 6 Strassen blocks, use 7 terms instead of 8.
    # 
    # The schoolbook has 105 terms. For each of the 6 blocks, we replace 8 terms with 7 terms.
    # Net: 105 - 6*8 + 6*7 = 105 - 48 + 42 = 99. ✓
    
    # To do this, I need to identify which 8 schoolbook terms are in each block,
    # and replace them with 7 Strassen terms.
    
    # Schoolbook term for (i,j) is: U[i*m+j]=1, V[j*p+j]=1, W[j*n+i]=1
    # Wait, that's not right either. Let me re-examine the seed code.
    
    # Seed code:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # for i in range(n), j in range(m), k in range(p)
    
    # So the schoolbook has n*m*p = 3*5*7 = 105 terms.
    # Each term is indexed by (i,j,k) and has:
    # U[i*m+j] = 1
    # V[j*p+k] = 1
    # W[k*n+i] = 1
    
    # The tensor entry T[a,b,c] where a=i*m+j, b=j'*p+k', c=k''*n+i'
    # gets contribution from term (i,j,k) if:
    # U[a, term] = 1 iff a = i*m+j iff i'=i, j'=j
    # V[b, term] = 1 iff b = j*p+k iff j''=j, k''=k
    # W[c, term] = 1 iff c = k*n+i iff k'''=k, i'''=i
    # So T gets contribution 1 iff i'=i, j'=j, j''=j, k''=k, k'''=k, i'''=i
    # i.e., iff i'=i, j'=j=j'', k''=k=k''', i'''=i
    # This is the correct condition for the matrix multiplication tensor.
    
    # Now, for a 2x2x2 Strassen block with i∈{i1,i2}, j∈{j1,j2}, k∈{k1,k2}:
    # The 8 schoolbook terms are (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i1,j2,k2),
    # (i2,j1,k1), (i2,j1,k2), (i2,j2,k1), (i2,j2,k2)
    # 
    # Wait, but in the schoolbook, j and k are independent! The term (i,j,k) has
    # U[i*m+j], V[j*p+k], W[k*n+i].
    # 
    # For the tensor to be non-zero, we need the indices to be consistent:
    # T[a,b,c] = 1 iff the (i,j) from a matches the j from b, and the k from b matches the k from c, etc.
    # 
    # Actually, I think I've been overthinking this. Let me just trust that the measured
    # result of 99 is achievable and implement it by carefully constructing the Strassen terms.
    
    # Let me try yet another approach: use numpy to verify my decomposition.
    
    # For now, let me just return the schoolbook solution (rank 105) to ensure correctness,
    # and then I'll iterate to improve it.
    
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
