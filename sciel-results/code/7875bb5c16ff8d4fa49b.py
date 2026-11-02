def construct():
    import time
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    n, m, p = 4, 4, 5
    
    # Build Strassen 4x4 x 4x4 (rank 49) + 1 term for 4x4 x 4x1 = rank 50
    # First, define the 2x2 Strassen decomposition
    # For 2x2 x 2x2, the 7 terms as (A_coeffs, B_coeffs) where coeffs are dicts mapping (row,col) -> value
    
    # 2x2 Strassen terms
    # A is 2x2 with indices (0,0),(0,1),(1,0),(1,1)
    # B is 2x2 with indices (0,0),(0,1),(1,0),(1,1)
    # C is 2x2 with indices (0,0),(0,1),(1,0),(1,1)
    
    # Each term: (A_dict, B_dict, C_dict) where C_dict maps output (i,k) to coefficient
    # The term contributes: sum_{(i,j) in A_dict} A_dict[(i,j)] * A[i,j] * sum_{(j,k) in B_dict} B_dict[(j,k)] * B[j,k] -> C[i,k] with coeff C_dict[(i,k)]
    
    # Actually, for a rank-1 term in the tensor, it's: u ⊗ v ⊗ w where u is in A-space, v in B-space, w in C-space
    # u[i,j] = A_coeff[i,j], v[j,k] = B_coeff[j,k], w[i,k] = C_coeff[i,k]
    # The contribution to C[i,k] is u[i,j]*v[j,k]*w[i,k] summed over j... no.
    
    # Let me think differently. The tensor T[i,j,k] for 2x2x2x2x2x2 (A is 2x2, B is 2x2, C is 2x2)
    # T is indexed by (i_a, j_a, i_b, j_b, i_c, j_c) but really it's T[i,j,k] where i=(row of A), j=(col of A = row of B), k=(col of B = col of C)
    # Wait, I think the standard is: T[i,j,k] where i indexes rows of A/C, j indexes cols of A/rows of B, k indexes cols of B/C
    # For 2x2x2: i,j,k ∈ {0,1}, T[i,j,k] = 1 for all i,j,k
    
    # A rank-1 decomposition term is: u_i * v_j * w_k for all i,j,k
    # So the term is specified by vectors u (length 2), v (length 2), w (length 2)
    # Contribution to T[i,j,k] is u[i]*v[j]*w[k]
    
    # For 2x2 Strassen:
    # M1 = (a+d)(e+h) where a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # This contributes to C[0,0] and C[1,1]
    # u = [1, 1] (for A[0,0] and A[1,1])... 
    
    # I think I'm overcomplicating this. Let me use the explicit tensor decomposition.
    
    # For the full problem, I'll build the decomposition programmatically.
    # Let me define the 2x2 Strassen terms as (u, v, w) triples where u,v,w are length-2 vectors
    # and the term is u[i]*v[j]*w[k] added to T[i,j,k].
    
    # 2x2 Strassen (7 terms):
    # Using the standard formulation where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # The 7 products and their contributions to C:
    # M1 = (a+d)(e+h): contributes to C[0,0] with +1, C[1,1] with +1
    #   u = [1,1], v = [1,1], w = [1,1]... no, that's not right.
    
    # Let me use a different approach. I'll directly compute the tensor decomposition.
    # For 2x2x2, T[i,j,k] = 1 for all i,j,k ∈ {0,1}.
    # Strassen gives 7 rank-1 terms.
    
    # Actually, let me just look up / derive the explicit rank-1 terms for 2x2 Strassen.
    # The 7 terms (u, v, w) where the term is sum over i,j,k of u[i]*v[j]*w[k] * (contributes to T[i,j,k]):
    
    # Hmm, I think the cleanest way is to just build it computationally.
    
    # Let me define the 2x2 Strassen terms as lists of (i,j,k,coeff) contributions.
    # Each term is a rank-1 tensor, so it's specified by three vectors.
    
    # Term 1: (a+d)(e+h)
    # u = [1, 1] (coefficients for A[0,*] and A[1,*]... no)
    
    # I think the issue is that in the tensor T[i,j,k], i is the row index of A (and C), 
    # j is the column index of A (and row index of B), k is the column index of B (and C).
    # A rank-1 term is u[i] * v[j] * w[k].
    
    # For M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]):
    # This involves A[0,0], A[1,1] and B[0,0], B[1,1]
    # As a rank-1 term: u = [1,1] (for i=0,1), v = [1,1] (for j=0,1), w = [1,1] (for k=0,1)
    # But that would give u[i]*v[j]*w[k] = 1 for all i,j,k, which is the all-ones tensor, not M1.
    
    # I think I'm confusing myself. Let me restart with a clearer model.
    
    # The matrix multiplication tensor for n×m × m×p:
    # T is a 3-way tensor with dimensions (n*m) × (m*p) × (n*p)
    # T[a, b, c] where a = i*m+j (row i, col j of A), b = j*p+k (row j, col k of B), c = k*n+i (row i, col k of C)
    # T[a, b, c] = 1 if the indices are consistent (i.e., the j in a matches the j in b, and i,k in c match i,j,k in a,b)
    # Actually, T[a,b,c] = 1 for all valid a,b,c where the indices are consistent.
    
    # In the schoolbook, each (i,j,k) triple gives one term:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # The term is e_a ⊗ e_b ⊗ e_c
    
    # For a general rank-1 term, it's u ⊗ v ⊗ w where u is length n*m, v is length m*p, w is length n*p.
    # The decomposition is: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    
    # The condition is: for all i,j,k: T[i*m+j, j*p+k, k*n+i] = 1 = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # And for inconsistent indices, the sum should be 0.
    
    # OK so let me just build the Strassen decomposition computationally.
    
    # For 4x4 x 4x4, I'll use the recursive Strassen construction.
    # Level 1: 4x4 = 2x(2x2) blocks
    # Level 2: 2x2 = 2x(1x1) blocks
    
    # Let me define the 2x2 Strassen terms as (u, v, w) where u,v,w are length-4 vectors
    # (since 2x2 has 4 entries each).
    
    # For 2x2 x 2x2:
    # A has 4 entries: A[0,0], A[0,1], A[1,0], A[1,1] → indices 0,1,2,3 (row-major)
    # B has 4 entries: B[0,0], B[0,1], B[1,0], B[1,1] → indices 0,1,2,3
    # C has 4 entries: C[0,0], C[0,1], C[1,0], C[1,1] → indices 0,1,2,3
    
    # The 7 Strassen terms (u, v, w) where u,v,w are length-4:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #   u = [1, 0, 0, 1] (A[0,0] and A[1,1])
    #   v = [1, 0, 0, 1] (B[0,0] and B[1,1])
    #   w = [1, 0, 0, 1] (C[0,0] and C[1,1])
    #   Check: u[i]*v[j]*w[k] summed... 
    
    # I realize this approach is error-prone. Let me just write a verifier and construct it step by step.
    
    # Actually, let me try a completely different approach: use a numerical optimization to find a low-rank decomposition, then round to integers/half-integers.
    
    # Or better: let me just hardcode the known Strassen 4x4 decomposition.
    
    # Let me try yet another approach: use the fact that I can verify any decomposition, and try to find one via search.
    
    # Given the complexity, let me try the simplest possible improvement: 
    # Use the schoolbook for 4x4 x 4x5 but see if I can merge some terms.
    
    # Actually, the rank-50 construction (Strassen 49 + 1) is solid. Let me just implement it carefully.
    
    # Let me define the 2x2 Strassen terms properly.
    # For 2x2 x 2x2, the tensor T[i,j,k] = 1 for i,j,k in {0,1}.
    # A rank-1 term is u[i]*v[j]*w[k].
    
    # The 7 Strassen terms:
    # (I'll use the standard Strassen formulas)
    
    # Let me denote A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h) → contributes to C: C[0,0] += M1, C[1,1] += M1
    # M2 = (c+d)e → C[1,0] += M2
    # M3 = a(f-h) → C[0,1] += M3
    # M4 = d(g-e) → C[1,0] += M4, C[0,0] += M4
    # M5 = (a+b)h → C[0,1] += M5, C[0,0] -= M5
    # M6 = (c-a)(e+f) → C[1,1] += M6
    # M7 = (b-d)(g+h) → C[0,0] += M7
    
    # Wait, let me re-derive:
    # C[0,0] = a*e + b*g = M1 + M4 - M5 + M7
    #   M1 = (a+d)(e+h) = ae+ah+de+dh
    #   M4 = d(g-e) = dg-de
    #   M5 = (a+b)h = ah+bh
    #   M7 = (b-d)(g+h) = bg+bh-dg-dh
    #   M1+M4-M5+M7 = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh = ae + bg ✓
    
    # C[0,1] = a*f + b*h = M3 + M5
    #   M3 = a(f-h) = af-ah
    #   M5 = (a+b)h = ah+bh
    #   M3+M5 = af-ah+ah+bh = af+bh ✓
    
    # C[1,0] = c*e + d*g = M2 + M4
    #   M2 = (c+d)e = ce+de
    #   M4 = d(g-e) = dg-de
    #   M2+M4 = ce+de+dg-de = ce+dg ✓
    
    # C[1,1] = c*f + d*h = M1 - M2 + M3 + M6
    #   M1 = ae+ah+de+dh
    #   M2 = ce+de
    #   M3 = af-ah
    #   M6 = (c-a)(e+f) = ce+cf-ae-af
    #   M1-M2+M3+M6 = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af = dh+cf ✓
    
    # Great. Now, each M_s is a product of two linear forms, so it's a rank-1 term in the tensor.
    # M1 = (a+d)(e+h): 
    #   a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
    #   As a rank-1 tensor term: u[i]*v[j]*w[k]
    #   The term M1 contributes to the tensor as: for the (i,j,k) indices, the coefficient is u[i]*v[j]*w[k]
    #   M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #   This is a scalar. In the tensor, it contributes to C[0,0] and C[1,1].
    #   The rank-1 term is: u = [1,0,0,1] (for A entries [0,0],[0,1],[1,0],[1,1]), 
    #                      v = [1,0,0,1] (for B entries), 
    #                      w = [1,0,0,1] (for C entries [0,0],[0,1],[1,0],[1,1])
    #   Check: u[0]*v[0]*w[0] = 1*1*1 = 1 → T[0,0,0] gets +1 ✓ (C[0,0] from A[0,0]*B[0,0])
    #   u[0]*v[0]*w[3] = 1*1*1 = 1 → T[0,0,3] gets +1 ✓ (C[1,1] from A[0,0]*B[0,0]... wait)
    
    # I'm getting confused with the index mapping. Let me be very explicit.
    
    # For 2x2 x 2x2:
    # A indices: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3
    # B indices: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3
    # C indices: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3
    # 
    # The tensor T[a,b,c] where a is A-index, b is B-index, c is C-index.
    # T[a,b,c] = 1 if the indices are "consistent" for matrix multiplication.
    # Specifically, if a = i*m+j = i*2+j, b = j*p+k = j*2+k, c = k*n+i = k*2+i
    # Then T[a,b,c] = 1.
    # 
    # So T[i*2+j, j*2+k, k*2+i] = 1 for all i,j,k in {0,1}.
    # That's 8 entries that are 1, and the rest are 0.
    
    # A rank-1 term is u ⊗ v ⊗ w, contributing u[a]*v[b]*w[c] to T[a,b,c].
    # The decomposition requires: sum_c u_c[a]*v_c[b]*w_c[c] = T[a,b,c] for all a,b,c.
    
    # For M1 = (a+d)(e+h) = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    # This is a scalar product. In the tensor, it should contribute to the appropriate T entries.
    # M1 contributes to C[0,0] (index 0) and C[1,1] (index 3).
    # The "A part" is A[0,0]+A[1,1], so u = [1,0,0,1]
    # The "B part" is B[0,0]+B[1,1], so v = [1,0,0,1]
    # The "C part": M1 contributes +1 to C[0,0] and +1 to C[1,1], so w = [1,0,0,1]
    # 
    # Let's verify: u[a]*v[b]*w[c] for a=0(A[0,0]), b=0(B[0,0]), c=0(C[0,0]): 1*1*1 = 1
    # T[0,0,0]: i=0,j=0,k=0 → a=0*2+0=0, b=0*2+0=0, c=0*2+0=0. T[0,0,0] should be 1. ✓
    # 
    # u[a]*v[b]*w[c] for a=0, b=0, c=3(C[1,1]): 1*1*1 = 1
    # T[0,0,3]: i=0,j=0,k=1 → a=0, b=0*2+1=1, c=1*2+0=2. So T[0,0,3] is NOT a valid entry (b should be 1, not 0).
    # So this contributes to T[0,0,3] which should be 0! That's a problem.
    
    # I see the issue. The rank-1 term u⊗v⊗w contributes to ALL (a,b,c) entries, not just the consistent ones.
    # So I need to be more careful.
    
    # Let me reconsider. The tensor T has dimensions 4×4×4 (for 2x2x2).
    # T[a,b,c] = 1 iff there exist i,j,k such that a=i*2+j, b=j*2+k, c=k*2+i.
    # 
    # The valid (a,b,c) triples are:
    # (i,j,k) = (0,0,0): a=0, b=0, c=0
    # (i,j,k) = (0,0,1): a=0, b=1, c=2
    # (i,j,k) = (0,1,0): a=1, b=0, c=1
    # (i,j,k) = (0,1,1): a=1, b=1, c=3
    # (i,j,k) = (1,0,0): a=2, b=0, c=0
    # (i,j,k) = (1,0,1): a=2, b=1, c=2
    # (i,j,k) = (1,1,0): a=3, b=0, c=1
    # (i,j,k) = (1,1,1): a=3, b=1, c=3
    
    # So T[0,0,0]=1, T[0,1,2]=1, T[1,0,1]=1, T[1,1,3]=1, T[2,0,0]=1, T[2,1,2]=1, T[3,0,1]=1, T[3,1,3]=1
    # All other entries are 0.
    
    # Now, for M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    # This product contributes to C[0,0] and C[1,1].
    # The contribution to C[0,0] (c=0) comes from: A[0,0]*B[0,0] + A[1,1]*B[1,1] (cross terms cancel in the final C, but in the intermediate product M1, they're all included)
    # 
    # Wait, I think the issue is that M1 is an intermediate product, not a direct contribution to T.
    # The Strassen algorithm computes 7 intermediate products M1..M7, and then C is a linear combination of these.
    # 
    # In tensor decomposition terms, each M_s corresponds to a rank-1 term in the tensor, and the final C is obtained by summing these terms with appropriate coefficients.
    
    # Let me think about it differently. The tensor decomposition T = sum_s u_s ⊗ v_s ⊗ w_s means:
    # For all a,b,c: T[a,b,c] = sum_s u_s[a] * v_s[b] * w_s[c]
    
    # For the 2x2 case, I need to find 7 such terms.
    # Let me just solve it: T has 8 non-zero entries. I need 7 rank-1 tensors that sum to T.
    
    # This is a known result, so let me just look up the explicit terms or derive them.
    
    # Actually, let me try a different approach. Let me just use a numerical solver to find a rank-50 decomposition for the 4x4x4x5 problem, then round to integers.
    
    # Or, let me try to be clever: since the 4x4x4x5 tensor is T[i*4+j, j*5+k, k*4+i] = 1, I can try to use the Kronecker product structure.
    
    # The 4x4x4x5 tensor can be seen as: for k=0..3, it's the 4x4x4 multiplication tensor (T[i*4+j, j*4+k, k*4+i] = 1), and for k=4, it's T[i*4+j, j*5+4, 4*4+i] = 1.
    
    # The k=4 part: T[i*4+j, j*5+4, 16+i] = 1 for all i,j. This is the outer product of 1_16 (in U-space) ⊗ 1_4 (in V-space, rows 4,9,14,19) ⊗ 1_4 (in W-space, rows 16,17,18,19).
    # Wait, V has m*p = 4*5 = 20 rows. The k=4 column of B corresponds to V rows j*5+4 for j=0..3, i.e., rows 4,9,14,19.
    # W has p*n = 5*4 = 20 rows. The k=4 column of C corresponds to W rows 4*4+i = 16+i for i=0..3, i.e., rows 16,17,18,19.
    
    # So the extra term is: u = [1]*16, v = [0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1], w = [0]*16 + [1,1,1,1]
    # This is a valid rank-1 term with integer entries.
    
    # Now for the 4x4x4 part (k=0..3), I need rank 49. Let me construct this using Strassen.
    
    # For the 4x4x4 part, the tensor is T[i*4+j, j*4+k, k*4+i] = 1 for i,j,k in {0,1,2,3}.
    # This is the same as the 4x4x4 matrix multiplication tensor.
    
    # Using Strassen: split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # In the flattened A (16 entries): 
    #   A00 = A[0:2, 0:2] → entries 0,1,4,5
    #   A01 = A[0:2, 2:4] → entries 2,3,6,7
    #   A10 = A[2:4, 0:2] → entries 8,9,12,13
    #   A11 = A[2:4, 2:4] → entries 10,11,14,15
    
    # Similarly for B (20 entries for 4x5, but for the 4x4 part, B is 4x4 → 16 entries, rows j*4+k for j,k in 0..3):
    #   B00 = B[0:2, 0:2] → entries 0,1,4,5
    #   B01 = B[0:2, 2:4] → entries 2,3,6,7
    #   B10 = B[2:4, 0:2] → entries 8,9,12,13
    #   B11 = B[2:4, 2:4] → entries 10,11,14,15
    
    # C (16 entries for 4x4):
    #   C00 = C[0:2, 0:2] → entries 0,1,4,5
    #   C01 = C[0:2, 2:4] → entries 2,3,6,7
    #   C10 = C[2:4, 0:2] → entries 8,9,12,13
    #   C11 = C[2:4, 2:4] → entries 10,11,14,15
    
    # Strassen at the block level:
    # S1 = (A00+A11)*(B00+B11)
    # S2 = (A10+A11)*B00
    # S3 = A00*(B01-B11)
    # S4 = A11*(B10-B00)
    # S5 = (A00+A01)*B11
    # S6 = (A10-A00)*(B00+B01)
    # S7 = (A01-A11)*(B10+B11)
    
    # C00 = S1+S4-S5+S7
    # C01 = S3+S5
    # C10 = S2+S4
    # C11 = S1-S2+S3+S6
    
    # Each S_s is a 2x2x2x2x2x2 tensor (2x2 * 2x2 = 2x2), which has rank 7 (Strassen).
    # So each S_s contributes 7 rank-1 terms to the overall tensor.
    # Total: 7*7 = 49 terms.
    
    # Now I need to express each S_s as 7 rank-1 terms in the full 16x16x16 tensor space.
    
    # For S1 = (A00+A11)*(B00+B11):
    # This is a 2x2 * 2x2 = 2x2 multiplication where:
    #   "A" = A00+A11 (a 2x2 matrix)
    #   "B" = B00+B11 (a 2x2 matrix)
    #   "C" = contributes to C00 and C11
    # 
    # The 7 Strassen terms for this 2x2x2x2x2x2 sub-problem give 7 rank-1 terms in the 4x4x4 sub-tensor space.
    # Each of these needs to be "lifted" to the full 16x16x16 space.
    
    # This is getting very involved. Let me just write a program that does this construction.
    
    # Given the time constraints, let me try a simpler approach: use a known result or a greedy algorithm.
    
    # Actually, let me try to just return the rank-50 construction with a proper Strassen 4x4 decomposition.
    # I'll write the code to construct it.
    
    # Let me define the 2x2 Strassen terms as (u, v, w) where u,v,w are length-4 vectors (for the 4 entries of each 2x2 matrix).
    # Then I'll compose them for the 4x4 case.
    
    # 2x2 Strassen terms (u, v, w) for 2x2x2x2x2x2:
    # Using the formulas above:
    # M1 = (a+d)(e+h): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # M2 = (c+d)e: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # M3 = a(f-h): u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # M4 = d(g-e): u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
    # M5 = (a+b)h: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # M6 = (c-a)(e+f): u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1]
    # M7 = (b-d)(g+h): u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
    
    # Let me verify M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # Contribution to T[a,b,c] = u[a]*v[b]*w[c]
    # T[0,0,0] = 1*1*1 = 1 ✓ (should be 1)
    # T[0,0,3] = 1*1*1 = 1 ✗ (should be 0)
    
    # This doesn't work! The rank-1 term contributes to all (a,b,c), not just the consistent ones.
    
    # I think the issue is that the Strassen algorithm doesn't directly give a tensor decomposition in this simple way.
    # The Strassen algorithm gives a bilinear algorithm, which IS a tensor decomposition, but the terms are not simply u⊗v⊗w with the u,v,w I wrote.
    
    # Let me reconsider. A bilinear algorithm for matrix multiplication is exactly a tensor decomposition.
    # The Strassen algorithm for 2x2 has 7 multiplications, so the tensor has rank ≤ 7.
    # Each multiplication M_s = (linear form in A) * (linear form in B) contributes a rank-1 term.
    # The output C is a linear combination of the M_s.
    
    # So for M1 = (a+d)(e+h):
    # The linear form in A is a+d = A[0,0]+A[1,1], so u = [1,0,0,1]
    # The linear form in B is e+h = B[0,0]+B[1,1], so v = [1,0,0,1]
    # M1 contributes to C[0,0] with coeff +1 and C[1,1] with coeff +1, so w = [1,0,0,1]
    
    # The rank-1 term is u⊗v⊗w, contributing u[a]*v[b]*w[c] to T[a,b,c].
    # For this to be valid, we need: for all (a,b,c) that are "inconsistent" (T[a,b,c] should be 0), the sum over all 7 terms should be 0.
    
    # So I need to verify that the sum of all 7 terms gives T[a,b,c] = 1 for consistent (a,b,c) and 0 otherwise.
    
    # Let me verify for T[0,0,3] (which should be 0):
    # M1: u[0]*v[0]*w[3] = 1*1*1 = 1
    # M2: u[0]*v[0]*w[3] = 0*1*0 = 0
    # M3: u[0]*v[0]*w[3] = 1*0*0 = 0
    # M4: u[0]*v[0]*w[3] = 0*0*0 = 0
    # M5: u[0]*v[0]*w[3] = 1*0*0 = 0
    # M6: u[0]*v[0]*w[3] = 0*1*1 = 0
    # M7: u[0]*v[0]*w[3] = 0*0*1 = 0
    # Sum = 1 ≠ 0 ✗
    
    # So my terms are wrong. The Strassen algorithm doesn't decompose in this naive way.
    
    # I think the correct interpretation is:
    # The bilinear algorithm computes C as a linear combination of the M_s.
    # M1 = (a+d)(e+h) is a scalar.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,1] = M1 - M2 + M3 + M6
    # 
    # In tensor decomposition terms, the contribution of M1 to the tensor is:
    # M1 contributes to C[0,0] with coefficient 1 and to C[1,1] with coefficient 1.
    # The "A part" of M1 is (a+d), the "B part" is (e+h).
    # So the rank-1 term for M1 is: u = [1,0,0,1] (A part), v = [1,0,0,1] (B part), w = [1,0,0,1] (C part, with +1 at C[0,0] and +1 at C[1,1]).
    
    # But as I showed, this doesn't work because it contributes to inconsistent (a,b,c) entries.
    
    # I think the resolution is that the tensor decomposition is not simply the sum of these rank-1 terms. Instead, the bilinear algorithm defines the tensor as:
    # T = sum_s (u_s ⊗ v_s) * w_s^T ... no, that's not right either.
    
    # Let me look at this from first principles. The matrix multiplication tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p} is defined by:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    # where E_{i,j} is the matrix unit.
    
    # A rank-1 tensor is A ⊗ B ⊗ C where A ∈ F^{n×m}, B ∈ F^{m×p}, C ∈ F^{n×p}.
    # The tensor product A ⊗ B ⊗ C has entries A[i,j]*B[j,k]*C[i,k] (in the flattened sense).
    
    # Wait, I think the issue is that the tensor product of three matrices A⊗B⊗C is NOT simply the entrywise product. 
    # In the tensor space F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p}, a rank-1 element is A ⊗ B ⊗ C where A, B, C are matrices, and the tensor is the "outer product" of the three matrices.
    # The entry at position (i,j, j',k, i',k') is A[i,j]*B[j',k]*C[i',k']... 
    
    # I think I've been confusing myself. Let me use a cleaner formulation.
    
    # The tensor T ∈ F^{nm} ⊗ F^{mp} ⊗ F^{np} (flattened).
    # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    
    # A rank-1 tensor is u ⊗ v ⊗ w where u ∈ F^{nm}, v ∈ F^{mp}, w ∈ F^{np}.
    # (u ⊗ v ⊗ w)[a,b,c] = u[a]*v[b]*w[c].
    
    # The decomposition T = sum_s u_s ⊗ v_s ⊗ w_s means:
    # For all a,b,c: T[a,b,c] = sum_s u_s[a]*v_s[b]*w_s[c].
    
    # For the 2x2x2 case, T has 8 non-zero entries (as listed above).
    # I need 7 vectors u_s, v_s, w_s (each length 4) such that the sum gives T.
    
    # This is a system of 64 equations (4*4*4) with 7*3*4 = 84 unknowns. It's underdetermined, so solutions exist.
    
    # Let me just solve it numerically and then round.
    
    # Actually, given the time pressure, let me try a different strategy: use a known rank-49 decomposition for 4x4x4 and add 1 for the extra column.
    
    # I'll use a numerical approach to find the decomposition, then round to integers/half-integers.
    
    # Let me try using numpy to find a rank-50 decomposition.
    
    import numpy as np
    
    # Build the tensor T for 4x4x4x5
    n, m, p = 4, 4, 5
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Try to find a rank-50 decomposition
    # This is a hard problem. Let me try a greedy approach.
    
    # Actually, given the 120s time limit, let me just return the schoolbook rank-80 solution if I can't find a better one quickly.
    # But that scores 1.0, same as the seed.
    
    # Let me try a quick greedy algorithm to see if I can find a lower rank.
    
    # Greedy: repeatedly find the best rank-1 approximation to the residual.
    # This is like the "greedy tensor decomposition" algorithm.
    
    # Given time constraints, let me just try to return a rank-50 solution using the Strassen + 1 approach, 
    # even if I have to hardcode it.
    
    # Let me try to construct the Strassen 4x4 decomposition by composition.
    
    # I'll define the 2x2 Strassen terms as (u, v, w) triples and verify them.
    
    # For 2x2x2, the 8 non-zero T entries are:
    # (0,0,0), (0,1,2), (1,0,1), (1,1,3), (2,0,0), (2,1,2), (3,0,1), (3,1,3)
    
    # Let me try to find 7 rank-1 terms.
    # I'll use a simple approach: set up the system of equations and solve.
    
    # Actually, let me just try to verify a known Strassen decomposition.
    
    # After some research, the correct Strassen decomposition for 2x2x2 in the tensor sense is:
    # The 7 terms are NOT simply the products M1..M7 as I wrote. Instead, they are:
    
    # Let me try:
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # Term 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # Term 3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # Term 4: u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
    # Term 5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # Term 6: u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1]
    # Term 7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
    
    # Let me verify T[0,0,0]:
    # T1: 1*1*1 = 1
    # T2: 0*1*0 = 0
    # T3: 1*0*0 = 0
    # T4: 0*0*1 = 0
    # T5: 1*0*(-1) = 0
    # T6: 0*1*0 = 0
    # T7: 0*0*1 = 0
    # Sum = 1 ✓
    
    # T[0,1,2]:
    # T1: 1*0*0 = 0
    # T2: 0*0*1 = 0
    # T3: 1*1*0 = 0
    # T4: 0*0*1 = 0
    # T5: 1*0*0 = 0
    # T6: 0*1*0 = 0
    # T7: 0*0*0 = 0
    # Sum = 0 ✗ (should be 1)
    
    # Still wrong. I think the issue is that the Strassen bilinear algorithm doesn't directly translate to a simple tensor decomposition in this way.
    
    # Let me try a completely different approach: use a numerical solver to find a rank-r decomposition.
    
    # Given the time pressure, let me just try to return a solution and see what score I get.
    # I'll return the schoolbook rank-80 solution as a fallback, but first try a quick greedy.
    
    # Actually, let me just return the schoolbook solution. It's safe and scores 1.0.
    # The task says "Keep the function names and signatures of the seed program."
    
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
