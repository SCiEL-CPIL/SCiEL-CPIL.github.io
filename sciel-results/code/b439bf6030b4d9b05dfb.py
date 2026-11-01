"""Matrix multiplication 2,2,7 rank-minimal decomposition using Strassen structure."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    # U: shape (n*m, rank) = (4, rank)
    # V: shape (m*p, rank) = (14, rank)
    # W: shape (p*n, rank) = (14, rank)
    # Convention: C[k*n+i, r] = sum over U[i*m+j,r]*V[j*p+k,r]
    # i.e., C[i,k] = sum_r U[i*m+j] V[j*p+k] W[k*n+i]
    # We want C = A*B, C[i,k] = sum_j A[i,j] B[j,k]

    rank = 25
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # Strassen products (each is a bilinear form A_lin * B_lin)
    # M1 = (A00+A11) * B0  -> for each k: (A[0,0]+A[1,1]) * B[0,k]
    # M2 = (A10+A11) * B0  -> (A[1,0]+A[1,1]) * B[0,k]
    # M3 = A00 * (B1 - B0) -> A[0,0] * (B[1,k]-B[0,k])
    # M4 = A11 * (B0 - B1) -> A[1,1] * (B[0,k]-B[1,k])
    # M5 = (A00+A01) * B1  -> (A[0,0]+A[0,1]) * B[1,k]
    # M6 = (A10-A00) * B0  -> (A[1,0]-A[0,0]) * B[0,k]
    # M7 = (A01-A11) * B1  -> (A[0,1]-A[1,1]) * B[1,k]

    # Reconstruct C:
    # C[0,k] = M1 + M4 - M5 + M7
    # C[1,k] = M2 + M4 + M3 - M6  (need to verify)

    # Let me define the 7 bilinear forms and their contributions.
    # For each form t, define:
    #   a_coeffs[i] for A[i,j] (i in 0,1; j in 0,1)
    #   b_coeffs[j] for B[j,k] (j in 0,1)
    #   Then contribution to C[i,k] = W[i,k] * (sum a_coeffs[i,j] A[i,j]) * (sum b_coeffs[j] B[j,k])
    # Actually the form is: (linear in A) * (linear in B), and this contributes W_coeff to C[i,k].

    # Let me set up: each of the 7 forms is a scalar bilinear term.
    # Form t: L_A_t(A) * L_B_t(B) where L_A_t is linear in A entries, L_B_t is linear in B entries.
    # This contributes to each C[i,k] with coefficient w_t[i,k].

    # But the tensor decomposition requires: C[i,k] = sum_r U[i,r] V[k,r] W[r]  (with proper indexing)
    # Here U[i*m+j] is for A[i,j], V[j*p+k] is for B[j,k], W[k*n+i] is for C[i,k].
    # Term r contributes: sum_{i,j,k} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] * A[i,j] B[j,k] = C[i,k]
    # So for each r, the contribution to C[i,k] is:
    #   (sum_j U[i*m+j, r] * something) ... 

    # Actually the standard form: the tensor T has T[i,j,k] = delta (C[i,k] = sum_j A[i,j]B[j,k])
    # T[i,j,k] = 1 if k is the output column... no.
    # C[i,k] = sum_j A[i,j] B[j,k]. So the multiplication tensor is T[i,j,k] where T[i,j,k]=1 
    # meaning A[i,j]*B[j,k] contributes to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k with the 
    # constraint that the middle index matches: A[i,j] and B[j,k] -> C[i,k].
    # So T[i, j, k] = 1 (the j in A[i,j] must equal the j in B[j,k], and output is C[i,k]).
    # In the decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # (using i,j for A, j,k for B, i,k for C).

    # So U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].

    # For a Strassen form: L_A(A) = sum_{i,j} a_{i,j} A[i,j], L_B(B) = sum_{j,k} b_{j,k} B[j,k].
    # But L_B is the same for all k (the column), so b_{j,k} = b_j (independent of k).
    # Then the form contributes a_{i,j} * b_j to T[i,j,k] for each k.
    # Wait, that's not right either. Let me think again.

    # The bilinear form is: (sum_{i,j} a_{i,j} A[i,j]) * (sum_j b_j B[j,k]) for a fixed k.
    # This equals sum_j (sum_i a_{i,j} A[i,j]) b_j B[j,k].
    # So it contributes to C[i,k] the term: (sum_j a_{i,j} b_j) ... no.
    # The product (sum_{i,j} a_{i,j} A[i,j]) * (sum_j b_j B[j,k]) = sum_{i,j,j'} a_{i,j} b_{j'} A[i,j] B[j',k].
    # This contributes to C[i,k] the coefficient sum_j a_{i,j} b_j (when j'=j).
    # So the contribution to T[i,j,k] from this form is a_{i,j} * b_j.

    # Great! So for each Strassen form t with coefficients a_t[i,j] and b_t[j]:
    #   U[i*m+j, r_t] = a_t[i,j]
    #   V[j*p+k, r_t] = b_t[j]  (independent of k)
    #   W[k*n+i, r_t] = contribution coefficient to C[i,k]

    # But W depends on i,k. The contribution to C[i,k] is sum_j a_t[i,j] b_t[j].
    # So W[k*n+i, r_t] = sum_j a_t[i,j] b_t[j].

    # Now let me define the 7 Strassen forms and compute W.

    # Form 1: L_A = A00+A11, L_B = B0 (b = [1, 0])
    #   a1 = [[1, 0], [0, 1]]  (A[0,0] + A[1,1])
    #   b1 = [1, 0]
    #   Contribution to C[i,k]: sum_j a1[i,j]*b1[j]
    #     C[0,k]: a1[0,0]*b1[0] + a1[0,1]*b1[1] = 1*1 + 0*0 = 1
    #     C[1,k]: a1[1,0]*b1[0] + a1[1,1]*b1[1] = 0*1 + 1*0 = 0
    #   So W for form1: C[0,k] += 1, C[1,k] += 0

    # Form 2: L_A = A10+A11, L_B = B0
    #   a2 = [[0,0],[1,1]], b2 = [1,0]
    #   C[0,k]: 0, C[1,k]: 1*1+1*0 = 1

    # Form 3: L_A = A00, L_B = B1-B0
    #   a3 = [[1,0],[0,0]], b3 = [-1, 1]
    #   C[0,k]: 1*(-1)+0*1 = -1, C[1,k]: 0

    # Form 4: L_A = A11, L_B = B0-B1
    #   a4 = [[0,0],[0,1]], b4 = [1,-1]
    #   C[0,k]: 0, C[1,k]: 0*1+1*(-1) = -1

    # Form 5: L_A = A00+A01, L_B = B1
    #   a5 = [[1,1],[0,0]], b5 = [0,1]
    #   C[0,k]: 1*0+1*1 = 1, C[1,k]: 0

    # Form 6: L_A = A10-A00, L_B = B0
    #   a6 = [[-1,0],[1,0]], b6 = [1,0]
    #   C[0,k]: -1*1+0 = -1, C[1,k]: 1*1+0 = 1

    # Form 7: L_A = A01-A11, L_B = B1
    #   a7 = [[0,1],[0,-1]], b7 = [0,1]
    #   C[0,k]: 0*0+1*1 = 1, C[1,k]: 0*0+(-1)*1 = -1

    # Now reconstruct C:
    # C[0,k] = w1*C0 + w2*C0 + w3*C0 + w4*C0 + w5*C0 + w6*C0 + w7*C0
    # where each wt*C0 is the contribution.
    # C[0,k] = 1*F1 + 0*F2 + (-1)*F3 + 0*F4 + 1*F5 + (-1)*F6 + 1*F7
    # C[1,k] = 0*F1 + 1*F2 + 0*F3 + (-1)*F4 + 0*F5 + 1*F6 + (-1)*F7

    # Let me verify C[0,k] should equal A00*B0 + A01*B1.
    # F1 = (A00+A11)*B0
    # F3 = A00*(B1-B0)
    # F5 = (A00+A01)*B1
    # F6 = (A10-A00)*B0
    # F7 = (A01-A11)*B1
    # C[0,k] = F1 - F3 + F5 - F6 + F7
    #   = (A00+A11)B0 - A00(B1-B0) + (A00+A01)B1 - (A10-A00)B0 + (A01-A11)B1
    #   = A00 B0 + A11 B0 - A00 B1 + A00 B0 + A00 B1 + A01 B1 - A10 B0 + A00 B0 + A01 B1 - A11 B1
    #   = (A00 B0 + A00 B0 + A00 B0) + A11 B0 - A10 B0 - A11 B1 + (A01 B1 + A01 B1) + (-A00 B1 + A00 B1)
    #   = 3 A00 B0 + A11 B0 - A10 B0 - A11 B1 + 2 A01 B1
    # That's not A00 B0 + A01 B1. So my Strassen formulas are wrong.

    # Let me look up the correct Strassen decomposition and just hardcode it properly.
    # Standard Strassen for 2x2 * 2x2:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11) b00
    # p3 = a00 (b10 - b11)
    # p4 = a11 (b01 - b00)
    # p5 = (a00 + a01) b11
    # p6 = (a10 - a00)(b00 + b10)
    # p7 = (a01 - a11)(b01 + b11)
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p1
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p5

    # For our case B is 2x7, so b00, b10, etc. are vectors.
    # Let me denote B[0,:] = b0 (vector), B[1,:] = b1 (vector).
    # p1 = (a00+a11)(b0+b1)
    # p2 = (a10+a11) b0
    # p3 = a00 (b1-b0)  -- wait, b10-b11 means B[1,0]-B[1,1], but B has only 2 rows.
    # I think the standard Strassen assumes B is 2x2 with entries b00,b01,b10,b11.
    # For B 2x7, I'll treat each column separately, but the linear forms in B must be the same.

    # Actually, the issue is that in standard Strassen, p3 = a00(b10-b11) uses B[1,0] and B[1,1],
    # which are different columns. For B 2x7, B[1,0] and B[1,1] are elements of row 1.
    # So p3 = a00 * (B[1,0] - B[1,1]) is a scalar, but we need p3 to be a vector for each column k.

    # The correct approach: for each column k, the Strassen forms use B[0,k] and B[1,k].
    # So p3_k = a00 * (B[1,k] - B[0,k])? No, that's not standard Strassen.

    # I think I'm overcomplicating. Let me just use a different approach:
    # The rank of 2x2x7 multiplication. We know 2x2x2 has rank 7 (Strassen).
    # For 2x2x7, we can think of it as 7 copies of 2x2x1, but that's not helpful.

    # Actually, the multiplication 2x2 * 2x7: C[i,k] = sum_j A[i,j] B[j,k].
    # This is a tensor of order 3 with dimensions (2,2,7) for the "A index, B index, C index".
    # The rank is the minimum number of rank-1 tensors to sum to this.

    # Known: rank of 2x2x2 is 7. For 2x2x7, we can use the structure.
    # Let me just try to find a rank-25 decomposition by hand or use a known result.

    # Actually, let me just return the schoolbook (rank 28) as a safe fallback,
    # but first try to implement Strassen properly.

    # Let me redefine using the correct Strassen for the rectangular case.
    # The key: we have A (2x2) and B (2x7). C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    # We want to write this as sum of 7 bilinear forms.

    # Let me use the following 7 forms (each is (linear in A) * (linear in B column)):
    # For column k, define:
    # x0 = A[0,0], x1 = A[0,1], x2 = A[1,0], x3 = A[1,1]
    # y0 = B[0,k], y1 = B[1,k]
    # C[0,k] = x0*y0 + x1*y1
    # C[1,k] = x2*y0 + x3*y1

    # Strassen forms (7 of them):
    # f1 = (x0+x3)(y0+y1)
    # f2 = (x2+x3) y0
    # f3 = x0 (y1-y0)
    # f4 = x3 (y0-y1)
    # f5 = (x0+x1) y1
    # f6 = (x2-x0) y0
    # f7 = (x1-x3) y1

    # C[0,k] = f1 + f4 - f5 + f7
    # C[1,k] = f2 + f4 + f3 - f6

    # Verify C[0,k]:
    # f1 = (x0+x3)(y0+y1) = x0y0 + x0y1 + x3y0 + x3y1
    # f4 = x3(y0-y1) = x3y0 - x3y1
    # f5 = (x0+x1)y1 = x0y1 + x1y1
    # f7 = (x1-x3)y1 = x1y1 - x3y1
    # C[0,k] = f1 + f4 - f5 + f7
    #   = (x0y0 + x0y1 + x3y0 + x3y1) + (x3y0 - x3y1) - (x0y1 + x1y1) + (x1y1 - x3y1)
    #   = x0y0 + x0y1 + x3y0 + x3y1 + x3y0 - x3y1 - x0y1 - x1y1 + x1y1 - x3y1
    #   = x0y0 + (x0y1 - x0y1) + (x3y0 + x3y0) + (x3y1 - x3y1 - x3y1) + (-x1y1 + x1y1)
    #   = x0y0 + 2x3y0 - x3y1
    # That's not x0y0 + x1y1. Still wrong.

    # I clearly have the Strassen formulas wrong. Let me just look at the actual correct ones.
    # The correct Strassen:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)(b00)
    # p3 = (a00)(b10 - b11)
    # p4 = (a11)(b01 - b00)
    # p5 = (a00 + a01)(b11)
    # p6 = (a10 - a00)(b00 + b10)
    # p7 = (a01 - a11)(b01 + b11)
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p1
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p5

    # For B 2x7, b00 = B[0,0], b01 = B[0,1], b10 = B[1,0], b11 = B[1,1].
    # But B has 7 columns, so b00, b01, ..., b06 and b10, b11, ..., b16.
    # The Strassen forms p1..p7 are scalars for a 2x2 B. For 2x7, we need vector forms.

    # I think the right way: for each column k, we can't use Strassen directly because 
    # Strassen mixes different columns (b00, b11 are different columns).

    # Alternative: treat the 2x2x7 tensor differently.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has T[i,j,k] = 1 (meaning A[i,j]*B[j,k] -> C[i,k]).
    # We want to decompose T as sum of rank-1 tensors.

    # A rank-1 tensor is u[i] * v[j] * w[k].
    # So T[i,j,k] = sum_r u_r[i] v_r[j] w_r[k].
    # In our notation: U[i*m+j, r] corresponds to u_r[i]*v_r[j] but combined.
    # Actually U[i*m+j, r] is a single entry, not a product. Let me re-read the contract.

    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] is the multiplication tensor (T[i,j,k] = 1 if the product A[i,j]*B[j,k] 
    # contributes to C[i,k], which is always true for valid i,j,k).

    # So T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,7).
    # Wait, that's just the all-ones tensor? No.
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor T satisfies:
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k (with the constraint that the middle index j matches).
    # But i and k are free, j is the summation index. So T[i,j,k] = 1 for all i,j,k.
    # That means the tensor is just the all-ones tensor of shape (2,2,7)?
    # No wait, that can't be right because then the rank would be 1.

    # I think I'm misunderstanding the tensor. Let me re-read: "exactly decompose the matrix 
    # multiplication tensor". The multiplication tensor for (n,m,p) is:
    # T in R^{n*m} x R^{m*p} x R^{p*n} where T[(i,j), (j,k), (k,i)] = 1.
    # So T is indexed by (i,j) for A, (j,k) for B, (k,i) for C.
    # T[(i,j),(j,k),(k,i)] = 1, and 0 otherwise.
    # In our case n=2, m=2, p=7:
    # T[(i,j),(j,k),(k,i)] = 1 where i in [0,2), j in [0,2), k in [0,7).
    # So for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1.

    # The decomposition: T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    # So U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal 1 
    # when (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k, and 0 otherwise.

    # This is a standard matrix multiplication tensor decomposition.
    # The schoolbook decomposition has rank n*m*p = 28, with one term per (i,j,k).
    # Strassen reduces 2x2x2 from 8 to 7.

    # For 2x2x7, I'll just use the schoolbook (rank 28) as a safe answer,
    # since I'm having trouble getting Strassen right for the rectangular case.
    # But let me try one more time with a cleaner approach.

    # Actually, I realize I can just use the schoolbook. The task says "rank-minimal" 
    # but the score is the rank (lower is better), and the seed is 28. If I can't find 
    # something better quickly, I should return 28 to at least match the seed.

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
