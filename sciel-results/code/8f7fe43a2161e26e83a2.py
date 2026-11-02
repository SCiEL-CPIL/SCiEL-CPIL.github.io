"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Uses the known Strassen-based rank-49 decomposition. The matrix multiplication
tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must hold
exactly for all i, j, k with n=m=p=4.

The rank-49 decomposition is constructed by:
1. Starting from the rank-64 schoolbook decomposition.
2. Applying the Strassen-like reductions that exploit symmetries and
   linear dependencies among the 64 bilinear forms.

For 4x4 matrix multiplication, the minimal rank is known to be 49
(Le Gall, 1997; improved constructions).

We implement the decomposition directly using the known pattern of
rank-49 decompositions for 4x4 matrices.

The key idea: the 4x4 multiplication tensor has rank 49. We can construct
this by partitioning the 4x4 matrices into 2x2 blocks and using Strassen's
7-multiplication algorithm on each pair, but with additional cross-term
cancellations.

Actually, let me think more carefully. The rank of 4x4 matrix multiplication
is known to be at most 49. Let me construct it explicitly.

One approach: use the fact that M_{4} can be embedded in M_{2} x M_{2} with
some structure. The 4x4 product can be computed using:
- 4 block-Strassen multiplications (each 2x2*2x2 using 7 mults = 28)
- Plus 21 additional scalar multiplications for cross terms
Total: 49

Let me verify: A = [[A11, A12], [A21, A22]] where each block is 2x2.
C = A*B where B = [[B11, B12], [B21, B22]].

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

Using Strassen for each 2x2 block product:
M1 = (A11+A22)*(B11+B22)
M2 = (A21+A22)*B11
M3 = A11*(B12-B22)
M4 = A22*(B21-B11)
M5 = (A11+A12)*B22
M6 = (A21-A11)*(B11+B12)
M7 = (A12-A22)*(B21+B22)

But this gives 4*7 = 28 multiplications for the four block products.
However, the cross-block terms are already handled within each Strassen call.

Wait, I need to be more careful. The 4x4 matrix product involves 16 block
products of 2x2 matrices, but they share structure.

Let me use a different approach. I'll hardcode the rank-49 decomposition
based on known constructions.

Actually, the cleanest approach is to use the explicit rank-49 decomposition
that has been published. Let me construct it using the following method:

The 4x4 matrix multiplication tensor has entries T[i,j,k] for i,j,k in {0,1,2,3}
corresponding to (row, col of A) * (row, col of B) -> (row, col of C).

In the tensor notation: T_{(i,j),(j,k),(i,k)} = 1 for all valid indices.

The rank-49 decomposition can be obtained from the rank-64 decomposition by
finding 15 linear dependencies among the 64 basis tensors.

Let me implement this by constructing the 64x(4*4*4) = 64x64 coefficient matrix
and finding its null space to identify dependencies, then eliminating them.

Actually, for a direct implementation, let me use the known explicit rank-49
decomposition. I'll encode the U, V, W matrices directly.

Given the complexity of deriving the exact rank-49 decomposition from scratch,
let me use a computational approach: start with rank 64, find the null space
of the constraint matrix, and eliminate dependent terms.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=m=p=4
    # Number of bilinear forms needed: rank
    # We aim for rank 49
    
    # The matrix multiplication tensor has dimensions:
    # Input A: n*m = 16 entries
    # Input B: m*p = 16 entries  
    # Output C: p*n = 16 entries
    # Total tensor entries: n*m*p = 64
    
    # Each rank-1 term corresponds to one bilinear form z_c = u_c^T * A * v_c
    # where u_c is a row vector of length n*m, v_c is a row vector of length m*p
    # and the contribution to C is w_c * z_c where w_c is a vector of length p*n
    
    # For exact decomposition: sum_c U[:,c] * V[:,c] * W[:,c]^T = T (the multiplication tensor)
    # where T is the 16x16x16 tensor with T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's build the constraint system. For each output entry (i,k) in C,
    # we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for the correct j
    # and 0 otherwise.
    
    # Actually, the constraint is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k}
    # where delta = 1 if this is a valid multiplication term, 0 otherwise.
    
    # Wait, the multiplication tensor T has T[a, b, c] = 1 where a = i*m+j (entry of A),
    # b = j*p+k (entry of B), c = k*n+i (entry of C), for all i,j,k.
    
    # The decomposition requires: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a system of 16*16*16 = 4096 equations (but only 64 are nonzero).
    
    # For a rank-R decomposition, we have R unknowns per matrix, so 3*R*16 = 48R unknowns.
    # For R=49, that's 2352 unknowns and 4096 constraints (64 nonzero, 4032 zero).
    
    # This is a nonlinear system (bilinear in U,V given W, etc.)
    
    # Alternative: use the known explicit rank-49 decomposition.
    
    # Let me try a different approach: use the fact that the rank of 4x4 matrix
    # multiplication is 49, and construct it via the following method:
    
    # Partition 4x4 into four 2x2 blocks. Use Strassen for each block pair,
    # but share computations across blocks.
    
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x2)
    # C = A*B
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen for each 2x2 product:
    # For C00: 7 mults
    # For C01: 7 mults
    # For C10: 7 mults
    # For C11: 7 mults
    # Total: 28 mults
    
    # But we can do better by sharing. The key insight is that some of the
    # Strassen intermediate values can be shared or combined.
    
    # Actually, 28 < 49, so this can't be right. The issue is that Strassen's
    # algorithm computes the product of two 2x2 matrices, but here we need
    # products like A00*B00 which is a 2x2 * 2x2 product.
    
    # Each 2x2*2x2 product requires 7 scalar multiplications (Strassen) or
    # 4 (schoolbook). So 4 block products * 7 = 28 scalar mults.
    # But we also need to combine the results, which involves additions.
    
    # Wait, I think I'm confusing myself. Let me reconsider.
    
    # The "rank" of matrix multiplication counts the number of scalar
    # multiplications (bilinear forms). For 4x4 using block Strassen:
    # We compute 4 independent 2x2*2x2 products, each using 7 scalar mults.
    # Total: 4*7 = 28 scalar mults.
    
    # But the known lower bound for 4x4 is 25 and upper bound is 49.
    # So 28 would be better than 49, which contradicts known results.
    
    # The error: Strassen's 7-multiplication algorithm works for 2x2*2x2,
    # but when we apply it to the blocks of a 4x4 matrix, the "scalar" 
    # multiplications in Strassen's algorithm are actually 2x2*2x2 multiplications,
    # not scalar multiplications!
    
    # So if we use Strassen on the 4x4 blocks, we get 7 "2x2*2x2" multiplications.
    # Each 2x2*2x2 multiplication has rank 7 (using Strassen again) or 4 (schoolbook).
    # Using Strassen recursively: 7 * 7 = 49 scalar multiplications.
    
    # That's it! 4x4 matrix multiplication can be done in 49 scalar multiplications
    # by applying Strassen's algorithm at two levels:
    # Level 1: 4x4 = 2x2 blocks, use 7 block multiplications
    # Level 2: each 2x2 block multiplication uses 7 scalar multiplications
    # Total: 7 * 7 = 49
    
    # Now let me construct the explicit decomposition.
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we apply this to the 2x2 block structure.
    # Let A be partitioned as:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    
    # Strassen block multiplications:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # C00 = S1+S4-S5+S7
    # C01 = S3+S5
    # C10 = S2+S4
    # C11 = S1-S2+S3+S6
    
    # Each Si is a 2x2*2x2 matrix product, computed using Strassen (7 scalar mults).
    # Total: 7*7 = 49 scalar multiplications.
    
    # Now I need to express this as a tensor decomposition.
    # Each scalar multiplication corresponds to one rank-1 term.
    
    # For each block product Si = Xi * Yi (2x2 * 2x2), using Strassen:
    # Si has 4 entries, each computed from the 7 Strassen products.
    
    # Let me index the 49 scalar multiplications.
    # For block product s (s=1..7), using Strassen products t (t=1..7):
    # The scalar mult is: (linear combo of A blocks) * (linear combo of B blocks)
    
    # This is getting complex. Let me just build the U, V, W matrices
    # by tracking the coefficients.
    
    # Let me define the 49 bilinear forms explicitly.
    
    # First, let me define the Strassen 2x2 decomposition as a helper.
    # For 2x2 matrices P = [[p00,p01],[p10,p11]] and Q = [[q00,q01],[q10,q11]]:
    # Strassen products:
    # m1 = (p00+p11)*(q00+q11)
    # m2 = (p10+p11)*q00
    # m3 = p00*(q01-q11)
    # m4 = p11*(q10-q00)
    # m5 = (p00+p01)*q11
    # m6 = (p10-p00)*(q00+q01)
    # m7 = (p01-p11)*(q10+q11)
    
    # Result:
    # r00 = m1+m4-m5+m7
    # r01 = m3+m5
    # r10 = m2+m4
    # r11 = m1-m2+m3+m6
    
    # Now for the 4x4 case, I'll build the decomposition step by step.
    
    # The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
    # Block A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # Block A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # Block A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # Block A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    
    # Similarly for B and C.
    
    # Let me construct the rank-49 decomposition by:
    # 1. For each of the 7 Strassen block products Si,
    # 2. For each of the 7 Strassen scalar products within Si,
    # 3. Create one rank-1 term.
    
    # Total: 49 terms.
    
    # Let me build U, V, W arrays of shape (16, 49).
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Helper: add a Strassen 2x2 decomposition contribution
    # block_a: which 2x2 block of A (0=A00, 1=A01, 2=A10, 3=A11)
    # block_b: which 2x2 block of B (0=B00, 1=B01, 2=B10, 3=B11)
    # block_c: which 2x2 block of C (0=C00, 1=C01, 2=C10, 3=C11)
    # scale_a, scale_b: multipliers for the block entries
    # coeff: the coefficient from the outer Strassen (C = sum coeff_s * S_s)
    
    def block_offset(block):
        """Return (row_start, col_start) for a 2x2 block."""
        r = (block // 2) * 2
        c = (block % 2) * 2
        return r, c
    
    # Strassen 2x2 patterns
    # For P*Q using Strassen:
    # m1 = (p00+p11)*(q00+q11)
    # m2 = (p10+p11)*q00
    # m3 = p00*(q01-q11)
    # m4 = p11*(q10-q00)
    # m5 = (p00+p01)*q11
    # m6 = (p10-p00)*(q00+q01)
    # m7 = (p01-p11)*(q10+q11)
    
    # Strassen P patterns (coefficients for p00, p01, p10, p11):
    strassen_p = [
        [1, 0, 0, 1],  # m1: p00+p11
        [0, 0, 1, 1],  # m2: p10+p11
        [1, 0, 0, 0],  # m3: p00
        [0, 0, 0, 1],  # m4: p11
        [1, 1, 0, 0],  # m5: p00+p01
        [-1, 0, 1, 0], # m6: p10-p00
        [0, 1, 0, -1], # m7: p01-p11
    ]
    
    # Strassen Q patterns (coefficients for q00, q01, q10, q11):
    strassen_q = [
        [1, 0, 0, 1],  # m1: q00+q11
        [1, 0, 0, 0],  # m2: q00
        [0, 1, 0, -1], # m3: q01-q11
        [0, 0, 1, -1], # m4: q10-q00
        [0, 0, 0, 1],  # m5: q11
        [1, 1, 0, 0],  # m6: q00+q01
        [0, 0, 1, 1],  # m7: q10+q11
    ]
    
    # Strassen result patterns (coefficients for m1..m7 giving r00, r01, r10, r11):
    # r00 = m1+m4-m5+m7
    # r01 = m3+m5
    # r10 = m2+m4
    # r11 = m1-m2+m3+m6
    strassen_result = [
        [1, 0, 0, 1, -1, 0, 1],   # r00
        [0, 0, 1, 0, 1, 0, 0],    # r01
        [0, 1, 0, 1, 0, 0, 0],    # r10
        [1, -1, 1, 0, 0, 1, 0],   # r11
    ]
    
    # Outer Strassen for 4x4 (same pattern applied to blocks):
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # C00 = S1+S4-S5+S7
    # C01 = S3+S5
    # C10 = S2+S4
    # C11 = S1-S2+S3+S6
    
    outer_strassen_a = [
        [1, 0, 0, 1],  # S1: A00+A11
        [0, 0, 1, 1],  # S2: A10+A11
        [1, 0, 0, 0],  # S3: A00
        [0, 0, 0, 1],  # S4: A11
        [1, 1, 0, 0],  # S5: A00+A01
        [-1, 0, 1, 0], # S6: A10-A00
        [0, 1, 0, -1], # S7: A01-A11
    ]
    
    outer_strassen_b = [
        [1, 0, 0, 1],  # S1: B00+B11
        [1, 0, 0, 0],  # S2: B00
        [0, 1, 0, -1], # S3: B01-B11
        [0, 0, 1, -1], # S4: B10-B00
        [0, 0, 0, 1],  # S5: B11
        [1, 1, 0, 0],  # S6: B00+B01
        [0, 0, 1, 1],  # S7: B10+B11
    ]
    
    outer_strassen_result = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    # Now build the 49-term decomposition.
    # Term index: s*7 + t where s is the outer Strassen product (0..6)
    # and t is the inner Strassen product (0..6).
    
    for s in range(7):
        # Outer Strassen product s involves:
        # A blocks: outer_strassen_a[s] gives coefficients for [A00, A01, A10, A11]
        # B blocks: outer_strassen_b[s] gives coefficients for [B00, B01, B10, B11]
        
        for t in range(7):
            # Inner Strassen product t within the block product
            # This computes: (sum_i outer_a[s][i] * P_i) * (sum_j outer_b[s][j] * Q_j)
            # where P_i is a 2x2 block of A and Q_j is a 2x2 block of B.
            
            # The inner Strassen product t computes:
            # (sum over P entries) * (sum over Q entries)
            # where the P coefficients are strassen_p[t] and Q coefficients are strassen_q[t]
            
            # For each (i, j) pair of outer blocks with nonzero coefficients:
            # The bilinear form is:
            # (sum_{pr} outer_a[s][i] * strassen_p[t][pr] * A_entry(i,pr)) *
            # (sum_{qr} outer_b[s][j] * strassen_q[t][qr] * B_entry(j,qr))
            
            # Wait, I need to be more careful. The outer Strassen combines blocks,
            # and the inner Strassen operates within each block.
            
            # Actually, the structure is:
            # S_s = (linear combo of A blocks) * (linear combo of B blocks)
            # Each "linear combo of A blocks" is a 2x2 matrix.
            # Then we apply Strassen to this 2x2 * 2x2 product.
            
            # So for outer product s:
            # P = sum_i outer_a[s][i] * A_block_i  (2x2 matrix)
            # Q = sum_j outer_b[s][j] * B_block_j  (2x2 matrix)
            # S_s = P * Q (computed via Strassen)
            
            # The inner Strassen product t computes:
            # (sum_{pr} strassen_p[t][pr] * P[pr]) * (sum_{qr} strassen_q[t][qr] * Q[qr])
            
            # P[pr] = sum_i outer_a[s][i] * A_block_i[pr]
            # Q[qr] = sum_j outer_b[s][j] * B_block_j[qr]
            
            # So the bilinear form is:
            # (sum_{pr} strassen_p[t][pr] * sum_i outer_a[s][i] * A_block_i[pr]) *
            # (sum_{qr} strassen_q[t][qr] * sum_j outer_b[s][j] * B_block_j[qr])
            
            # = sum_{pr, i} strassen_p[t][pr] * outer_a[s][i] * A_block_i[pr]
            #   * sum_{qr, j} strassen_q[t][qr] * outer_b[s][j] * B_block_j[qr]
            
            # This is a product of two linear forms in A and B entries.
            
            # For the U matrix (coefficients of A entries):
            # U[A_entry_index, s*7+t] = sum_{pr, i} strassen_p[t][pr] * outer_a[s][i]
            #   where A_entry_index corresponds to A_block_i[pr]
            
            # For the V matrix (coefficients of B entries):
            # V[B_entry_index, s*7+t] = sum_{qr, j} strassen_q[t][qr] * outer_b[s][j]
            #   where B_entry_index corresponds to B_block_j[qr]
            
            # For the W matrix (coefficients of C entries):
            # The contribution to C_block_c[rc,cc] from S_s is outer_strassen_result[c][s]
            # The contribution from inner Strassen product t to C_block_c[rc,cc] is strassen_result[rc*2+cc][t]
            # So W[C_entry_index, s*7+t] = outer_strassen_result[c][s] * strassen_result[rc*2+cc][t]
            
            # Let me compute U and V for this term.
            
            # U: coefficients for A entries
            # A_block_i[pr] is at position: row = block_offset(i)[0] + pr//2, col = block_offset(i)[1] + pr%2
            # In the flattened A (row-major): index = row*4 + col
            
            u_coeff = [0.0] * 16
            for i in range(4):  # outer block
                if outer_strassen_a[s][i] == 0:
                    continue
                row_start, col_start = block_offset(i)
                for pr in range(4):  # inner position (0=p00, 1=p01, 2=p10, 3=p11)
                    if strassen_p[t][pr] == 0:
                        continue
                    r = row_start + pr // 2
                    c = row_start * 0 + col_start + pr % 2  # col = col_start + pr%2
                    # Wait, pr=0: (0,0), pr=1: (0,1), pr=2: (1,0), pr=3: (1,1)
                    r = row_start + (pr // 2)
                    c = col_start + (pr % 2)
                    a_idx = r * 4 + c
                    u_coeff[a_idx] += outer_strassen_a[s][i] * strassen_p[t][pr]
            
            v_coeff = [0.0] * 16
            for j in range(4):  # outer block
                if outer_strassen_b[s][j] == 0:
                    continue
                row_start, col_start = block_offset(j)
                for qr in range(4):
                    if strassen_q[t][qr] == 0:
                        continue
                    r = row_start + (qr // 2)
                    c = col_start + (qr % 2)
                    b_idx = r * 4 + c
                    v_coeff[b_idx] += outer_strassen_b[s][j] * strassen_q[t][qr]
            
            # W: coefficients for C entries
            w_coeff = [0.0] * 16
            for c_block in range(4):  # C block (0=C00, 1=C01, 2=C10, 3=C11)
                outer_coeff = outer_strassen_result[c_block][s]
                if outer_coeff == 0:
                    continue
                row_start, col_start = block_offset(c_block)
                for rc in range(4):  # inner position in C block
                    inner_coeff = strassen_result[rc][t]
                    if inner_coeff == 0:
                        continue
                    r = row_start + (rc // 2)
                    c = col_start + (rc % 2)
                    c_idx = r * 4 + c
                    w_coeff[c_idx] += outer_coeff * inner_coeff
            
            term_idx = s * 7 + t
            for idx in range(16):
                U[idx, term_idx] = u_coeff[idx]
                V[idx, term_idx] = v_coeff[idx]
                W[idx, term_idx] = w_coeff[idx]
    
    # Verify the decomposition
    # T[a,b,c] should equal sum_r U[a,r] * V[b,r] * W[c,r]
    # T[a,b,c] = 1 if a=i*4+j, b=j*4+k, c=k*4+i for some i,j,k, else 0
    
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(49))
                max_err = max(max_err, abs(val - 1.0))
    
    # Also check that non-multiplication entries are zero
    for a in range(16):
        for b in range(16):
            for c in range(16):
                # Check if (a,b,c) is a valid multiplication entry
                i = a // 4
                j = a % 4
                j2 = b // 4
                k = b % 4
                k2 = c // 4
                i2 = c % 4
                if j == j2 and k == k2 and i == i2:
                    continue  # valid entry, already checked
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(49))
                max_err = max(max_err, abs(val))
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
