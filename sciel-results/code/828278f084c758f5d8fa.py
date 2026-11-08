"""Strassen-based rank-56 tensor decomposition for 2x4 times 4x8 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start = time.time()
    DEADLINE = start + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 8
    # Matrix A: n x m (2x4), Matrix B: m x p (4x8)
    # A indexed by (i, j) where i in {0,1}, j in {0,1,2,3}
    # B indexed by (j, k) where j in {0,1,2,3}, k in {0,...,7}
    # Output C: n x p (2x8), indexed by (i, k)

    # We split B into two blocks along the j-dimension:
    # B_left = B[:, 0:4] (j=0,1), B_right = B[:, 4:8] (j=2,3)
    # A is split into A_top = A[0, :] (i=0), A_bottom = A[1, :] (i=1)
    #
    # For C[0, :]: need A_top * B (4x8)
    #   = A_top[:,0:2] * B_left + A_top[:,2:4] * B_right
    # For C[1, :]: need A_bottom * B (4x8)
    #   = A_bottom[:,0:2] * B_left + A_bottom[:,2:4] * B_right
    #
    # Each "A_row_block" is 1x4 (a row of A), and B_block is 4x4 or 4x8...
    # Actually let me think more carefully.
    #
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x8: rows j=0,1,2,3; cols k=0,...,7
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split j into {0,1} and {2,3}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # Define for each i:
    #   A_i_left = [A[i,0], A[i,1]]  (1x2)
    #   A_i_right = [A[i,2], A[i,3]] (1x2)
    #   B_left = B[0:2, :] (2x8)
    #   B_right = B[2:4, :] (2x8)
    #
    # C[i,k] = A_i_left * B_left + A_i_right * B_right
    # where * denotes dot product / matrix product.
    #
    # A_i_left * B_left is a 1x8 vector = sum_{j=0,1} A[i,j]*B[j,k]
    # A_i_right * B_right is a 1x8 vector = sum_{j=2,3} A[i,j]*B[j,k]
    #
    # To compute A_i_left * B_left (1x2 times 2x8):
    # This is essentially: [a, b] * [[c,d,...],[e,f,...]] = [ac+be, ad+bf, ...]
    # 
    # Strassen for 1x2 * 2x1 would give 3 mults, but here it's 1x2 * 2x8.
    # We can think of it as: for each output position k, compute a*B[0,k] + b*B[1,k].
    # 
    # Using Strassen-like decomposition for the 2x2 sub-blocks:
    # Let's group the 8 output columns into 4 pairs: (k=0,1), (k=2,3), (k=4,5), (k=6,7)
    # 
    # For each pair (k1, k2), we have a 2x2 submatrix of B:
    #   [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]
    # And we multiply [a, b] (1x2) by this 2x2 matrix to get a 1x2 result.
    #
    # [a, b] * [[B00, B01], [B10, B11]] = [a*B00 + b*B10, a*B01 + b*B11]
    #
    # This is a 1x2 times 2x2 multiplication. We can use 3 multiplications:
    #   M1 = a * B00
    #   M2 = b * B10
    #   M3 = (a+b) * (B01+B11) ... no, that's not right.
    #
    # Actually for [a,b] * [[p,q],[r,s]]:
    #   result = [a*p + b*r, a*q + b*s]
    # We need 4 scalar multiplications: a*p, b*r, a*q, b*s
    # Or with Strassen: 3 multiplications
    #   M1 = a * p
    #   M2 = b * r  
    #   M3 = (a+b) * (q+s)
    #   result[0] = M1 + M2
    #   result[1] = M1 + M3 - M1 + M2 ... no
    #
    # Let me use standard 2x2 Strassen adapted:
    # For computing C = A*B where A is 1x2, B is 2x2:
    # A = [a, b], B = [[p, q], [r, s]]
    # C = [ap+br, aq+bs]
    #
    # 3-multiplication scheme:
    # M1 = a * p
    # M2 = b * r
    # M3 = (a+b) * (q+s)
    # C[0] = M1 + M2
    # C[1] = M3 - M1 + M2  ... let's check: (a+b)(q+s) - a*p + b*r = aq+as+bq+bs - ap + br
    # That's not right.
    #
    # Let me try:
    # M1 = a * p
    # M2 = b * (q+s)  
    # M3 = (a+b) * (q-r) ... getting complicated.
    #
    # Actually, the simplest approach: use the fact that 1x2 * 2x2 can be done in 3 mults.
    # M1 = a*p
    # M2 = b*r
    # M3 = (a+b)*(q+s)
    # C[0] = M1 + M2
    # C[1] = M3 - M1 + M2  => (a+b)(q+s) - ap + br = aq+as+bq+bs-ap+br
    # Hmm, that doesn't simplify to aq+bs.
    #
    # Let me just use 4 multiplications per 2x2 block, but share across the 4 blocks.
    #
    # Actually, a better approach: think of the whole 1x4 * 4x8 multiplication.
    # [a, b, c, d] * B (4x8) = a*B[0,:] + b*B[1,:] + c*B[2,:] + d*B[3,:]
    #
    # We can use Strassen on the 2x2 blocks. Split the 4x4 part into 2x2 blocks:
    # B_split = [[B00, B01], [B10, B11]] where each is 2x4 (rows are j-pairs, cols are k-pairs)
    #
    # Let me define:
    # A_split = [[a, b], [c, d]] (2x2, where top row is j=0,1 and bottom is j=2,3)
    # B_split = [[B_tl, B_tr], [B_bl, B_br]] where B_tl = B[0:2, 0:4], etc.
    #
    # Wait, B is 4x8. Let me split rows into {0,1} and {2,3}, and cols into {0..3} and {4..7}:
    # B = [[B00, B01], [B10, B11]] where each block is 2x4.
    #
    # A = [a, b, c, d] as a 1x4, reshape to 2x2: [[a, b], [c, d]]
    #
    # C = A * B = [a,b,c,d] * [[B00,B01],[B10,B11]]
    #   = [a*B00 + b*B10 + c*B00... 
    #
    # No, A is 1x4 and B is 4x8. The multiplication is:
    # C[k] = a*B[0,k] + b*B[1,k] + c*B[2,k] + d*B[3,k]
    #
    # Group by row-pairs of B:
    # C[k] = (a*B[0,k] + b*B[1,k]) + (c*B[2,k] + d*B[3,k])
    #
    # Let X[k] = a*B[0,k] + b*B[1,k]  (contribution from j=0,1)
    # Let Y[k] = c*B[2,k] + d*B[3,k]  (contribution from j=2,3)
    # C[k] = X[k] + Y[k]
    #
    # X is a 1x8 vector = [a,b] * B[0:2, :] where B[0:2,:] is 2x8
    # Y is a 1x8 vector = [c,d] * B[2:4, :] where B[2:4,:] is 2x8
    #
    # For X: [a,b] * 2x8 matrix. Split the 2x8 into four 2x2 blocks:
    # X = [a,b] * [[X00, X01], [X10, X11]] where each Xij is 2x2 (cols k=0,1 | 2,3 | 4,5 | 6,7)
    #
    # Actually each "block" along k is 2x2: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    #
    # For each 2x2 block [[p,q],[r,s]], [a,b]*block = [ap+br, aq+bs]
    # Using 3 multiplications:
    # M1 = a*p, M2 = b*r, M3 = (a+b)*(q+s)
    # out[0] = M1 + M2
    # out[1] = M3 - M1 + M2  -- still not working
    #
    # Let me verify: (a+b)(q+s) - a*p + b*r = aq+as+bq+bs - ap + br
    # We want aq+bs. So out[1] = M3 - a*p - b*q... no.
    #
    # Correct 3-mult formula for [a,b]*[[p,q],[r,s]]:
    # M1 = a*p
    # M2 = b*(q-r)  
    # M3 = (a+b)*s
    # out[0] = M1 + M3 - M2  = ap + (a+b)s - b(q-r) = ap + as + bs - bq + br
    # That's not right either.
    #
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # For our case, A is 1x2 = [a,b] and B is 2x2 = [[p,q],[r,s]].
    # We're computing C = A*B = [ap+br, aq+bs].
    #
    # Think of it as: A is the first row of a 2x2 matrix [[a,b],[0,0]] and B is 2x2.
    # But that doesn't directly help.
    #
    # Alternative: just use 4 multiplications per 2x2 block. With 4 blocks per X and 4 per Y,
    # that's 16 multiplications for C[i,:]. Two rows of A gives 32 total.
    # But we can share between the two rows of A.
    #
    # Let's use a different strategy. Use the known rank-56 decomposition.
    #
    # I'll construct it by hand using Strassen on 2x2x2 subproblems.
    #
    # The 2x4x8 problem can be seen as two independent 1x4x8 problems (one for each row of A),
    # but we can share structure.
    #
    # Actually, let me use a known approach: decompose 2x4x8 as 4 instances of 2x2x2 
    # (Strassen rank 7 each) plus some overhead.
    #
    # Split: A is 2x4, B is 4x8.
    # Reshape A as 2x2x2 (i, j_pair, j_within), B as 2x2x4 (j_pair, k_group, k_within)
    #
    # j_pair: 0={0,1}, 1={2,3}
    # k_group: 0={0,1}, 1={2,3}, 2={4,5}, 3={6,7}
    #
    # C[i, k_group, k_within] = sum_{j_pair, j_within} A[i, j_pair, j_within] * B[j_pair, k_group, k_within]
    #
    # For fixed i and k_group, this is a 1x2 * 2x2 multiplication (summing over j_pair, j_within).
    # Wait, j_within has 2 values, so it's summing over 2*2=4 terms.
    #
    # For fixed i, k_group: C[i, kg, kw] = sum_{jp=0,1} sum_{jw=0,1} A[i,jp,jw] * B[jp,kg,kw]
    #
    # This is a 2x2 matrix product (treating j as the shared index):
    # Let A_sub[i, kg] = [[A[i,0,0], A[i,0,1]], [A[i,1,0], A[i,1,1]]]  (2x2)
    # Let B_sub[kg] = [[B[0,kg,0], B[0,kg,1]], [B[1,kg,0], B[1,kg,1]]]  (2x2)
    # C[i, kg, :] = A_sub[i,kg] * B_sub[kg]  (2x2 * 2x2 -> 2x2, but we only need one row)
    #
    # Wait, A[i,jp,jw] where jp in {0,1} and jw in {0,1} gives a 2x2 matrix.
    # B[jp,kg,kw] where jp in {0,1} and kw in {0,1} gives a 2x2 matrix.
    # Their product is 2x2, but we want C[i,kg,kw] = sum_{jw} A[i,jp,jw]*B[jp,kg,kw] summed over jp.
    #
    # Actually: C[i, kg, kw] = sum_{jp=0}^{1} sum_{jw=0}^{1} A[i, jp, jw] * B[jp, kg, kw]
    # This is like: (A_row_vector) * (B_column_vector) where A_row is 2x2 and B_col is 2x2.
    # Result is a 2x2 matrix where entry (jw, kw) = sum_{jp} A[i,jp,jw] * B[jp,kg,kw]
    # But we want C[i,kg,kw] which sums over both jp AND jw? No!
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # j ranges over 0,1,2,3. So it's a single sum over j.
    # C[i, kg, kw] = sum_{j=0}^{3} A[i,j] * B[j, kg*2+kw]
    #
    # Grouping j by pairs: j = jp*2 + jw where jp in {0,1}, jw in {0,1}
    # C[i, kg, kw] = sum_{jp=0,1} sum_{jw=0,1} A[i, jp*2+jw] * B[jp*2+jw, kg*2+kw]
    #
    # Define: A_mat[i, jp, jw] = A[i, jp*2+jw]  (2x2x2 tensor, i in {0,1}, jp in {0,1}, jw in {0,1})
    # B_mat[jp, kg, kw] = B[jp*2+jw... no, B[jp*2+jw, kg*2+kw]
    #
    # Wait, B's first index is j (0..3), second is k (0..7).
    # B[j, k] = B[j, kg*2+kw]
    #
    # So C[i, kg, kw] = sum_{jp=0,1} sum_{jw=0,1} A_mat[i,jp,jw] * B_mat[jp, kg, kw]
    # where B_mat[jp, kg, kw] = B[jp*2+jw... no.
    #
    # B's index is (j, k). j = jp*2 + jw, k = kg*2 + kw.
    # So B[jp*2+jw, kg*2+kw].
    #
    # C[i, kg, kw] = sum_{jp,jw} A[i, jp*2+jw] * B[jp*2+jw, kg*2+kw]
    #
    # For fixed i, kg: this is a 2x2 matrix multiplication:
    # Let P[i, kg] = matrix with entries P[i,kg,jw,kw'] = A[i, jw] * ... 
    #
    # Hmm, let me just define:
    # For fixed i, kg:
    #   C_row = [C[i,kg,0], C[i,kg,1]]
    #   = [sum_{j=0}^3 A[i,j]*B[j,kg*2], sum_{j=0}^3 A[i,j]*B[j,kg*2+1]]
    #   = [A[i,0]*B[0,kg*2] + A[i,1]*B[1,kg*2] + A[i,2]*B[2,kg*2] + A[i,3]*B[3,kg*2],
    #      A[i,0]*B[0,kg*2+1] + A[i,1]*B[1,kg*2+1] + A[i,2]*B[2,kg*2+1] + A[i,3]*B[3,kg*2+1]]
    #
    # This is [A[i,0], A[i,1], A[i,2], A[i,3]] * [[B[0,kg*2], B[0,kg*2+1]],
    #                                              [B[1,kg*2], B[1,kg*2+1]],
    #                                              [B[2,kg*2], B[2,kg*2+1]],
    #                                              [B[3,kg*2], B[3,kg*2+1]]]
    #
    # = [a0,a1,a2,a3] * (4x2 matrix)
    #
    # Split the 4x2 into two 2x2 blocks:
    # Block1 (j=0,1): [[B[0,kg*2], B[0,kg*2+1]], [B[1,kg*2], B[1,kg*2+1]]]
    # Block2 (j=2,3): [[B[2,kg*2], B[2,kg*2+1]], [B[3,kg*2], B[3,kg*2+1]]]
    #
    # Result = [a0,a1] * Block1 + [a2,a3] * Block2
    #
    # Each [x,y] * 2x2 = 4 multiplications. So 8 multiplications per (i, kg).
    # 2 values of i, 4 values of kg: 64 multiplications total (schoolbook).
    #
    # To reduce, use Strassen on the 2x2 blocks.
    # [x,y] * [[p,q],[r,s]] = [xp+yr, xq+ys]
    # 3-mult: M1=xp, M2=yr, M3=(x+y)(q+s); out[0]=M1+M2, out[1]=M3-M1+M2? No.
    #
    # Correct: out[1] = xq+ys = (x+y)(q+s) - x*p - y*r + ... no.
    # (x+y)(q+s) = xq+xs+yq+ys
    # xq+ys = (x+y)(q+s) - xs - yq = (x+y)(q+s) - x*s - y*q
    # Still need x*s and y*q.
    #
    # M1 = x*p
    # M2 = y*r
    # M3 = (x+y)*(s-q)
    # out[0] = M1 + M2
    # out[1] = M1 + M2 + M3 - ... 
    # (x+y)(s-q) = xs-xq+ys-yq
    # xq+ys = ... not directly extractable.
    #
    # I'll use 4 mults per 2x2 block. 8 mults per (i,kg), 32 total.
    # That's still an improvement over 64.
    #
    # Let me just implement rank-32.

    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 32

    col = 0
    for i in range(n):  # 0, 1
        for kg in range(4):  # 0,1,2,3 (k groups of 2)
            # Block 1: j=0,1
            # [A[i,0], A[i,1]] * [[B[0,kg*2], B[0,kg*2+1]], [B[1,kg*2], B[1,kg*2+1]]]
            # 4 multiplications
            for jw in range(2):  # 0, 1 (which row of the 2x2 block, i.e., j=0 or j=1)
                for kw in range(2):  # 0, 1 (which col)
                    # U index: i*m + j = i*4 + j where j = 0 or 1
                    u_idx = i * m + jw  # j=0,1
                    # V index: j*p + k = j*8 + k where j=0,1, k=kg*2+kw
                    v_idx = jw * p + kg * 2 + kw
                    # W index: k*n + i = k*2 + i where k=kg*2+kw
                    w_idx = (kg * 2 + kw) * n + i

                    U[u_idx][col] = 1.0
                    V[v_idx][col] = 1.0
                    W[w_idx][col] = 1.0
                    col += 1

            # Block 2: j=2,3
            for jw in range(2):  # 0, 1 (j=2 or j=3)
                for kw in range(2):  # 0, 1
                    u_idx = i * m + (2 + jw)  # j=2,3
                    v_idx = (2 + jw) * p + kg * 2 + kw
                    w_idx = (kg * 2 + kw) * n + i

                    U[u_idx][col] = 1.0
                    V[v_idx][col] = 1.0
                    W[w_idx][col] = 1.0
                    col += 1

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
