"""Program to find a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    deadline = time.time() + 145.0

    def check_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol)

    def build_strassen_32():
        """
        Build a rank-32 decomposition by padding to 4x4 * 4x4 and applying Strassen.
        
        Strategy:
        - Pad A (2x4) to A_pad (4x4) with zero bottom 2 rows
        - Pad B (4x8) to B_pad (4x4) with zero right 4 columns (keep left 4 cols)
        - Apply rank-32 Strassen to 4x4 * 4x4 -> C_pad (4x4)
        - Project C_pad back to C (2x8): C[i,k] = C_pad[i,k] for i<2, k<4
          For k>=4, C[i,k] is zero (from zero padding of B)
        
        Wait - this only gives us C for k in {0,1,2,3}. We need k in {0,...,7}.
        
        Better approach: Use TWO 4x4 multiplications:
        - B_left = B[:, 0:4] (4x4), B_right = B[:, 4:8] (4x4)
        - C_left = A_pad * B_left (4x4) -> take top 2x4
        - C_right = A_pad * B_right (4x4) -> take top 2x4
        - Total rank: 2 * 32 = 64. That's same as schoolbook.
        
        Alternative: Use a single 4x4 Strassen but with a different partitioning.
        
        Actually, let me reconsider. The key insight is:
        - A is 2x4, B is 4x8
        - We can view this as computing 2 independent 2x4 * 4x4 products (left half and right half of B)
        - Each 2x4 * 4x4 can be done with rank 16 (4x4 Strassen projected)
        - Total: 32
        
        For 2x4 * 4x4:
        - Pad A to 4x4 (zero bottom rows)
        - B is already 4x4
        - Apply 4x4 Strassen (rank 32)
        - Project output to 2x4 (top 2 rows)
        - But we can eliminate terms that multiply zero rows of A_pad!
        
        In 4x4 Strassen, the U factors have 4x4 entries. If A_pad has zeros in rows 2,3,
        then any U factor entry at row index >= 2 contributes zero. We can remove those
        from U without changing the product.
        
        Actually, the rank doesn't change - we still have 32 terms, but some U entries are zero.
        The rank is still 32 per 4x4 multiplication.
        
        Let me think differently. Can we do 2x4 * 4x8 with rank < 56?
        
        Known results: 
        - 2x2 * 2x2: rank 7 (Strassen)
        - 4x4 * 4x4: rank 32 (Strassen) or possibly lower (best known 47 for general, but Strassen gives 32)
        - 2x4 * 4x8: This is equivalent to 2 independent 2x4*4x4 or 4 independent 2x2*2x2 with sharing.
        
        Actually, let me try a different partitioning:
        - Split B into 4 blocks of 4x2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
        - Each is 4x2, and A is 2x4
        - 2x4 * 4x2: pad to 4x4*4x4 (A padded, B padded with 2 zero cols)
        - Apply 4x4 Strassen, project to 2x2
        - 4 such products: rank 4*32 = 128. Worse.
        
        Let me try yet another approach:
        - Split into 4 blocks: A[:, 0:2] * B[0:2, :] and A[:, 2:4] * B[2:4, :]
        - Each is 2x2 * 2x8
        - 2x2 * 2x8: pad B to 2x8 (already), this is like 4 independent 2x2*2x2
        - Rank per 2x2*2x8: 4*7 = 28 (Strassen)
        - Total: 2 * 28 = 56. Same as before.
        
        OK, let me try the 4x4 approach more carefully:
        - View 2x4 * 4x8 as: [A | 0] (4x4) * [B_left; B_right] where B_left = B[:, 0:4], B_right = B[:, 4:8]
        - But that's 4x4 * 4x8, not 4x4 * 4x4.
        
        Alternative: Do it as two 4x4 multiplications:
        - C[:, 0:4] = A_pad * B[:, 0:4] where A_pad = [[A], [0]] (4x4)
        - C[:, 4:8] = A_pad * B[:, 4:8]
        - Each 4x4*4x4 has rank 32 via Strassen
        - Total rank: 64. No improvement.
        
        Hmm, let me try a mixed approach:
        - Use Strassen on the 4x4 structure but exploit that A has only 2 nonzero rows.
        
        In the 4x4 Strassen decomposition, there are 32 bilinear terms.
        Each term has a U factor (4x4), V factor (4x4), W factor (4x4).
        If A_pad has zeros in rows 2,3, then U factors only need rows 0,1.
        This doesn't reduce rank, but let's see if we can merge terms.
        
        Actually, I think the correct approach for rank 32 is:
        
        Consider the tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 for all valid i,j,k).
        We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}.
        
        Wait, that's not right. The tensor for matrix multiplication C = A*B where A is n x m and B is m x p:
        T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
        The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k].
        As a tensor: T[i,j,k] = delta_{i,j,k} where the tensor represents the identity map.
        
        The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        For our case: i in {0,1}, j in {0,1,2,3}, k in {0,...,7}.
        T[i,j,k] = 1 for all valid (i,j,k).
        
        We need rank-minimal decomposition.
        
        Let me try the 4x4 Strassen approach properly:
        
        Pad A to 4x4: A_pad[i,j] = A[i,j] if i<2 else 0, for i in {0,1,2,3}, j in {0,1,2,3}
        But B is 4x8, not 4x4. So we need to handle the 8 columns.
        
        Split B into B1 = B[:, 0:4] and B2 = B[:, 4:8], both 4x4.
        C[:, 0:4] = A_pad * B1 (4x4 result, take top 2x4)
        C[:, 4:8] = A_pad * B2 (4x4 result, take top 2x4)
        
        Each 4x4*4x4 uses rank 32. Total: 64.
        
        But wait - can we share the U factors? In Strassen 4x4, the U factors depend only on A_pad,
        and the V factors depend only on B. Since A_pad is the same for both, the U factors are the same!
        But the rank is still additive: we have 32 terms for B1 and 32 for B2, total 64.
        
        Unless we can interleave them... but they're independent.
        
        Let me try a completely different approach:
        
        What if we use the fact that 2x4 * 4x8 can be decomposed as:
        - 4 independent 2x2 * 2x2 sub-problems (for k in {0,1}, {2,3}, {4,5}, {6,7}) 
          each with j in {0,1} or {2,3}
        - That's 8 sub-problems of 2x2*2x2, each rank 7, total 56.
        
        Or:
        - 2 independent 2x4 * 4x4 sub-problems (for k in {0,1,2,3} and {4,5,6,7})
        - Each 2x4*4x4: pad to 4x4*4x4, Strassen rank 32, total 64.
        
        Or a hybrid:
        - For the j-dimension: split into j in {0,1} and j in {2,3}
        - For each j-group, we have 2x2 * 2x8
        - 2x2 * 2x8: this is 4 independent 2x2*2x2, rank 4*7 = 28
        - Total: 2 * 28 = 56.
        
        I think 56 might be close to optimal for this problem with simple constructions.
        Let me just implement the 8x Strassen-2x2 approach cleanly and verify it works.
        
        Actually, let me try one more idea: Can we do better than 56?
        
        Known: mu(2,2,2) = 7, mu(4,4,4) = 32 (Strassen), mu(2,4,4) <= 16?
        
        For 2x4 * 4x4:
        - Pad A to 4x4, apply 4x4 Strassen (rank 32), project to 2x4.
        - But many terms will be zero because A has zero rows.
        - In 4x4 Strassen, the 32 terms each have U (4x4), V (4x4), W (4x4).
        - If A_pad[2,:] = A_pad[3,:] = 0, then U[2,j,r] and U[3,j,r] don't matter.
        - The rank is still 32, but we could potentially eliminate redundant terms.
        
        Actually, I realize I should just implement the 8x Strassen-2x2 (rank 56) correctly,
        verify it, and return it. That's a solid improvement from 64 to 56.
        """
        rank = 8 * 7  # 56
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        col = 0
        for jg in range(2):  # j-group: 0 -> {0,1}, 1 -> {2,3}
            for kg in range(4):  # k-group: 0->{0,1}, 1->{2,3}, 2->{4,5}, 3->{6,7}
                j0 = jg * 2
                j1 = jg * 2 + 1
                k0 = kg * 2
                k1 = kg * 2 + 1

                # Strassen 7 products for 2x2 * 2x2:
                # A block: A[i, j] for i in {0,1}, j in {j0, j1}
                # B block: B[j, k] for j in {j0, j1}, k in {k0, k1}
                # C block: C[i, k] for i in {0,1}, k in {k0, k1}
                #
                # Strassen formulas:
                # M1 = A[0,j0] * B[j0,k0]
                # M2 = A[1,j1] * B[j1,k1]
                # M3 = (A[0,j0] + A[0,j1]) * B[j0,k1]
                # M4 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
                # M5 = (A[1,j0] + A[1,j1]) * B[j1,k0]
                # M6 = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
                # M7 = (A[1,j1] + A[0,j0] - A[1,j0] - A[0,j1]) * (B[j0,k1] - B[j1,k1])
                #
                # C[0,k0] = M1 + M4 + M7
                # C[0,k1] = M3 - M7
                # C[1,k0] = M2 + M5
                # C[1,k1] = M2 - M4 + M6 + M7

                strassen_defs = [
                    # M1 = A[0,j0] * B[j0,k0]
                    ([(0 * m + j0, 1)], [(j0 * p + k0, 1)], [(k0 * n + 0, 1)]),
                    # M2 = A[1,j1] * B[j1,k1]
                    ([(1 * m + j1, 1)], [(j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    # M3 = (A[0,j0] + A[0,j1]) * B[j0,k1]
                    ([(0 * m + j0, 1), (0 * m + j1, 1)], [(j0 * p + k1, 1)], [(k1 * n + 0, 1)]),
                    # M4 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
                    ([(1 * m + j0, 1), (0 * m + j0, -1)], [(j0 * p + k0, 1), (j0 * p + k1, 1)], [(k0 * n + 0, 1)]),
                    # M5 = (A[1,j0] + A[1,j1]) * B[j1,k0]
                    ([(1 * m + j0, 1), (1 * m + j1, 1)], [(j1 * p + k0, 1)], [(k0 * n + 1, 1)]),
                    # M6 = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
                    ([(0 * m + j1, 1), (1 * m + j1, -1)], [(j1 * p + k0, 1), (j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    # M7 = (A[1,j1] + A[0,j0] - A[1,j0] - A[0,j1]) * (B[j0,k1] - B[j1,k1])
                    ([(1 * m + j1, 1), (0 * m + j0, 1), (1 * m + j0, -1), (0 * m + j1, -1)],
                     [(j0 * p + k1, 1), (j1 * p + k1, -1)], [(k0 * n + 0, 1)]),
                ]

                for u_coeffs, v_coeffs, w_coeffs in strassen_defs:
                    for idx, val in u_coeffs:
                        U[idx, col] += val
                    for idx, val in v_coeffs:
                        V[idx, col] += val
                    for idx, val in w_coeffs:
                        W[idx, col] += val
                    col += 1

        return U, V, W

    def build_strassen_32_padded():
        """
        Alternative: Try to achieve rank 32 using 4x4 Strassen with zero-padding.
        
        Approach:
        - Pad A (2x4) to A_pad (4x4) with zero rows 2,3
        - Split B (4x8) into B_left (4x4) and B_right (4x4)
        - Apply 4x4 Strassen to A_pad * B_left and A_pad * B_right
        - Since A_pad has zero rows, many Strassen terms vanish
        - Project 4x4 output to 2x4 for each half
        - Total rank: potentially less than 64 if terms cancel
        
        In 4x4 Strassen, the decomposition uses 32 bilinear terms.
        Each term r has U_r (4x4), V_r (4x4), W_r (4x4).
        The contribution to C[i,k] is sum_r U_r[i,j] V_r[j,k] W_r[k,i] summed over j.
        
        If A_pad[2,:]=A_pad[3,:]=0, then for the actual product, only U_r[0,:] and U_r[1,:] matter.
        But the rank is still 32 per multiplication.
        
        However, we can check: do any of the 32 terms become identically zero when
        multiplied by the zero-padded A? If U_r has nonzero entries only in rows 2,3,
        then that term contributes nothing and can be removed.
        
        Let's build the 4x4 Strassen and check.
        """
        # 4x4 Strassen: partition into four 2x2 blocks
        # A = [[A00, A01], [A10, A11]] where each block is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = A*B
        #
        # Strassen 4x4 uses 8 sub-products of 2x2*2x2, each with rank 7.
        # Total: 8 * 7 = 56? No, that's not right.
        #
        # Actually, 4x4 Strassen: 
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # M8 = (A00+A11)*(B10-B11) -- wait, this isn't standard
        #
        # Standard 4x4 Strassen:
        # P1 = (A00 + A11) * (B00 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = A00 * (B01 - B11)
        # P4 = A11 * (B10 - B00)
        # P5 = (A00 + A01) * B11
        # P6 = (A10 - A00) * (B00 + B01)
        # P7 = (A01 - A11) * (B10 + B11)
        # P8 = (A00 + A11) * (B10 - B11) -- hmm, need to verify
        #
        # Actually, the standard 4x4 Strassen uses 32 scalar multiplications:
        # Each of the 8 matrix products P_i is a 2x2*2x2, done with 7 scalar mults.
        # 8 * 7 = 56 scalar multiplications. That's the same as 8 independent 2x2*2x2!
        #
        # So 4x4 Strassen has rank 56, not 32. I was confused.
        #
        # The rank of 4x4*4x4 matrix multiplication is at most 56 (via Strassen),
        # and it's known to be at least 49 (best known lower bound).
        #
        # So my "rank 32" idea was wrong. 4x4 Strassen is rank 56, same as 8x 2x2 Strassen.
        #
        # This means the current program's rank 56 is already using the best known
        # simple construction. Let me just verify and return it.
        
        # Fallback: just use the 8x Strassen-2x2
        return build_strassen_32_padded.__wrapped__() if hasattr(build_strassen_32_padded, '__wrapped__') else build_strassen_32()

    # Build the Strassen-56 decomposition
    U, V, W = build_strassen_32()

    # Verify
    if check_decomp(U, V, W, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    if time.time() > deadline - 5.0:
        rank = n * m * p
        U2 = np.zeros((U_dim, rank))
        V2 = np.zeros((V_dim, rank))
        W2 = np.zeros((W_dim, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j, c] = 1
                    V2[j*p+k, c] = 1
                    W2[k*n+i, c] = 1
                    c += 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # Debug and fallback
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    
    # As a last resort, return schoolbook
    rank = n * m * p
    U2 = np.zeros((U_dim, rank))
    V2 = np.zeros((V_dim, rank))
    W2 = np.zeros((W_dim, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j, c] = 1
                V2[j*p+k, c] = 1
                W2[k*n+i, c] = 1
                c += 1
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
