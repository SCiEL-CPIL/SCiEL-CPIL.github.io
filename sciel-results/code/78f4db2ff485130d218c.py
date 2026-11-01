import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = 2, 4, 5
    dim_a = n * m  # 8
    dim_b = m * p  # 20
    dim_c = p * n  # 10

    # Indexing conventions:
    # A is n x m = 2 x 4, index a = i*m + j  (i in 0..1, j in 0..3)
    # B is m x p = 4 x 5, index b = j'*p + k (j' in 0..3, k in 0..4)
    # C is p x n = 5 x 2, index c = k'*n + i' (k' in 0..4, i' in 0..1)
    # C[k', i'] = sum_{j=0}^{3} A[i, j] * B[j, k']

    # Build the multiplication tensor T[a, b, c] = 1 iff i==i', j==j', k==k'
    T = np.zeros((dim_a, dim_b, dim_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Rank-36 block-Strassen decomposition.
    #
    # Structure: Split A into left half A_L (cols 0,1) and right half A_R (cols 2,3),
    # each 2x2. Split B into top half B_T (rows 0,1) and bottom half B_B (rows 2,3),
    # each 2x5.
    #
    # C = A_L @ B_T + A_R @ B_B
    #
    # Each 2x2 @ 2x5 = 2x5 product.
    # A 2x2 @ 2x5 can be viewed as: for each of the 5 columns of B, multiply 2x2 by 2x1.
    # But more efficiently: split the 2x5 into a 2x2 + 2x2 + 2x1.
    #   B_2x5 = [B1(2x2), B2(2x2), B3(2x1)]
    #   A @ B_2x5 = A@B1 + A@B2 + A@B3
    #
    # A@B1 (2x2 @ 2x2 = 2x2): use Strassen rank 7
    # A@B2 (2x2 @ 2x2 = 2x2): use Strassen rank 7
    # A@B3 (2x2 @ 2x1 = 2x1): trivial, rank 2
    #
    # So per 2x2@2x5 block: 7+7+2 = 16
    # Two such blocks: 16*2 = 32
    #
    # Wait, that gives 32, not 36. Let me reconsider.
    #
    # Actually, the 2x5 has 5 columns. Split as 2x2 + 2x2 + 2x1 gives 5 columns. Good.
    # But we have TWO 2x2@2x5 blocks (A_L@B_T and A_R@B_B).
    # Each needs 7+7+2 = 16. Total = 32.
    #
    # Hmm, but the known result is 36. Let me re-examine.
    #
    # Alternative split: Maybe the 2x5 should be split differently.
    # 2x5 = 2x1 + 2x1 + 2x1 + 2x1 + 2x1 (five 2x1 columns)
    # Each 2x2 @ 2x1 = 2x1: rank 2. Five of them: 10.
    # Two blocks: 20. That's worse.
    #
    # Or: 2x5 = 2x2 + 2x3.
    # 2x2 @ 2x2: Strassen rank 7.
    # 2x2 @ 2x3: split 2x3 = 2x2 + 2x1. 2x2@2x2 (rank 7) + 2x2@2x1 (rank 2) = 9.
    # Total per block: 7 + 9 = 16. Two blocks: 32. Still 32.
    #
    # Let me think about this differently. The known rank-36 for 2x4x5:
    # Perhaps the decomposition is:
    #   C = A_L @ B_T + A_R @ B_B
    # where A_L, A_R are 2x2 and B_T, B_B are 2x5.
    #
    # For 2x2 @ 2x5:
    #   Let A = [[a,b],[c,d]], B = [b1 b2 b3 b4 b5] (each bi is 2x1)
    #   C = A @ B = [a*b1+b*b3, a*b2+b*b4, a*b3+b*b5, ...]
    #
    # Actually, 2x2 @ 2x5:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # This is 2*5 = 10 scalar multiplications in schoolbook.
    # With Strassen on the 2x2 part: we can compute A@B by treating B as 5 columns.
    # Strassen computes M1..M7 from A and B (where B is 2x2).
    # For B being 2x5, we can apply Strassen's formulas column-wise:
    #   M1 = (A[0,0]+A[1,1]) * (B[0,:]+B[1,:])  -- but this is 2x5 result
    #   Actually Strassen works on 2x2 blocks. For 2x2 @ 2x5, we can split B into
    #   B_left (2x2) and B_right (2x3), or handle it as:
    #
    #   Let's just use: 2x2 @ 2x5 requires rank at most 7 (same as 2x2@2x2)
    #   because the "width" in the output doesn't increase the rank of the 2x2
    #   multiplication structure. Wait, no.
    #
    #   Actually, the tensor for 2x2 @ 2x5 has dimensions 4 x 10 x 10.
    #   Its rank is at most 7 * ceil(5/2)? No, that's not right either.
    #
    #   Let me think about it as: 2x2 @ 2x5 = (2x2 @ 2x2) + (2x2 @ 2x3)
    #   Rank(2x2@2x2) = 7
    #   Rank(2x2@2x3) <= Rank(2x2@2x2) + Rank(2x2@2x1) = 7 + 2 = 9
    #   So Rank(2x2@2x5) <= 16
    #
    #   Total: 2 * 16 = 32.
    #
    # But the known answer is 36. So maybe my split is suboptimal, or the
    # decomposition works differently.
    #
    # Alternative: Don't split into 2x2 @ 2x5 blocks. Instead:
    #   A (2x4) @ B (4x5) = C (2x5)
    #   Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    #   Split B into B1 (2x5, rows 0-1) and B2 (2x5, rows 2-3)
    #   C = A1 @ B1 + A2 @ B2
    #
    # Each A_i @ B_i is 2x2 @ 2x5.
    #
    # For 2x2 @ 2x5, another approach:
    #   A = [[a, b], [c, d]]
    #   B = [[b00, b01, b02, b03, b04],
    #        [b10, b11, b12, b13, b14]]
    #   C = [[a*b00+b*b10, a*b01+b*b11, a*b02+b*b12, a*b03+b*b13, a*b04+b*b14],
    #        [c*b00+d*b10, c*b01+d*b11, c*b02+d*b12, c*b03+d*b13, c*b04+d*b14]]
    #
    # This has 10 scalar products. Can we do better?
    #
    # Strassen for 2x2@2x2 uses 7 multiplications. For 2x2@2x5, we can
    # group the 5 columns. The key insight: the Strassen "multiplications"
    # involve sums of A entries times sums of B entries. Since B has 5 columns,
    # each Strassen multiplication produces a 1x5 vector (or 2x5 if we're
    # careful).
    #
    # Strassen's 7 products for 2x2@2x2:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)h
    #   M3 = a(f-h)
    #   M4 = d(g-e)
    #   M5 = (a+b)e
    #   M6 = (c-d)(h+g)
    #   M7 = (b-d)(e+f)
    #
    # For 2x2@2x5, let B have columns B0..B4 (each 2x1):
    #   B = [B0 | B1 | B2 | B3 | B4] where Bj = [bj0, bj1]^T
    #
    # We can apply Strassen's structure:
    #   M1 = (a+d) * (B0+B1+B2+B3+B4)  -- but this isn't quite right
    #
    # Actually, Strassen's algorithm is specifically for 2x2 @ 2x2. For 2x2 @ 2x5,
    # we need a different approach.
    #
    # Let me just use the straightforward bound:
    #   2x2 @ 2x5: split 2x5 into 2x2 + 2x2 + 2x1
    #   2x2 @ 2x2: rank 7 (Strassen)
    #   2x2 @ 2x1: rank 2
    #   Total: 7 + 7 + 2 = 16
    #
    # Two such: 32.
    #
    # Hmm, but 32 < 36. If rank 32 is achievable, that would be even better.
    # But maybe the tensor rank of 2x2@2x5 is actually higher than 16?
    #
    # The tensor for 2x2@2x5 has shape (4, 10, 10).
    # Lower bound via flattening: rank of (4, 100) flattening <= 4.
    # Rank of (10, 40) flattening <= 10.
    # So lower bound is 4. Upper bound is 16.
    #
    # Actually, I think the issue is that I'm conflating two things. Let me just
    # directly construct a rank-36 decomposition using a known structure.
    #
    # Known result: R(2,4,5) = 36.
    #
    # One construction:
    #   C = A @ B where A is 2x4, B is 4x5.
    #
    # Write A = [A1 | A2] where A1, A2 are 2x2.
    # Write B = [B1; B2] where B1, B2 are 2x5.
    #
    # C = A1 @ B1 + A2 @ B2.
    #
    # For each 2x2 @ 2x5 product, use a rank-18 decomposition:
    #   2x2 @ 2x5 can be decomposed with rank 18.
    #   How? 2x2 @ 2x5 = 2x2 @ (2x2 + 2x3)
    #   2x2 @ 2x2: rank 7
    #   2x2 @ 2x3: rank 11 (since 2x2@2x3 tensor has shape 4x6x6, and
    #     the rank is known to be 11? Or we can bound it as 7+2=9 or 7+4=11)
    #
    #   If 2x2@2x3 has rank 11, then 2x2@2x5 has rank 7+11=18.
    #   Two such: 36. That matches!
    #
    # So the decomposition is:
    #   2x2 @ 2x5 = (2x2 @ 2x2) + (2x2 @ 2x3)
    #   Rank: 7 + 11 = 18
    #   Total: 2 * 18 = 36
    #
    # Now I need to construct:
    #   1. Strassen rank-7 for 2x2 @ 2x2
    #   2. Rank-11 for 2x2 @ 2x3
    #
    # For 2x2 @ 2x3:
    #   A = [[a,b],[c,d]] (2x2), B = [[e,f,g],[h,i,j]] (2x3)
    #   C = [[ae+bh, af+bi, ag+bj],
    #        [ce+dh, cf+di, cg+dj]]
    #
    # This is 6 scalar products in schoolbook. Can we do 11? No, 11 > 6.
    # Wait, that can't be right. The rank should be at most 6 for 2x2@2x3
    # since there are only 6 scalar products.
    #
    # Oh wait, I'm confusing myself. The "rank" here is the tensor rank, which
    # is the minimum number of rank-1 tensors. For 2x2@2x3, the tensor has
    # shape (4, 6, 6). The schoolbook uses 6 rank-1 tensors (one per scalar
    # product). So rank <= 6.
    #
    # So 2x2@2x5 = 2x2@(2x2+2x3) has rank <= 7 + 6 = 13.
    # Two such: 26. That's way less than 36.
    #
    # I think I'm making an error somewhere. Let me reconsider.
    #
    # The tensor for matrix multiplication m x n x p has rank R(m,n,p).
    # R(2,2,2) = 7 (Strassen).
    # R(2,2,3) <= ? It's at most 6 (schoolbook) and at least... 
    # Actually R(2,2,3) is known to be 7? No, that doesn't make sense
    # since schoolbook gives 6.
    #
    # Wait, I think I've been confusing the dimensions. Let me be very careful.
    #
    # For multiplying an (m x n) matrix by an (n x p) matrix:
    #   A is m x n, B is n x p, C is m x p.
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C.
    # Shape: (m*n, n*p, m*p).
    #
    # For 2x2x2: shape (4, 4, 4). Rank 7.
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3. Shape (4, 6, 6).
    #   Schoolbook: 4*3 = 12 scalar products? No.
    #   C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    #   Number of terms: 2 * 2 * 3 = 12? No, it's sum over j (2 values) for
    #   each (i,k) pair (2*3=6 pairs). So 12 scalar products total? 
    #   No: for each (i,k), we do 2 multiplications. Total: 6 * 2 = 12.
    #   But the tensor rank counts rank-1 tensors, not scalar products.
    #   Each scalar product A[i,j]*B[j,k] contributes to C[i,k].
    #   The rank-1 tensor for term (i,j,k) has A-factor with 1 at (i,j),
    #   B-factor with 1 at (j,k), C-factor with 1 at (i,k).
    #   So schoolbook rank = m*n*p = 2*2*3 = 12.
    #
    # Ah, I see my confusion. The schoolbook rank for m x n x p is m*n*p,
    # not m*p (which is the number of output elements).
    #
    # So:
    #   R(2,2,2) schoolbook = 8, actual = 7.
    #   R(2,2,3) schoolbook = 12, actual = ?
    #   R(2,4,5) schoolbook = 40, actual = 36 (known).
    #
    # For 2x2x3: shape (4, 6, 6).
    #   Lower bound: max flattening rank.
    #   Flattening (4, 36): rank <= 4.
    #   Flattening (6, 24): rank <= 6.
    #   Flattening (6, 24): rank <= 6.
    #   So lower bound is 4. Upper bound is 12.
    #
    # Known: R(2,2,3) = 7? Let me check.
    #   Actually, I recall that R(2,2,3) might be 7 or 8.
    #
    # Regardless, for our purpose, let's just directly construct the rank-36
    # decomposition for 2x4x5.
    #
    # Here's a cleaner approach:
    #
    # C = A @ B, A is 2x4, B is 4x5.
    #
    # Split: A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x5).
    # C = A1 @ B1 + A2 @ B2.
    #
    # For A1 @ B1 (2x2 @ 2x5):
    #   A1 = [[a11, a12], [a21, a22]]
    #   B1 = [[b11, b12, b13, b14, b15],
    #         [b21, b22, b23, b24, b25]]
    #   C1 = A1 @ B1 (2x5)
    #
    # We can compute this using Strassen-like 7 "multiplications" where each
    # multiplication is a scalar times a 1x5 vector (or we can think of it as
    # 7 rank-1 contributions to the 2x5 output).
    #
    # Actually, the cleanest way: use the fact that 2x2 @ 2x5 can be done
    # with rank 7 by applying Strassen's 7 products, where each "product"
    # is a linear combination of A entries times a linear combination of
    # B columns.
    #
    # Strassen for 2x2 @ 2x2:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)h  
    #   M3 = a(f-h)
    #   M4 = d(g-e)
    #   M5 = (a+b)e
    #   M6 = (c-d)(h+g)
    #   M7 = (b-d)(e+f)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # For 2x2 @ 2x5, let B = [B_1 | B_2 | B_3 | B_4 | B_5] where each B_k is 2x1.
    # We can write B as a 2x2 + 2x2 + 2x1, or we can directly extend Strassen.
    #
    # Actually, the simplest correct approach: just use 7 rank-1 terms for
    # the 2x2@2x2 part and 6 for the 2x2@2x3 part (schoolbook for that part).
    # But 7+6=13 per 2x2@2x5, giving 26 total. That's < 36.
    #
    # I think the issue is that I don't actually know the exact rank-36
    # construction off the top of my head. Let me try a different strategy:
    #
    # Use a randomized search to find a rank-36 decomposition, starting from
    # the schoolbook rank-40 and trying to eliminate 4 terms.
    #
    # Actually, let me try yet another approach: directly solve for the
    # factors using the known structure.
    #
    # The multiplication tensor for 2x4x5:
    #   T[a, b, c] = 1 if (i,j) from a, (j',k) from b, (k',i') from c
    #   satisfy i=i', j=j', k=k'.
    #
    # a = i*4 + j, i in {0,1}, j in {0,1,2,3}
    # b = j'*5 + k, j' in {0,1,2,3}, k in {0,1,2,3,4}
    # c = k'*2 + i', k' in {0,1,2,3,4}, i' in {0,1}
    #
    # T[a,b,c] = 1 iff i=i', j=j', k=k'.
    #
    # There are 40 such (a,b,c) triples (one for each i,j,k combination).
    #
    # For a rank-R decomposition: T = sum_{r=1}^{R} u_r v_r w_r^T
    # where u_r in R^8, v_r in R^20, w_r in R^10.
    #
    # Each term u_r v_r w_r^T is a rank-1 tensor.
    #
    # The schoolbook has R=40, with each term being a single (i,j,k) triple.
    #
    # To get R=36, we need to "merge" 4 of the 40 terms into combinations
    # that cancel out.
    #
    # This is equivalent to finding a 4-dimensional subspace of the 40-dimensional
    # space of rank-1 tensors that sums to zero (or rather, that can be
    # re-expressed with 0 rank-1 tensors, i.e., the 40 terms can be replaced
    # by 36 terms).
    #
    # This is the same as finding a non-trivial relation among the 40 rank-1
    # tensors, which means finding a linear dependence in the space they span
    # that allows reducing the count.
    #
    # OK, I think the most reliable approach given time constraints is:
    # 1. Try to find a rank-36 decomposition via randomized least-squares /
    #    alternating minimization.
    # 2. If that fails, fall back to rank 40.
    #
    # Let me implement a randomized search.

    def verify(U, V, W, T, tol=1e-9):
        """Check if T = sum_r u_r v_r w_r^T."""
        R = U.shape[1]
        T_recon = np.zeros_like(T)
        for r in range(R):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T_recon - T)) < tol

    def try_random_search(T, dim_a, dim_b, dim_c, target_rank, n_trials=200, time_limit=120):
        """Try to find a rank-target_rank decomposition via alternating minimization."""
        start = time.time()
        best_err = float('inf')
        best_factors = None
        
        for trial in range(n_trials):
            if time.time() - start > time_limit:
                break
            
            R = target_rank
            # Initialize with random factors
            np.random.seed(trial)
            U = np.random.randn(dim_a, R) * 0.1
            V = np.random.randn(dim_b, R) * 0.1
            W = np.random.randn(dim_c, R) * 0.1
            
            # Alternating minimization
            for it in range(50):
                if time.time() - start > time_limit:
                    break
                
                # Update U: for each a, U[a,:] should satisfy
                #   T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
                #   => T[a,:,:].reshape(-1) = U[a,:] @ (V * W).reshape(R, -1)
                #   => U[a,:] = pinv((V*W).reshape(R,-1)) @ T[a,:,:].reshape(-1)
                VW = (V * W).T  # shape (R, dim_b * dim_c)
                # Actually: (V[:,r] outer W[:,r]).flatten() = (V[:,r] * W[:,r]) reshaped? No.
                # V[:,r] is dim_b, W[:,r] is dim_c.
                # V[:,r] outer W[:,r] has shape (dim_b, dim_c).
                # Flattened: shape (dim_b * dim_c,).
                # Element (b,c) = V[b,r] * W[c,r].
                # So the vector is V[:,r] * W[:,r] but that's element-wise, not outer.
                # We need: for each r, the vector is np.outer(V[:,r], W[:,r]).flatten()
                
                # Build matrix M where M[r, (b,c)] = V[b,r] * W[c,r]
                # Shape: (R, dim_b * dim_c)
                M = np.zeros((R, dim_b * dim_c))
                for r in range(R):
                    M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                
                # U[a, :] = M^+ @ T[a, :, :].flatten()
                try:
                    U_new = np.zeros((dim_a, R))
                    for a in range(dim_a):
                        t_vec = T[a, :, :].flatten()
                        U_new[a, :] = np.linalg.lstsq(M, t_vec, rcond=None)[0]
                    U = U_new
                except:
                    pass
                
                # Update V similarly
                MW = np.zeros((R, dim_a * dim_c))
                for r in range(R):
                    MW[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                try:
                    V_new = np.zeros((dim_b, R))
                    for b in range(dim_b):
                        t_vec = T[:, b, :].transpose(1, 0).flatten()  # T[a, b, c] -> T[:, b, :] shape (dim_a, dim_c)
                        V_new[b, :] = np.linalg.lstsq(MW, t_vec, rcond=None)[0]
                    V = V_new
                except:
                    pass
                
                # Update W
                MU = np.zeros((R, dim_a * dim_b))
                for r in range(R):
                    MU[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                try:
                    W_new = np.zeros((dim_c, R))
                    for c in range(dim_c):
                        t_vec = T[:, :, c].transpose(0, 1).flatten()  # T[a, b, c] -> T[:, :, c] shape (dim_a, dim_b)
                        W_new[c, :] = np.linalg.lstsq(MU, t_vec, rcond=None)[0]
                    W = W_new
                except:
                    pass
                
                # Check error
                T_recon = np.zeros_like(T)
                for r in range(R):
                    T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.max(np.abs(T_recon - T))
                
                if err < best_err:
                    best_err = err
                    best_factors = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-9:
                    return U, V, W, err
        
        return best_factors[0], best_factors[1], best_factors[2], best_err

    # Try to find rank 36
    found = False
    if time.time() < deadline - 30:
        U36, V36, W36, err36 = try_random_search(
            T, dim_a, dim_b, dim_c, 36, 
            n_trials=50, 
            time_limit=min(100, deadline - time.time() - 20)
        )
        if err36 < 1e-9:
            found = True
            best_U, best_V, best_W, best_R = U36, V36, W36, 36
        else:
            # Fall back to rank 40
            best_R = 40
            best_U = np.zeros((dim_a, 40))
            best_V = np.zeros((dim_b, 40))
            best_W = np.zeros((dim_c, 40))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        best_U[a, idx] = 1.0
                        best_V[b, idx] = 1.0
                        best_W[c, idx] = 1.0
                        idx += 1
    else:
        # No time for search, use rank 40
        best_R = 40
        best_U = np.zeros((dim_a, 40))
        best_V = np.zeros((dim_b, 40))
        best_W = np.zeros((dim_c, 40))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    best_U[a, idx] = 1.0
                    best_V[b, idx] = 1.0
                    best_W[c, idx] = 1.0
                    idx += 1

    # Final verification
    if not verify(best_U, best_V, best_W, T):
        # Emergency fallback: definitely return rank 40
        best_R = 40
        best_U = np.zeros((dim_a, 40))
        best_V = np.zeros((dim_b, 40))
        best_W = np.zeros((dim_c, 40))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    best_U[a, idx] = 1.0
                    best_V[b, idx] = 1.0
                    best_W[c, idx] = 1.0
                    idx += 1

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
