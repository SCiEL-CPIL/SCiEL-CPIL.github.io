import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        # Check: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(decomp):
        """Vectorized verification using matrix operations."""
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Reshape for efficient computation
        # For each c, the contribution is outer product of U[:,c], V[:,c], W[:,c]
        # We need T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
            # This is a 3-way outer product contraction
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    # === STRATEGY 1: Block-circulant Strassen ===
    # Model: A is 2x2, B is 2x16, C = A*B is 2x16
    # Split B into 8 blocks of 2 columns: B = [B_0, B_1, ..., B_7] where B_g is 2x2
    # C = [C_0, C_1, ..., C_7] where C_g = A * B_g
    # Each C_g = A * B_g is 2x2 matrix multiplication (rank 7 via Strassen)
    # Total rank = 8 * 7 = 56
    #
    # Key insight: We can potentially share structure across blocks if the
    # W factors have periodic structure. But since each C_g depends on
    # different columns of B, the V factors must be block-diagonal.
    # However, the U factors (for A) are the same across all blocks!
    # So we can factor out the A-dependence.
    #
    # Actually, the tensor decomposition requires:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For the block structure, k = 2*g + l where g is block index, l in {0,1}
    # W[2*g+l, i] depends on g and l and i
    # V[j, 2*g+l] depends on j, g, l
    # U[i*m+j] depends on i, j
    #
    # The Strassen structure gives us 7 multipliers per block.
    # Since A is the same for all blocks, the U coefficients repeat.
    # This means we can potentially reduce rank by sharing U columns.
    #
    # Let's try: use 7 columns total (one set of Strassen multipliers)
    # and let V and W handle the block indexing.
    #
    # For this to work, we need:
    # T[i,j,2g+l] = sum_{s=0}^{6} U[i*m+j,s] * V[j,2g+l,s] * W[2g+l,i,s]
    #
    # From Strassen for block g:
    # C_g = A * B_g
    # The (i,l) entry of C_g is sum_j A[i,j]*B_g[j,l]
    # In Strassen form: C_g[i,l] = sum_s u_s[i] * v_s[l] * (A*B_g terms)
    #
    # Actually, the standard Strassen decomposition for 2x2*2x2 is:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1-M5+M4+M7
    # c01 = M5+M3
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    #
    # For block g, B_g has entries B_g[j,l] = B[j, 2g+l]
    # So M1_g = (A00+A11)(B[0,2g]+B[1,2g+1])
    #
    # In the tensor decomposition:
    # T[i,j,2g+l] = sum_s U[i*m+j,s] * V[j,2g+l,s] * W[2g+l,i,s]
    #
    # For the Strassen multiplier s=0 (M1):
    # Contribution to C_g[i,l] is:
    # c00 += 1, c01 += 1, c10 += 1, c11 += 1
    # So W[2g,0,0]=1, W[2g+1,0,0]=1, W[2g,1,0]=1, W[2g+1,1,0]=1
    #
    # The multiplier value is (A00+A11)*(B[0,2g]+B[1,2g+1])
    # = sum_{j1,j2} (delta_{j1,0}+delta_{j1,1})*(delta_{j2,0}+delta_{j2,1}) * A[j1,?]*B[j2,?]
    #
    # This is getting complex. Let me just use the straightforward 56-rank solution
    # and then try to optimize.

    def strassen_56():
        """Standard 8-block Strassen: rank 56."""
        num_blocks = p // 2  # 8
        rank = num_blocks * 7  # 56
        U = [[0] * rank for _ in range(n * m)]  # 4 x 56
        V = [[0] * rank for _ in range(m * p)]  # 32 x 56
        W = [[0] * rank for _ in range(p * n)]  # 32 x 56

        for block in range(num_blocks):
            k0 = block * 2
            k1 = k0 + 1
            base = block * 7

            # M1 = (A00+A11)(B0k0+B1k1)
            U[0][base + 0] = 1
            U[3][base + 0] = 1
            V[0 * 16 + k0][base + 0] = 1
            V[1 * 16 + k1][base + 0] = 1
            W[k0 * 2 + 0][base + 0] = 1
            W[k1 * 2 + 0][base + 0] = 1
            W[k0 * 2 + 1][base + 0] = 1
            W[k1 * 2 + 1][base + 0] = 1

            # M2 = (A10+A11)B0k0
            U[2][base + 1] = 1
            U[3][base + 1] = 1
            V[0 * 16 + k0][base + 1] = 1
            W[k0 * 2 + 1][base + 1] = 1

            # M3 = A00(B1k0-B1k1)
            U[0][base + 2] = 1
            V[1 * 16 + k0][base + 2] = 1
            V[1 * 16 + k1][base + 2] = -1
            W[k1 * 2 + 0][base + 2] = 1

            # M4 = A11(B0k1-B0k0)
            U[3][base + 3] = 1
            V[0 * 16 + k1][base + 3] = 1
            V[0 * 16 + k0][base + 3] = -1
            W[k0 * 2 + 0][base + 3] = 1
            W[k0 * 2 + 1][base + 3] = 1

            # M5 = (A00+A01)B1k1
            U[0][base + 4] = 1
            U[1][base + 4] = 1
            V[1 * 16 + k1][base + 4] = 1
            W[k0 * 2 + 0][base + 4] = -1
            W[k1 * 2 + 0][base + 4] = 1

            # M6 = (A10-A00)(B0k0+B0k1)
            U[2][base + 5] = 1
            U[0][base + 5] = -1
            V[0 * 16 + k0][base + 5] = 1
            V[0 * 16 + k1][base + 5] = 1
            W[k1 * 2 + 1][base + 5] = 1

            # M7 = (A01-A11)(B1k0+B1k1)
            U[1][base + 6] = 1
            U[3][base + 6] = -1
            V[1 * 16 + k0][base + 6] = 1
            V[1 * 16 + k1][base + 6] = 1
            W[k0 * 2 + 0][base + 6] = 1

        return {"U": U, "V": V, "W": W}

    # === STRATEGY 2: Shared-U Strassen (rank 7 * 8 = 56, but try to reduce) ===
    # The key observation: U depends only on (i,j), not on block g.
    # In the 56-rank solution, U has 8 copies of the same 7-column pattern.
    # We can't simply share them because V and W are block-specific.
    #
    # However, we can try a different grouping: use 4 blocks of 4 columns.
    # For 2x2 * 2x4, the schoolbook rank is 8. Can we do better?
    # The tensor T[i,j,k] for k in {0,1,2,3} with fixed (i,j) pattern...
    # This is equivalent to computing 2x2 * 2x4, which is 4 independent
    # 2x2 multiplications (one for each column pair), so rank is at least 7*2=14?
    # No, it's 2x2 * 2x4, which has 2*2*4 = 16 entries in C.
    # Schoolbook: 8 multiplications. Can we do better?
    # Actually, 2x2 * 2x4 = 2x(2*4) = 2x8 output.
    # Each output row is a linear combination of B's rows.
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]  (2*4 = 8 entries)
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]  (2*4 = 8 entries)
    # This is just 4 scalar multiplications per row, 8 total.
    # The rank of this tensor is 8 (each entry is a product of one A and one B).
    # So 4 blocks of 4 columns: rank = 4*8 = 32? No wait.
    #
    # Actually for 2x2 * 2x4:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2,3}
    # The tensor T[i,j,k] = delta (it's the identity tensor for this multiplication)
    # Rank of 2x2*2x4 multiplication tensor:
    # It's the Kronecker product of (2x2*2x1) with... no.
    # T is in R^{2} x R^{2} x R^{4} (output x A x B)
    # Actually T[i,j,k] where i is output row, j is A col, k is B col
    # T[i,j,k] = 1 if this is a valid multiplication term
    # The rank of the 2x2*2x4 tensor is 8 (it's just 8 independent products)
    # because each output C[i,k] involves exactly 2 products, and they don't share.
    #
    # So 4 groups of 4 columns: rank = 4 * 8 = 32. But that's worse than 56? No, 32 < 56.
    # Wait, schoolbook for 2x2*2x16 is 64. 4 groups of 4 columns with rank 8 each = 32.
    # But that's just schoolbook regrouped! The rank is still 64 total.
    # 4 * 8 = 32... no, 4 groups * 8 rank each = 32 total rank.
    # But schoolbook is 64. So this would be an improvement?
    # No! Each group of 4 columns requires 8 rank-1 terms, so 4*8=32 total.
    # But wait, the full schoolbook is 2*2*16 = 64. How can 4*8=32 work?
    #
    # I think I'm confusing myself. Let me reconsider.
    # For 2x2 * 2x4: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    # Number of (i,j,k) triples: 2*2*4 = 16
    # Schoolbook: for each (i,j,k), one term: A[i,j]*B[j,k]
    # That's 16 terms, but many can be combined.
    # Actually no: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=0,1,2,3
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] for k=0,1,2,3
    # Total: 8 products. Each product is A[i,j]*B[j,k] for specific (i,j,k).
    # These are 8 independent products, so rank is 8.
    #
    # For 2x2*2x16: 2*2*16 = 64 products. Rank is 64 (schoolbook).
    # 4 groups of 4 columns: 4 * 8 = 32 rank. But this is still schoolbook!
    # The rank of the full tensor is 64, not 32.
    #
    # I think the issue is that when we split into groups, the rank of each
    # sub-tensor is 8, and since they're independent (different k ranges),
    # the total rank is the sum: 4*8 = 32. But the full tensor has rank 64.
    # This is a contradiction, so I must be wrong.
    #
    # The resolution: the tensor T for 2x2*2x16 has entries T[i,j,k] for
    # i in {0,1}, j in {0,1}, k in {0,...,15}. The rank is the minimum r
    # such that T[i,j,k] = sum_{c=1}^{r} U[i,j,c]*V[j,k,c]*W[k,i,c].
    #
    # For the sub-tensor with k in {0,1,2,3} (first group):
    # T[i,j,k] for k in {0,1,2,3} has rank 8.
    # For the sub-tensor with k in {4,5,6,7} (second group):
    # T[i,j,k] for k in {4,5,6,7} also has rank 8.
    # Since the k-ranges are disjoint, the total rank is 8+8+8+8 = 32.
    #
    # But wait, the full schoolbook has rank 64 = 2*2*16.
    # How can the rank be 32?
    #
    # OH! I see the error. For 2x2*2x4, the tensor is:
    # T[i,j,k] = 1 if this term appears in the multiplication
    # The multiplication is C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), and the "value" is 1.
    # The number of nonzero entries is 2*2*4 = 16.
    # The rank of this tensor...
    #
    # Actually, I think the rank of the 2x2*2x4 multiplication tensor is 8.
    # Because we can write:
    # T[i,j,k] = sum_{c=1}^{8} U[i,j,c]*V[j,k,c]*W[k,i,c]
    # where c indexes the 8 (i',j') pairs... no.
    #
    # Let me think differently. The tensor T is in U^* x V^* x W^* where
    # dim U = 4 (A entries), dim V = 8 (B entries), dim W = 8 (C entries).
    # T is the "multiplication tensor" which has rank equal to the bilinear
    # complexity of 2x2*2x4 matrix multiplication.
    #
    # For 2x2*2x4: we compute C = A*B where A is 2x2, B is 2x4, C is 2x4.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # We can use Strassen on the "block" structure:
    # Treat B as two 2x2 blocks: B = [B_left, B_right]
    # C = [C_left, C_right] where C_left = A*B_left, C_right = A*B_right
    # Each is 2x2*2x2, rank 7. Total rank = 14.
    #
    # So 2x2*2x4 has rank 14, not 8!
    # And 2x2*2x16 = 8 blocks of 2x2*2x2, each rank 7, total rank 56.
    #
    # OK so my original 56-rank solution is correct. Let me verify it works.

    # Try the 56-rank Strassen solution
    try:
        decomp = strassen_56()
        if verify(decomp):
            return decomp
    except Exception:
        pass

    # === STRATEGY 3: Try to find a lower-rank solution ===
    # The theoretical minimum rank for 2x2*2x16:
    # Lower bound: at least max(2*2, 2*16, 2*16) = 32? No.
    # The rank of the multiplication tensor for m x n * n x p is at least
    # max(m*n, n*p, m*p) / something...
    # Actually, a simple lower bound is: rank >= max(m*n, n*p, m*p) is wrong.
    # The correct lower bound from the "slice rank" or other methods...
    #
    # For 2x2*2x16, the schoolbook rank is 64.
    # Strassen gives 56.
    # Can we do better?
    #
    # One approach: use the fact that A is only 2x2 (4 entries) and try to
    # share structure more aggressively.
    #
    # Another approach: use a different algorithm for 2x2*2x16.
    # For example, treat it as 2 independent 1x2*2x16 multiplications
    # (one for each row of C). Each 1x2*2x16 has rank 2*16 = 32 (schoolbook).
    # Total: 64. Same as schoolbook.
    #
    # Or: treat it as 2x2*2x16 = (2x2*2x2) repeated 8 times.
    # With Strassen: 8*7 = 56.
    #
    # Can we share the Strassen multipliers across blocks?
    # The issue is that B has different entries in each block, so the
    # multipliers M1_g, M2_g, etc. are different for each g.
    #
    # However, the U factors (coefficients of A) are the same for all blocks.
    # In the tensor decomposition, this means:
    # U[i,j,c] is the same for c corresponding to the same Strassen multiplier
    # across different blocks.
    #
    # In our 56-rank solution, U has 8 identical 7-column blocks.
    # We can't reduce this because V and W are block-specific.
    #
    # Let me try a different approach: use a single set of 7 multipliers
    # and encode the block index in V and W.
    #
    # For this to work, we need:
    # T[i,j,2g+l] = sum_{s=0}^{6} U[i,j,s] * V[j,2g+l,s] * W[2g+l,i,s]
    #
    # From Strassen for block g:
    # C_g[i,l] = sum_s u_s[i] * (linear combo of B_g entries)
    # where u_s[i] are the Strassen U coefficients.
    #
    # The multiplier for Strassen term s in block g is:
    # M_s(g) = (linear combo of A entries) * (linear combo of B_g entries)
    #
    # In the tensor decomposition:
    # T[i,j,2g+l] = sum_s U[i,j,s] * V[j,2g+l,s] * W[2g+l,i,s]
    #
    # We need this to equal the indicator that (i,j,2g+l) is a valid term.
    #
    # This is complex. Let me just try the 56-rank solution and see if it verifies.

    # Fallback to schoolbook
    return schoolbook()
