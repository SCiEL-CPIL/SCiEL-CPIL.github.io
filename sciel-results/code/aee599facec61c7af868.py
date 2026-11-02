import time
import numpy as np

_DEADLINE = time.time() + 120.0
_TIMEOUT_MARGIN = 8.0

def _remaining():
    return _DEADLINE - _TIMEOUT_MARGIN - time.time()

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Build the Strassen rank-56 base
    rank0 = 56
    U0 = np.zeros((nm, rank0), dtype=float)
    V0 = np.zeros((mp, rank0), dtype=float)
    W0 = np.zeros((pn, rank0), dtype=float)

    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)
            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1
            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1
            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1

            U0[a00, base] = 1; U0[a11, base] = 1
            V0[b00, base] = 1; V0[b11, base] = 1
            W0[c00, base] = 1; W0[c11, base] = 1

            U0[a10, base+1] = 1; U0[a11, base+1] = 1
            V0[b00, base+1] = 1
            W0[c10, base+1] = 1; W0[c11, base+1] = -1

            U0[a00, base+2] = 1
            V0[b01, base+2] = 1; V0[b11, base+2] = -1
            W0[c01, base+2] = 1; W0[c11, base+2] = 1

            U0[a11, base+3] = 1
            V0[b10, base+3] = 1; V0[b00, base+3] = -1
            W0[c00, base+3] = 1; W0[c10, base+3] = 1

            U0[a00, base+4] = 1; U0[a01, base+4] = 1
            V0[b11, base+4] = 1
            W0[c00, base+4] = -1; W0[c01, base+4] = 1

            U0[a10, base+5] = 1; U0[a00, base+5] = -1
            V0[b00, base+5] = 1; V0[b01, base+5] = 1
            W0[c11, base+5] = 1

            U0[a01, base+6] = 1; U0[a11, base+6] = -1
            V0[b10, base+6] = 1; V0[b11, base+6] = 1
            W0[c00, base+6] = 1

    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Build list of all (i,j,k) triples where T=1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    triples = np.array(triples, dtype=int)  # shape (64, 3)

    def _verify(U, V, W):
        """Check sum_r U[a,r]*V[b,r]*W[c,r] == 1 for all nonzero triples."""
        rank = U.shape[1]
        # For each triple, compute the sum
        # Vectorized: for each triple (a,b,c), compute dot(U[a,:]*V[b,:], W[c,:])
        total_err = 0.0
        for a, b, c in triples:
            s = np.dot(U[a, :] * V[b, :], W[c, :])
            total_err += abs(s - 1.0)
        return total_err

    def _verify_zero_sample(U, V, W, sample_size=200):
        """Check that off-diagonal entries are ~0 (sampled)."""
        rank = U.shape[1]
        rng = np.random.RandomState(42)
        total_err = 0.0
        for _ in range(sample_size):
            a = rng.randint(nm)
            b = rng.randint(mp)
            c = rng.randint(pn)
            # Skip if this is a valid triple
            i = a // m; j = a % m
            j2 = b // p; k = b % p
            k2 = c // n; i2 = c % n
            if j == j2 and i == i2 and k == k2:
                continue
            s = np.dot(U[a, :] * V[b, :], W[c, :])
            total_err += abs(s)
        return total_err

    # Verify base
    base_err = _verify(U0, V0, W0)

    # Strategy: iterative rank reduction
    # For each rank term, try to remove it and compensate by adjusting others
    # This is a linear system: find deltas dU, dV, dW such that
    # the perturbation from removing term r is canceled
    
    # Simpler approach: try removing term r, then check if the residual
    # (the contribution that term r was providing) can be expressed as a
    # combination of the remaining terms.
    
    # The residual tensor from removing term r is:
    # R[a,b,c] = -U[a,r]*V[b,r]*W[c,r]  (for all a,b,c)
    # We need: sum_{r' != r} delta_r' * U[a,r']*V[b,r']*W[c,r'] = R[a,b,c]
    # This is generally nonlinear, but if we only adjust ONE factor at a time,
    # it becomes linear.
    
    # Approach 1: Try removing term r and adjusting W only
    # R[a,b,c] = -U[a,r]*V[b,r]*W[c,r]
    # We need sum_{r'!=r} delta_W[c,r'] * U[a,r']*V[b,r'] = -U[a,r]*V[b,r]*W[c,r]
    # For fixed c: sum_{r'!=r} delta_W[c,r'] * (U[:,r'] * V[:,r']^T)[a,b] = -W[c,r] * (U[:,r]*V[:,r]^T)[a,b]
    # This is a linear system in delta_W[c,r'] for each c.
    
    # Approach 2: Try removing term r and adjusting U only
    # R[a,b,c] = -U[a,r]*V[b,r]*W[c,r]
    # sum_{r'!=r} delta_U[a,r'] * V[b,r']*W[c,r'] = -U[a,r]*V[b,r]*W[c,r]
    # For fixed a: sum_{r'!=r} delta_U[a,r'] * (V[:,r']*W[:,r']^T)[b,c] = -U[a,r]*(V[:,r]*W[:,r']^T)... 
    
    # Let me think differently. The simplest robust approach:
    # 1. Try removing each term and check if it's already redundant (err=0)
    # 2. If not, try to find a linear combination that absorbs it
    
    # For now, do the simple check first, then try compensation
    
    current_rank = best_rank
    U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
    
    # Iterative removal with compensation
    # For each term to remove, solve a least-squares problem to adjust remaining terms
    # We'll adjust all three matrices simultaneously using a linearized approach
    
    # Actually, the cleanest approach: 
    # When removing term r, the error at each nonzero triple (a,b,c) is:
    # err[a,b,c] = -U[a,r]*V[b,r]*W[c,r]
    # We want to find adjustments to U, V, W (excluding column r) such that
    # the new sum at each triple is still 1.
    
    # Linearized: if we add small deltas dU, dV, dW to the remaining terms,
    # the change in the product sum at (a,b,c) is approximately:
    # sum_{r'!=r} [dU[a,r']*V[b,r']*W[c,r'] + U[a,r']*dV[b,r']*W[c,r'] + U[a,r']*V[b,r']*dW[c,r']]
    # = err[a,b,c]
    
    # This is a big linear system. Let's set it up.
    # Variables: dU (nm x (rank-1)), dV (mp x (rank-1)), dW (pn x (rank-1))
    # But this is too many variables. Let's use a different approach.
    
    # Better: since the error is a rank-1 tensor (outer product of three vectors),
    # we can try to express it as a sum of rank-1 tensors from the remaining terms.
    
    # For the Strassen decomposition, each term has a specific structure.
    # Let's just do the simple redundancy check first, then try pair removals.
    
    # Simple redundancy check: try removing each term
    removed_any = True
    while removed_any:
        removed_any = False
        for r in range(current_rank):
            if _remaining() < 10.0:
                break
            # Zero out column r
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, r] = 0
            V_test[:, r] = 0
            W_test[:, r] = 0
            err = _verify(U_test, V_test, W_test)
            if err < 1e-9:
                # Remove this column
                mask = np.ones(current_rank, dtype=bool)
                mask[r] = False
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                current_rank -= 1
                removed_any = True
                break  # Restart from beginning
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Now try to reduce further using a more sophisticated approach
    # Try removing pairs of terms and compensating
    
    # For pair removal (r1, r2):
    # Error = -(U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2])
    # This is a sum of two rank-1 tensors, which is generally not rank-1
    # but might be expressible using the remaining terms
    
    # Let's try a different strategy: use the structure of the problem
    # The 2x4x8 matmul can be viewed as:
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # 
    # We can group j into pairs: j = 2s + t, s in {0,1}, t in {0,1}
    # C[i,k] = sum_s sum_t A[i,2s+t] B[2s+t,k]
    #
    # For each s, we have a 2x2x8 subproblem (2 rows of A, 2 rows of B, 8 cols)
    # This can be done with Strassen-like methods
    
    # Alternative: think of it as 4 independent 2x1x8 inner products per row of C
    # But that's just schoolbook.
    
    # Let's try a direct low-rank construction:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # If we could factor the j-sum using fewer terms...
    # For m=4, the minimum rank of the inner product is 4 (over reals, for generic matrices)
    # But we have n=2 rows sharing the same j-index, so we might do better
    
    # Actually, let's try to construct a rank-48 or lower decomposition directly
    # by exploiting the shared structure between the two rows of A.
    
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # Using Strassen on the "2x4" A matrix treated as 2 blocks of 2x2:
    # This is what the current code does, giving 56.
    
    # Can we do better by treating it as one 2x4 x 4x8 problem?
    # The bilinear complexity of 2x4 x 4x8 matmul...
    # Lower bound: max(8, 32, 16) = 32
    # Upper bound: 56 (Strassen) or 64 (schoolbook)
    
    # Let me try to construct a rank-48 decomposition:
    # Idea: use 4-point "Fourier" transform on the j-index with real coefficients
    # For m=4, we can use:
    # F0 = A0 + A1 + A2 + A3
    # F1 = A0 - A1 + A2 - A3  (this is like a "Hadamard" step)
    # F2 = A0 + A1 - A2 - A3
    # F3 = A0 - A1 - A2 + A3
    # These are 4 linear combinations, but they don't directly give the inner product
    
    # The 4-point DFT uses complex numbers. Let's use a real decomposition.
    # Actually, for the inner product sum_j A_j B_j with j=0..3:
    # We can use:
    # S0 = (A0+A1)(B0+B1) + (A2+A3)(B2+B3)
    # S1 = (A0-A1)(B0-B1) + (A2-A3)(B2-B3)
    # S2 = (A0+A2)(B0+B2) + (A1+A3)(B1+B3)  -- no this doesn't work
    
    # The correct approach for 4-term inner product with rank < 4 is impossible over reals
    # (the rank of the 4-term inner product tensor is 4).
    # But we have TWO rows of A sharing the same B, so we can exploit that.
    
    # Let me try a different grouping. Instead of 8 independent 2x2x2 Strassen blocks,
    # let's try to share some multiplications.
    
    # Current: for each (a,b) block, we do 7 mults for the 2x2x2 part.
    # Total: 8 * 7 = 56.
    
    # Can we share between blocks? The blocks share B entries.
    # Block (a,b) uses B[2a, 2b], B[2a, 2b+1], B[2a+1, 2b], B[2a+1, 2b+1]
    # Block (a,b+1) uses B[2a, 2b+2], B[2a, 2b+3], B[2a+1, 2b+2], B[2a+1, 2b+3]
    # No overlap in B between different b values for the same a.
    # But block (a,b) and block (a+1,b) share B[2a+1, 2b] and B[2a+1, 2b+1].
    
    # So there IS sharing between a=0 and a=1 blocks!
    # Let's exploit this.
    
    # For a fixed b, we have:
    # C[0, 2b]   = A[0,2b]*B[2b,2b] + A[0,2b+1]*B[2b+1,2b]
    # C[0, 2b+1] = A[0,2b]*B[2b,2b+1] + A[0,2b+1]*B[2b+1,2b+1]
    # C[1, 2b]   = A[1,2b]*B[2b,2b] + A[1,2b+1]*B[2b+1,2b]
    # C[1, 2b+1] = A[1,2b]*B[2b,2b+1] + A[1,2b+1]*B[2b+1,2b+1]
    
    # This is a 2x2x2x2 tensor (4 inputs from A, 4 from B, 4 outputs).
    # Can we do this with fewer than 14 multiplications (2*7)?
    
    # Let's use Strassen on the 2x2 A and 2x2 B matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # This gives 7 multiplications for the 2x2x2 block, which is what we have.
    # The question is whether we can share between the a=0 and a=1 blocks.
    
    # For block (0,b): uses A[0,2b], A[0,2b+1], A[1,2b], A[1,2b+1] and B[2b,*], B[2b+1,*]
    # Wait, no. Let me re-examine.
    
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x8: B[j,k] for j in {0,1,2,3}, k in {0,...,7}
    # C is 2x8: C[i,k] for i in {0,1}, k in {0,...,7}
    
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    # For k = 2b: C[i,2b] = A[i,0]*B[0,2b] + A[i,1]*B[1,2b] + A[i,2]*B[2,2b] + A[i,3]*B[3,2b]
    # For k = 2b+1: C[i,2b+1] = A[i,0]*B[0,2b+1] + A[i,1]*B[1,2b+1] + A[i,2]*B[2,2b+1] + A[i,3]*B[3,2b+1]
    
    # So for each b, we have 4 output values (C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1])
    # computed from 8 A-entries and 8 B-entries (4 B-entries for col 2b, 4 for col 2b+1).
    
    # The 8 A-entries are: A[0,0..3] and A[1,0..3] -- same for all b!
    # The B-entries differ for each b.
    
    # So we can precompute linear combinations of A that are shared across all b.
    # This is the key to potentially reducing the rank!
    
    # Let's define:
    # For the j-sum, we can use any bilinear decomposition of the 4-term inner product.
    # The minimum rank for a 4-term inner product is 4 (over reals).
    # But since we have 2 rows of A, we can use a joint decomposition.
    
    # Let's try: express C[i,k] using 4 "shared" linear combinations of A:
    # L1[i] = A[i,0] + A[i,1] + A[i,2] + A[i,3]  (for i=0,1)
    # L2[i] = A[i,0] - A[i,1] + A[i,2] - A[i,3]
    # L3[i] = A[i,0] + A[i,1] - A[i,2] - A[i,3]
    # L4[i] = A[i,0] - A[i,1] - A[i,2] + A[i,3]
    
    # And corresponding B combinations:
    # For each k:
    # R1[k] = B[0,k] + B[1,k] + B[2,k] + B[3,k]
    # R2[k] = B[0,k] - B[1,k] + B[2,k] - B[3,k]
    # R3[k] = B[0,k] + B[1,k] - B[2,k] - B[3,k]
    # R4[k] = B[0,k] - B[1,k] - B[2,k] + B[3,k]
    
    # Then C[i,k] = (1/4)(L1[i]*R1[k] + L2[i]*R2[k] + L3[i]*R3[k] + L4[i]*R4[k])
    
    # This gives 4 multiplications per (i,k) pair, but since we have 2 i's and 8 k's,
    # that's 2*8*4 = 64 multiplications. That's worse than schoolbook!
    
    # The issue is that we're not sharing the L and R computations.
    # In a tensor decomposition, the "multiplications" are the rank terms.
    # Each rank term corresponds to one scalar multiplication.
    
    # Let me reconsider. The tensor decomposition rank is the number of scalar multiplications.
    # For C[i,k] = sum_j A[i,j] B[j,k], the schoolbook uses n*m*p = 2*4*8 = 64 multiplications.
    # Strassen gives 56.
    
    # To beat 56, we need a smarter decomposition.
    # Let's try to use the fact that n=2 is small.
    
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # Define:
    # D[j] = A[0,j] + A[1,j]  (shared)
    # E[j] = A[0,j] - A[1,j]  (shared)
    # Then:
    # C[0,k] + C[1,k] = sum_j D[j] B[j,k]
    # C[0,k] - C[1,k] = sum_j E[j] B[j,k]
    
    # So we need to compute two 4-term inner products:
    # S1[k] = sum_j D[j] B[j,k]  for k=0..7
    # S2[k] = sum_j E[j] B[j,k]  for k=0..7
    
    # Each is a 1x4 x 4x8 matmul, which has rank 4*8 = 32 (schoolbook) or can be done with
    # a 4-point transform.
    
    # Using the 4-term inner product decomposition (rank 4):
    # For S1[k]: 4 multiplications per k, total 4*8 = 32
    # For S2[k]: 4 multiplications per k, total 4*8 = 32
    # Total: 64. Same as schoolbook.
    
    # But we can share the B-transforms!
    # Define for each k:
    # R1[k] = B[0,k] + B[1,k] + B[2,k] + B[3,k]
    # R2[k] = B[0,k] - B[1,k] + B[2,k] - B[3,k]
    # R3[k] = B[0,k] + B[1,k] - B[2,k] - B[3,k]
    # R4[k] = B[0,k] - B[1,k] - B[2,k] + B[3,k]
    
    # And for D:
    # D1 = D[0] + D[1] + D[2] + D[3]
    # D2 = D[0] - D[1] + D[2] - D[3]
    # D3 = D[0] + D[1] - D[2] - D[3]
    # D4 = D[0] - D[1] - D[2] + D[3]
    
    # Then S1[k] = (1/4)(D1*R1[k] + D2*R2[k] + D3*R3[k] + D4*R4[k])
    # Similarly for S2[k] with E instead of D.
    
    # Total multiplications: 4 (for D) * 8 (for k) + 4 (for E) * 8 (for k) = 64.
    # Still 64. The issue is that the rank-4 decomposition of the 4-term inner product
    # doesn't help when the "vector" changes for each k.
    
    # Wait, I think I'm confusing myself. Let me go back to basics.
    
    # The tensor M_{2,4,8} has entries T[i,j,k] = 1 (it's the identity on the contraction index).
    # No wait, T[i,j,k] = 1 for all valid i,j,k? That can't be right.
    
    # Let me re-read the problem. The tensor decomposition is:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # where a indexes A (0..7), b indexes B (0..31), c indexes C (0..15)
    # T[a,b,c] = 1 if the indices are consistent (j from a matches j' from b, etc.)
    
    # From the code: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 iff j==j', i==i', k==k'
    # Since a determines (i,j), b determines (j',k), c determines (k',i'),
    # T=1 iff j==j' AND i==i' AND k==k'.
    
    # So the tensor is 1 on a "diagonal" of size n*m*p = 64, and 0 elsewhere.
    # The total number of nonzero entries is 64.
    
    # The rank of this tensor is what we're trying to minimize.
    # Lower bound: the tensor has a "slice" of rank at least m=4 (fixing i and k, varying j).
    # Actually, the tensor rank is at least the matrix rank of any unfolding.
    # The (nm x mp) unfolding has rank at most n*p = 16, but that's an upper bound on the unfolding rank, not the tensor rank.
    
    # The tensor rank is at least max over all unfoldings of the matrix rank.
    # (nm x mp) unfolding: T[a,b] = 1 if consistent, 0 otherwise.
    # This is a block diagonal matrix with n*p = 16 blocks, each of size m x m = 4x4.
    # Each block is the 4x4 identity. So the matrix rank is 16*4 = 64? No.
    # The unfolding has shape (nm x mp) = (8 x 32).
    # T[a,b] = 1 if j_from_a == j_from_b, 0 otherwise.
    # This is a block matrix where each block (i,j) x (j',k) is 1 if j==j', 0 otherwise.
    # For fixed i and k, the submatrix over (j, j') is the identity.
    # The full matrix has rank... let me think.
    # Row a = (i,j), column b = (j',k). T[a,b] = 1 if j==j'.
    # So T = I_m ⊗ (ones_{n x p}) ... no.
    # T[a,b] = delta_{j,j'} where a=(i,j), b=(j',k).
    # This is the matrix that maps (i,j) to (j',k) with delta_{j,j'}.
    # It's like a "partial identity" on the j-index.
    # The rank of this matrix is m * min(n, p) = 4 * 2 = 8? No.
    # Let me just compute it. The matrix is 8x32.
    # Row (i,j) has 1s in columns (j, k) for all k. So each row has p=8 ones.
    # Two rows (i,j) and (i',j') are linearly dependent iff j==j' (regardless of i,i').
    # So the row space is spanned by the m=4 distinct patterns (one for each j).
    # Rank = 4.
    
    # Similarly, the (mp x pn) unfolding has rank... 
    # T[b,c] = 1 if j_from_b == j_from_c AND k_from_b == k_from_c AND i_from_c == i_from_b... 
    # This is getting complicated. Let me just trust that the rank is between 32 and 56.
    
    # Given the time constraints, let me focus on a practical improvement:
    # Try to find a rank-48 or lower decomposition using a more clever search.
    
    # Actually, let me try a completely different approach:
    # Use the fact that the tensor has a product structure.
    # T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # This is a "cycle" tensor.
    
    # For n=2, m=4, p=8, we can try to use a Kronecker product structure.
    # 4 = 2*2, 8 = 2*4, so maybe there's a 2D structure.
    
    # Let me just try to improve the search: after removing redundant terms,
    # try to find new lower-rank decompositions by solving a linear system.
    
    # Final verification
    final_err = _verify(best_U, best_V, best_W)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
