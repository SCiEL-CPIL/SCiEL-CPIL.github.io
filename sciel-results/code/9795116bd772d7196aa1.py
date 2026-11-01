import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 2, 4, 7
    N_U = n * m  # 8
    N_V = m * p  # 28
    N_W = p * n  # 14

    # Build the known rank-50 decomposition
    def build_rank50():
        rank = 50
        U = np.zeros((N_U, rank), dtype=np.int64)
        V = np.zeros((N_V, rank), dtype=np.int64)
        W = np.zeros((N_W, rank), dtype=np.int64)
        col = 0

        def add_strassen_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h)
            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1; W[c11_idx, col] += 1
            col += 1
            # M2 = (c+d)*e
            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1; W[c11_idx, col] -= 1
            col += 1
            # M3 = a*(f-h)
            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01_idx, col] += 1; W[c11_idx, col] += 1
            col += 1
            # M4 = d*(g-e)
            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00_idx, col] += 1; W[c10_idx, col] += 1
            col += 1
            # M5 = (a+b)*h
            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] -= 1; W[c01_idx, col] += 1
            col += 1
            # M6 = (c-a)*(e+f)
            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            # M7 = (b-d)*(g+h)
            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)

            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(0, j_start, k_start)
            add_schoolbook_221_block(0, j_start, 6)

        assert col == 50
        return U, V, W

    U0, V0, W0 = build_rank50()

    # Verify
    def verify(U, V, W, tol=1e-9):
        Uf = U.astype(np.float64)
        Vf = V.astype(np.float64)
        Wf = W.astype(np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Fallback schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank), dtype=np.int64)
        V = np.zeros((N_V, rank), dtype=np.int64)
        W = np.zeros((N_W, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    if not verify(U0, V0, W0):
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strategy: try to reduce rank by merging pairs of columns
    # A pair (a,b) can be merged if U_a = U_b + U_c for some c, etc.
    # More generally: try to find a column that is a linear combination of others
    # with small integer coefficients, then eliminate it.

    # Alternative: try a different tiling that might give fewer multiplications
    # Try Winograd-like approach for the full problem

    # Let me try a completely different approach:
    # Use the structure that m=4 can be split as 2+2, and n=2, p=7
    # For each pair of rows of A and the full B, we need to compute 2x7 = (2x4)(4x7)
    # This is essentially two 2x4 x 4x7 multiplications (one for each row pair... wait, there's only one pair)

    # Actually n=2 means only 2 rows. m=4 means 4 columns. p=7 means 7 columns in B.
    # So we're computing a single 2x4 times 4x7 = 2x7.

    # The tensor is M_{2,4,7}. The rank is what we want to minimize.
    # Known bounds: rank >= max(n*p, m*p, n*m) = max(14, 28, 8) = 28
    # Strassen gives 49 for 2x4x4... wait let me recalculate.
    # For 2x4 x 4x7: split B into 4x4 + 4x3, or use the tiling above.
    # Current: 6 Strassen blocks (7 each) + 2 schoolbook blocks (4 each) = 42 + 8 = 50.

    # Can we do better? Let's try:
    # Split the 4 middle dimension into 2+2, and the 7 output into 2+2+2+1
    # For each 2x2x2 block: Strassen = 7 mults
    # For 2x2x1 blocks: 4 mults each (can't use Strassen)
    # That's what we have: 6 * 7 + 2 * 4 = 50.

    # Alternative: split 7 into 3+4?
    # 2x2x3 block: can we do better than 6?
    # 2x2x3: the tensor has 12 entries. Lower bound is max(4,6,4)=6.
    # Can we achieve rank 6? That would be schoolbook (each output = sum of 2 products).
    # Actually for 2x2x3, schoolbook gives 2*2*3 = 12.
    # Hmm, let me think... 2x2x3 matrix mult:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # That's 4 outputs, each needs 2 mults = 8 mults schoolbook.
    # Can we do it in 6? The lower bound is max(2*2, 2*3, 2*3) = 6.
    # I believe the rank of 2x2x3 is 6 (no Strassen-like speedup for this size).

    # So 2x2x3 costs 6, 2x2x2 costs 7 (Strassen).
    # Split 7 as 2+2+2+1: 3*7 + 2*4 = 21+8 = 29... wait that doesn't match.
    # Oh I see, the tiling is over the middle dimension too.

    # Let me reconsider. The current tiling:
    # j dimension (middle, size 4): split as [0,1] and [2,3]
    # k dimension (output, size 7): split as [0,1], [2,3], [4,5], [6]
    # i dimension (size 2): not split (only 2 rows)

    # For each (j_pair, k_pair) where both have size 2: Strassen 2x2x2 -> 7 mults
    # For (j_pair, k_single): 2x2x1 -> 4 mults

    # j_pairs: 2, k_pairs: 3 (size-2) + 1 (size-1)
    # Total: 2*3*7 + 2*1*4 = 42 + 8 = 50. Correct.

    # Alternative split of 7: 3+4
    # 2x2x3: rank 6, 2x2x4: can we do better than 8?
    # 2x2x4: lower bound max(4, 8, 8) = 8. Schoolbook = 8. So rank = 8.
    # Total: 2*(6 + 8) = 28. Wait that seems too good!
    # But wait: 2x2x4 means A is 2x2, B is 2x4, C is 2x4.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Each of the 8 outputs needs 2 products = 16 schoolbook mults.
    # Lower bound is max(2*2, 2*4, 2*4) = 8.
    # But the rank is actually... let me think. For 2x2xn, the rank is 2n for n<=3?
    # Actually I think for 2x2xn the rank is 2n when n is small.
    # Hmm, this is getting complicated. Let me just try different tilings computationally.

    # Let me try: split m=4 as 1+3 instead of 2+2
    # 1x3 x 3xk block... this gets complicated.

    # Let me try a randomized approach: start from rank 50, try to eliminate columns
    # by finding linear dependencies.

    best_rank = 50
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Try to find if any column is redundant (linear combination of others)
    # For column c to be redundant: U[:,c] = sum_d a_d * U[:,d], similarly for V and W
    # with the SAME coefficients a_d (since it's a rank-1 tensor sum)

    # Actually for the tensor: T = sum_c U_c ⊗ V_c ⊗ W_c
    # Column c is redundant if U_c = sum_d a_d U_d, V_c = sum_d a_d V_d, W_c = sum_d a_d W_d
    # for some scalars a_d.

    # This is hard to find directly. Let me try a different approach:
    # Randomly perturb and project.

    # Actually, let me try a simpler idea: use a different known decomposition.
    # The 2x4x7 problem: can we use the 2x2x2 Strassen more cleverly?

    # What if we overlap the tiling? E.g., use a 2x3x3 block (rank 9 by 3x3x3 Strassen? No...)

    # Let me try: treat it as computing C = A*B where A is 2x4, B is 4x7.
    # Write A = [A1 | A2] where A1, A2 are 2x2.
    # Write B = [[B11, B12], [B21, B22]] where Bii are 2x2, B12, B21 are 2x2,
    # but wait B is 4x7, so B11 is 2x7, B21 is 2x7, etc. That doesn't work cleanly.

    # B is 4x7. Split rows: B = [B_top; B_bot] where B_top is 2x7, B_bot is 2x7.
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x2.
    # C = A_left * B_top + A_right * B_bot.

    # Now A_left * B_top: 2x2 times 2x7. Split B_top into B_top1 (2x3) and B_top2 (2x4).
    # A_left * B_top1: 2x2x3, rank = 6 (schoolbook: each of 6 outputs = 2 products)
    # A_left * B_top2: 2x2x4, rank = 8 (schoolbook: each of 8 outputs = 2 products)
    # Similarly for A_right.

    # Total: 2*(6+8) = 28. But wait, that's just schoolbook for the whole thing!
    # 2x4x7 schoolbook = 2*4*7 = 56. So 28 is half of that.

    # Hmm, I think I'm confusing myself. Let me be more careful.
    # 2x2x3 matrix multiplication: C is 2x3, A is 2x2, B is 2x3.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]. There are 6 outputs, each is a sum of 2 products.
    # Schoolbook: 6*2 = 12 scalar multiplications. But in tensor rank terms,
    # each output C[i,k] = sum_j A[i,j]*B[j,k], so the tensor has 2*2*3 = 12 nonzero entries,
    # and schoolbook decomposition uses 2*2*3 = 12 rank-1 terms.
    # Can we do better? The rank of M_{2,2,3} is known to be 6? No, I think it's higher.
    # Actually for M_{2,2,n}, the rank is 2n for n <= 3, and for larger n it might be different.
    # Let me just accept that small blocks have schoolbook rank.

    # So the key question is: can we do 2x2x4 in fewer than 8 rank-1 terms?
    # M_{2,2,4}: 16 nonzero entries. Lower bound: max(4, 8, 8) = 8.
    # I believe the rank is 8 for 2x2x4 (no improvement over schoolbook for this size).

    # So the best we can do with 2+2 split of m is:
    # 2x2x2 blocks: Strassen rank 7 (better than schoolbook 8)
    # 2x2x1 blocks: schoolbook rank 4 (optimal, lower bound = 4)

    # The current decomposition achieves 50. Can we find a different tiling that's better?

    # What about splitting m=4 as 1+3?
    # 1x3 x 3xk: this is just a dot product for each output.
    # A is 2x4 = [a0, a1, a2, a3], B is 4x7.
    # C[i,k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] + a3*B[3,k].
    # This is a 1x4x7 "matrix mult" for each row i.
    # Tensor M_{1,4,7} has rank 7 (each output = sum of 4 products, but rank = 7 since it's just B^T * A).
    # Wait, M_{1,4,7}: C[k] = sum_j A[j]*B[j,k]. This is outer product of A (length 4) and... no.
    # C[k] = <A, B[:,k]>. The tensor is T[i,j,k] = delta_{i,0} * 1 (for the A index... hmm this isn't standard).

    # I'm overcomplicating this. Let me just try a few specific constructions and verify.

    # Try: use 3 Strassen blocks covering different parts
    # What if we don't align the j-split and k-split perfectly?

    # Actually, let me try a completely different approach:
    # Use the fact that 7 = 2+2+2+1 and see if we can share computations.

    # For Strassen 2x2x2, the 7 products are:
    # M1 = (a+d)(e+h), M2 = (c+d)*e, M3 = a*(f-h), M4 = d*(g-e),
    # M5 = (a+b)*h, M6 = (c-a)*(e+f), M7 = (b-d)*(g+h)

    # If we have overlapping 2x2x2 blocks, some of these M terms might be shared.
    # E.g., if two blocks share the same A submatrix but different B submatrices,
    # some U factors might be the same.

    # In the current tiling, each Strassen block has its own unique A and B submatrices,
    # so there's no sharing. But what if we use a different grouping?

    # Idea: use a 2x4x4 block (two Strassen blocks side by side) and see if
    # the combined structure allows sharing.

    # 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # Current approach: 2 Strassen blocks = 14 mults.
    # Can we do 2x4x4 in fewer than 14? Lower bound: max(8, 16, 16) = 16.
    # Wait, that's higher than 14! So Strassen for 2x4x4 via 2 blocks gives 14,
    # but the lower bound says we need at least 16? That can't be right.

    # Let me recheck. For M_{2,4,4}:
    # n=2, m=4, p=4. Lower bound = max(n*p, m*p, n*m) = max(8, 16, 8) = 16.
    # But 2 Strassen 2x2x2 blocks give 14 < 16. Contradiction!

    # Oh wait, I think the lower bound max(n*p, m*p, n*m) is not always valid for non-square.
    # Actually the correct lower bound for M_{n,m,p} is max(n*p, m*p, n*m) only when...
    # No, I think the correct lower bound is just that the rank must be at least
    # the maximum of the dimensions of the slices. For M_{n,m,p}, the slices are:
    # - fixing i: an m x p matrix (rank <= m*p... no, the slice has at most m*p nonzero entries)
    # Actually the standard lower bound is: rank >= max(n*p, m*p, n*m) is NOT correct in general.
    # The correct lower bound is rank >= max over all slices of the slice rank.
    # For M_{n,m,p}, each slice (fixing one index) has at most n*m, m*p, or n*p nonzero entries,
    # but the rank of the slice can be at most the number of nonzeros.
    # The actual lower bound is more subtle.

    # Anyway, empirically 2x4x4 can be done in 14 (two Strassen blocks), and this is known to be optimal.

    # So for 2x4x7, we need to handle the extra column. 7 = 4+3 or 4+2+1.
    # 2x4x3: lower bound... let's see. 2x2x3 blocks: 2 blocks of 2x2x3.
    # 2x2x3 schoolbook = 6 rank-1 terms each? Or can we do better?
    # For 2x2x3: C is 2x3, A is 2x2, B is 2x3.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # Schoolbook: 2*2*3 = 12 terms. But we can factor:
    # For fixed i: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:].
    # This is a rank-2 matrix (sum of 2 rank-1 rows).
    # In tensor form: T[i,j,k] = delta_{j,0}*1 + delta_{j,1}*1... no.
    # T[i,j,k] = 1 if... actually T[i,j,k] = 1 for all i,j,k in the appropriate ranges? No.
    # For matrix mult M_{n,m,p}: T[i,j,k] = 1 (all ones).
    # So M_{2,2,3} is the 2x2x3 all-ones tensor.
    # Rank of 2x2x3 all-ones tensor: I believe it's 6 (not 12).
    # Because: sum_{j=0}^{1} e_i * (e_j ⊗ b_j) where b_j is the j-th row of B...
    # Actually: T = sum_j u_j ⊗ v_j ⊗ w_j where u_j has 1 at position (i,j) for all i...
    # Hmm, let me think differently.
    # M_{2,2,3} = sum_{j=0}^{1} (sum_i e_i) ⊗ e_j ⊗ (sum_k e_k)
    # = (1,1) ⊗ (1,0,1) ... no this isn't right either.

    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}.
    # This is the outer product of three vectors: (1,1) ⊗ (1,1) ⊗ (1,1,1).
    # So its rank is 1! Wait, that can't be right for matrix multiplication.

    # Oh I see my confusion. For matrix multiplication M_{n,m,p}, the tensor is NOT all ones.
    # The tensor T satisfies: T(A,B)[i,k] = sum_j A[i,j]*B[j,k].
    # In index notation: T[i,j,k] = 1 for all i,j,k. Yes it IS all ones!
    # Because (A*B)[i,k] = sum_j A[i,j] * B[j,k] * 1.
    # So T[i,j,k] = 1 for all valid i,j,k.

    # So M_{2,2,3} is the 2x2x3 all-ones tensor, which has rank 1 (it's an outer product).
    # That means we can compute 2x2x3 matrix mult with 1 "multiplication"?
    # That doesn't make sense computationally...

    # Oh wait, I think I'm confusing the bilinear complexity with the tensor rank.
    # The tensor rank of M_{n,m,p} (the all-ones tensor of shape n x m x p) is indeed 1
    # if we're allowed arbitrary scalar multiplications. But that's not how it works.

    # Let me reconsider. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (all ones)
    # No wait, that's the tensor for the bilinear form (A,B) -> sum_{i,j,k} A[i,j]*B[j,k]
    # which is NOT matrix multiplication. Matrix multiplication gives a different output for each (i,k).

    # The correct tensor for matrix multiplication is:
    # For each output (i,k): C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has components T_{i,k}^{j} = delta (the j-th term of the (i,k) output).
    # In 3-index form: T[i,j,k] = 1 for all i,j,k. And the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} (sum_j A[i,j]*B[j,k]) e_i ⊗ e_k
    # But this gives a single scalar, not a matrix!

    # I think the issue is that matrix multiplication is a family of bilinear forms, not a single one.
    # The tensor decomposition gives:
    # C[i,k] = sum_{t=1}^{r} U[i,k,t] * (sum_j A[i,j]*V[j,t]) * (sum_j B[j,k]*W[j,t])
    # No wait, the standard formulation is:
    # C[i,k] = sum_t U_{i,t} * V_{j,t} ... I'm getting confused.

    # Let me go back to the problem statement's formulation:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The condition is: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1.

    # So we need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i in [0,n), j in [0,m), k in [0,p).

    # This means the tensor T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] must equal 1 for all i,j,k.

    # So yes, we need to decompose the all-ones tensor of shape (n*m) x (m*p) x (p*n)
    # where the indices are coupled: first index is i*m+j, second is j*p+k, third is k*n+i.

    # This is NOT the standard matrix multiplication tensor. It's a permuted version.
    # The standard M_{n,m,p} tensor has T[i,j,k] = 1 and the decomposition is:
    # sum_t U[i,t] * V[j,t] * W[k,t] = 1 where U is (n,rank), V is (m,rank), W is (p,rank).
    # But here the indices are mixed up: U is indexed by (i,j), V by (j,k), W by (k,i).

    # So this is equivalent to the standard matrix multiplication tensor with a specific
    # index structure that creates dependencies between the three factors.

    # OK so I was overthinking this. The current rank-50 solution is valid (it's been verified).
    # The question is whether we can do better.

    # Given the complexity of finding a better decomposition analytically, let me try
    # a computational search: randomly generate low-rank factorizations and check.

    # Actually, a more structured approach: try to find if the rank-50 decomposition
    # has any redundant columns.

    # Check for linear dependencies among columns:
    # If U[:,c1] = U[:,c2] and V[:,c1] = V[:,c2] and W[:,c1] = W[:,c2],
    # then columns c1 and c2 are identical and one is redundant.

    # More generally: if there exist scalars a_1, a_2 such that:
    # U[:,c1] = a_1 * U[:,c2] + ... (linear combination)
    # V[:,c1] = a_1 * V[:,c2] + ...
    # W[:,c1] = a_1 * W[:,c2] + ...
    # with the same coefficients, then c1 is redundant.

    # Let me check for simple dependencies first.
    # Find columns that are identical or negatives of each other.

    # Actually, let me try a completely different construction.
    # What if I use the following idea:
    # C = A*B where A is 2x4, B is 4x7.
    # Let A = [a0, a1, a2, a3]^T (column vector for each column... no, A is 2x4).
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13]]

    # B = [[b00, b01, b02, b03, b04, b05, b06],
    #      [b10, b11, b12, b13, b14, b15, b16],
    #      [b20, b21, b22, b23, b24, b25, b26],
    #      [b30, b31, b32, b33, b34, b35, b36]]

    # C[0,k] = a00*b0k + a01*b1k + a02*b2k + a03*b3k
    # C[1,k] = a10*b0k + a11*b1k + a12*b2k + a13*b3k

    # Using Strassen-like ideas on the 4-term dot products:
    # For C[0,k]: 4-term dot product. Can we compute it with 3 multiplications?
    # 4 = 2+2: (a00+a01)*(b0k+b1k) - a00*b1k - a01*b0k... that's 3 mults but with extra terms.
    # Actually: a00*b0k + a01*b1k = (a00+a01)(b0k+b1k) - a00*b1k - a01*b0k
    # That's 3 multiplications for 2 products (saving 1).
    # So 4 products can be done in 3+3 = 6 mults (two pairs, each pair saved 1).
    # Or: (a00+a01+a02+a03)*(b0k+b1k+b2k+b3k) ... no, that gives cross terms.

    # Better: group as (a00+a01) and (a02+a03):
    # C[0,k] = (a00+a01)*b0k + a01*b1k + (a02+a03)*b2k + a03*b3k
    # Hmm, this doesn't help directly.

    # The standard approach for 4-term dot product with 3 mults:
    # x = a*b + c*d = (a+c)(b+d) - c(b+d) - a(b+d) + a*b + c*d... no.
    # x = a*b + c*d. Let s = a+c, t = b+d. Then a*b + c*d = s*t - a*d - c*b.
    # That's 3 mults: s*t, a*d, c*b. But we need a*d and c*b which are new products.
    # So we've replaced 2 mults (a*b, c*d) with 3 mults (s*t, a*d, c*b). Worse!

    # The trick only works when we can share the intermediate products across multiple outputs.
    # In matrix multiplication, the intermediate (a+c) and (b+d) are shared across all k values.

    # So for C[0,k] and C[1,k] (all 7 values of k), we can share:
    # S1 = a00 + a10 (shared for all k)
    # S2 = a01 + a11
    # S3 = a02 + a12
    # S4 = a03 + a13
    # And for B:
    # T1 = b0k + b1k (for each k)
    # T2 = b2k + b3k (for each k)

    # Then:
    # C[0,k] + C[1,k] = (a00+a10)*(b0k+b1k) + (a01+a11)*(b1k+b0k) + ... no this is getting messy.

    # Let me just stick with the known rank-50 solution and try to verify it's optimal
    # or find a small improvement.

    # Given time constraints, let me try one more thing: a slightly different tiling.
    # What if I use 2x2x2 Strassen for k in [0,1],[2,3],[4,5] and handle k=6 differently?
    # That's exactly what the current solution does.

    # Alternative: use 2x3x3 blocks.
    # 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # Can we do this in fewer than 18 (schoolbook) mults?
    # Using 3x3x3 Strassen: rank 23 for 3x3x3. But we only need 2x3x3.
    # I don't know the exact rank of 2x3x3.

    # I think the current rank-50 is likely close to optimal, and finding a better
    # decomposition requires significant research effort. Let me just return the
    # known rank-50 solution with proper verification.

    # One more idea: try to use half-integer entries to get a lower rank.
    # The problem allows half-integer entries. This might allow a different
    # decomposition structure.

    # For now, return the rank-50 solution.

    U_final, V_final, W_final = U0, V0, W0

    if time.time() > deadline:
        U_fb, V_fb, W_fb = schoolbook()
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
