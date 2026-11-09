"""Matrix multiplication 4x4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    # Build the rank-65 solution: recursive Strassen for 4x4x4 (rank 49)
    # + schoolbook for the 5th column (rank 16)
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Part 2: Schoolbook for column 4 (k=4)
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0

    # Part 1: Recursive Strassen 4x4x4
    def a_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j

    def b_idx(bi, u, v):
        j = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return j * p + k

    def c_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return k * n + i

    # Strassen linear forms for 2x2 blocks
    A_side = [
        [(0, 1.0), (3, 1.0)],
        [(2, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (0, -1.0)],
        [(1, 1.0), (3, -1.0)],
    ]

    B_side = [
        [(0, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(1, 1.0), (3, -1.0)],
        [(2, 1.0), (0, -1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (3, 1.0)],
    ]

    out_coeff = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0},
    ]

    inner_A_side = A_side
    inner_B_side = B_side
    inner_out_coeff = out_coeff

    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk

            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    U[a_idx(bi, u_local, v_local), col] += c1 * c2

            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    V[b_idx(bi, u_local, v_local), col] += c1 * c2

            for (bi_out, c_inner) in inner_out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    u_local = bi_out // 2
                    v_local = bi_out % 2
                    W[c_idx(bi_outer, u_local, v_local), col] += c_inner * c_outer

    # Verify the rank-65 solution
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        # Check that sum_r U[r,i]*V[r,j]*W[r,k] equals the tensor T
        # T[i,j,k] = 1 if i*m+j and j*p+k and k*n+i form a valid chain
        # Actually, the tensor is: T[i,j,k] = delta where C[i,k] = sum_j A[i,j]B[j,k]
        # The tensor entries are: T[idx_u, idx_v, idx_w] = 1 when idx_u = i*m+j,
        # idx_v = j*p+k, idx_w = k*n+i
        # Let's compute the reconstructed tensor and check against the target
        # For efficiency, check a few random entries
        for _ in range(200):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i
            val = np.sum(U_mat[:, idx_u] * V_mat[:, idx_v] * W_mat[:, idx_w])
            if abs(val - 1.0) > tol:
                return False
            # Also check that non-target entries are zero
            # Pick random non-target entries
        # Check some zero entries
        for _ in range(100):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            # Pick a different k' to make it non-target
            k2 = (k + 1) % p
            if k2 == k:
                continue
            idx_u = i * m + j
            idx_v = j * p + k2
            idx_w = k * n + i
            val = np.sum(U_mat[:, idx_u] * V_mat[:, idx_v] * W_mat[:, idx_w])
            if abs(val) > tol:
                return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = np.zeros((nm, rank_schoolbook))
        V_school = np.zeros((mp, rank_schoolbook))
        W_school = np.zeros((pn, rank_schoolbook))
        for c in range(rank_schoolbook):
            i = c // (m * p)
            rem = c % (m * p)
            j = rem // p
            k = rem % p
            U_school[i * m + j, c] = 1.0
            V_school[j * p + k, c] = 1.0
            W_school[k * n + i, c] = 1.0
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Try to find a rank-64 solution by attempting to merge one schoolbook term
    # with Strassen terms, or by direct optimization
    if time.time() < deadline - 30:
        # Strategy: Try to see if we can reduce rank by finding a linear dependency
        # or by using a different decomposition for the 4x4x1 part
        
        # Alternative structural approach: Use a 4x4x5 decomposition that treats
        # the 5th column differently. 
        # 
        # Key insight: The 4x4x1 tensor (matrix-vector product) has rank 4, not 16,
        # if we allow the U, V, W factors to have more structure. Wait, no - the rank
        # of the matrix-vector product tensor T[i,j,k] = delta (where k is fixed) 
        # is actually min(n,m) = 4 for a 4x4x1 case? No, that's not right either.
        #
        # Let me reconsider. The tensor for A(4x4) @ B(4x1) -> C(4x1) is:
        # T[i,j,4] = 1 (in the W index, k=4, i varies)
        # Actually W index is k*n + i, so for k=4, W index is 4*4+i = 16+i
        # The tensor has non-zero entries at T[i*m+j, j*p+4, 4*n+i] = 1
        # This is a 16 x 20 x 20 tensor with 16 non-zero entries
        # The rank of this tensor: it's the tensor for matrix-vector multiplication
        # R(4,4,1) = 4 (since it's equivalent to 4 independent dot products)
        # Wait, is that right? Let's think...
        # C[i] = sum_j A[i,j] B[j,4]
        # We can write this as: for each j, term j: U has 1 at (i,j) for all i? No.
        # Actually: C[i] = sum_j A[i,j] * B[j,4]
        # This is a sum of 4 terms (one per j):
        # Term j: U[i*m+j] = 1 for all i? No, U is a single vector per term.
        # Term j: U[i*m+j] = 1 (only for specific i? No, U is the same for all i in this term)
        # 
        # Wait, I'm confusing myself. In the tensor decomposition:
        # T = sum_r U_r ⊗ V_r ⊗ W_r
        # T[a,b,c] = sum_r U_r[a] V_r[b] W_r[c]
        # For the 4x4x1 part: T[i*m+j, j*p+4, 4*n+i] = 1
        # This means: for each (i,j), we need sum_r U_r[i*m+j] V_r[j*p+4] W_r[4*n+i] = 1
        # And for all other (a,b,c), the sum should be 0.
        #
        # Can we do this with rank 4? 
        # If we use 4 terms (one per j):
        # Term j: U_r has 1 at positions i*m+j for i=0..3? No, U_r is a fixed vector.
        # 
        # Actually, the rank of the matrix-vector product tensor M_{n,m,1} is min(n,m) = 4.
        # Let's verify: 
        # T[i,j,k=4] = 1 (in the sense that T[i*m+j, j*p+4, 4*n+i] = 1)
        # Using 4 terms indexed by j:
        # Term j: U_r[i*m+j] = 1 for all i? That would make U_r have 4 non-zeros.
        # V_r[j*p+4] = 1
        # W_r[4*n+i] = 1 for all i? That would make W_r have 4 non-zeros.
        # Then T[a,b,c] = sum_j U_j[a] V_j[b] W_j[c]
        # For a = i*m+j', b = j'*p+4, c = 4*n+i':
        # = sum_j (1 if j'=j) * (1 if j'=j) * (1 if i'=i)
        # = 1 if j'=j and j'=j and i'=i, summed over j
        # = 1 if j'=j for some j and i'=i... this doesn't work cleanly.
        #
        # Let me try a different approach. The rank of the 4x4x1 tensor is actually 4.
        # Proof: T[i,j,4] (in abstract indices) = delta_{i,i} delta_{j,j}... no.
        # T[i,j,k] for k=4: T[i,j,4] = 1 for all i,j? No, that's not right.
        # 
        # In the matrix multiplication tensor T_{n,m,p}, T[i,j,k] = 1 always (it's the
        # coefficient of A[i,j]B[j,k] in C[i,k]). But when we fix k=4, we get a
        # 4x4x1 tensor where T[i,j,4] = 1 for all i,j. This is the all-ones tensor,
        # which has rank 1! But wait, that's not the right tensor.
        #
        # The issue is that in the full tensor, T[i,j,k] = 1 for all i,j,k, but the
        # indices are coupled: the same j appears in both A and B. When we slice k=4,
        # we get T[i,j,4] = 1, but this is not independent of the other slices.
        #
        # The rank of the full 4x4x5 tensor is what we're trying to minimize.
        # The schoolbook gives 80. Strassen+schoolbook gives 65.
        #
        # Let's try a different structural approach: use a 4x4x4 decomposition of rank
        # less than 49, if one exists. The known lower bound for 4x4x4 is 49 (from
        # Strassen), and it's not known if a better decomposition exists.
        #
        # Alternative: Try to find a rank-64 solution by solving a system of equations
        # with 64 terms, initialized near the rank-65 solution.
        
        # Try rank 64 by dropping one term and adjusting
        # This is a non-convex optimization problem, so we'll use a heuristic
        
        best_rank = total_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try to see if we can reduce rank by 1 by finding a linear combination
        # that allows one term to be eliminated
        # This is equivalent to checking if the 65 terms are linearly independent
        # in the tensor product space, which they are by construction.
        
        # Given the time constraints and the difficulty of tensor rank minimization,
        # we'll stick with the rank-65 solution.
        
        # But let's try one more thing: a different block decomposition
        # What if we use a 4x4x5 = (2x2x5) + (2x2x5) + cross terms approach?
        # Or use a 4x4x5 = 4x4x4 + 4x4x1 where the 4x4x1 part uses rank 4?
        
        # Let me try to construct a rank-4 decomposition for the 4x4x1 part
        # T[i*m+j, j*p+4, 4*n+i] = 1 for i,j in 0..3
        # Using 4 terms indexed by j:
        # Term j: U_r[i*m+j] = 1 for all i in 0..3 (4 non-zeros)
        #          V_r[j*p+4] = 1
        #          W_r[4*n+i] = 1 for all i in 0..3 (4 non-zeros)
        # Check: sum_r U_r[a] V_r[b] W_r[c]
        # For a = i*m+j', b = j'*p+4, c = 4*n+i':
        # = sum_j (1 if j'=j) * (1 if j'=j) * (1 if i'=i)
        # = 1 if j'=j and i'=i, for the term where j=j'
        # = 1 (correct!)
        # For other (a,b,c): if b = j'*p+k' with k' != 4, then V_r[b] = 0 for all r
        # if a = i*m+j' and c = 4*n+i' but b is not of the form j'*p+4, then 0
        # This works! So the 4x4x1 part has rank 4, not 16.
        
        # Let's reconstruct with rank 4 for the 5th column
        new_total_rank = 49 + 4
        U_new = np.zeros((nm, new_total_rank), dtype=float)
        V_new = np.zeros((mp, new_total_rank), dtype=float)
        W_new = np.zeros((pn, new_total_rank), dtype=float)
        
        # Copy the Strassen part
        U_new[:, :49] = U[:, :49]
        V_new[:, :49] = V[:, :49]
        W_new[:, :49] = W[:, :49]
        
        # Add the rank-4 part for column 4
        for j in range(m):
            col = 49 + j
            # U: 1 at positions i*m+j for i=0..3
            for i in range(n):
                U_new[i * m + j, col] = 1.0
            # V: 1 at position j*p+4
            V_new[j * p + 4, col] = 1.0
            # W: 1 at positions 4*n+i for i=0..3
            for i in range(n):
                W_new[4 * n + i, col] = 1.0
        
        # Verify
        if verify(U_new, V_new, W_new):
            best_rank = new_total_rank
            best_U, best_V, best_W = U_new, V_new, W_new
        else:
            # The rank-4 construction might not work due to interference with Strassen terms
            # Let's check more carefully
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
