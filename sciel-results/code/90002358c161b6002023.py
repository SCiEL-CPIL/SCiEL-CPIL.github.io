import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # The 3x3 matrix multiplication tensor has 27 non-zero entries.
    # We seek a rank < 27 decomposition.
    #
    # Strategy: Use a block-structured approach.
    # Partition 3x3 matrices into:
    #   A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    #   B similarly.
    #
    # For the 2x2 blocks, Strassen's algorithm uses 7 multiplications instead of 4.
    # For the cross terms (2x1 * 1x2, 2x1 * 1x1, 1x2 * 1x2, 1x1 * 1x2, 1x1 * 1x1),
    # we need additional multiplications.
    #
    # Total: 7 (for 2x2*2x2) + 4 (for 2x1*1x2, 2x1*1x1, 1x2*1x2, 1x1*1x2) + 1 (for 1x1*1x1)
    # But some of these can be combined using Strassen-like tricks.
    #
    # Actually, let's use a more direct approach:
    # The known result is that the tensor rank of 3x3 matrix multiplication is 23.
    # But constructing an explicit rank-23 decomposition is very complex.
    #
    # Let's try a rank-26 approach first using a simpler block decomposition:
    # Use Strassen for the top-left 2x2 block (saves 1 multiplication: 7 instead of 8 for 2x2x2x2)
    # Wait, 2x2*2x2 needs 4 scalar multiplications normally, Strassen uses 7? No, that's wrong.
    # 2x2*2x2: 4 scalar mults normally. Strassen for 2x2 uses 7 mults but that's for larger blocks.
    # Actually Strassen's algorithm: for n x n where n is power of 2, it recursively uses 7 mults.
    # For 2x2, the base case is 4 mults. Strassen doesn't help at 2x2 level directly.
    #
    # Let me reconsider. The known minimal rank for 3x3 matrix multiplication is 23 (Alon-Boppana lower bound gives 19, but exact is known to be 23 via Landsberg).
    #
    # A practical approach: use the "border rank" or known explicit decompositions.
    #
    # Alternative: Use a rank-24 decomposition based on the following idea:
    # The 3x3 multiplication tensor can be decomposed using 24 rank-1 tensors.
    #
    # Let me try a different structural approach:
    # Embed 3x3 in 4x4, use Strassen's rank-19 for 4x4 (wait, Strassen gives 7^log2(4)=49 for 4x4? No.)
    # Strassen: T(4) = 7*T(2) = 7*4 = 28? No, T(2)=4 for base case, so T(4)=7*4=28.
    # That's worse than 16 for schoolbook 4x4.
    #
    # Actually, Strassen's algorithm: T(n) = 7*T(n/2), T(1)=1, so T(2)=7, T(4)=49.
    # Wait no: T(1)=1 (one multiplication), T(2)=7*T(1)=7, T(4)=7*T(2)=49.
    # But schoolbook for 4x4 is 16. So Strassen only wins for larger n.
    #
    # For 3x3, let's use a direct algebraic construction.
    #
    # Known: The tensor rank of M_{3,3,3} is exactly 23.
    # Reference: "The rank of the matrix multiplication tensor" by various authors.
    #
    # Let me try to construct a rank-24 decomposition using a systematic approach:
    # Use the fact that we can share some multiplications between overlapping blocks.
    #
    # Actually, let me try a completely different approach:
    # Use a rank-25 decomposition by finding one shared multiplication.
    #
    # The 27 scalar multiplications in schoolbook are:
    # C[i][k] = sum_j A[i][j] * B[j][k] for i=0,1,2 and k=0,1,2
    # That's 9 output entries, each requiring 3 multiplications = 27 total.
    #
    # To reduce rank, we need to find linear combinations where the same product
    # of linear combinations of A and B entries appears in multiple C entries.
    #
    # Let me try a rank-24 construction based on the following:
    # Group the 27 terms into 24 by finding 3 pairs that can be merged.
    #
    # For example, consider C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    # and C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]
    #
    # These don't obviously share structure.
    #
    # Let me try yet another approach: use a known explicit rank-23 or rank-24 decomposition.
    #
    # Actually, I recall that there's a rank-23 decomposition for 3x3 matrix multiplication
    # that was found using Gröbner basis methods. Let me try to implement one.
    #
    # The key insight: we can use the "Coppersmith-Winograd" style decompositions
    # or more practically, use a parametrized family and solve for the parameters.
    #
    # Let me try a rank-24 decomposition with the following structure:
    # Use 24 rank-1 terms where each term is u_k * v_k * w_k with:
    # u_k in R^9 (flattened A), v_k in R^9 (flattened B), w_k in R^9 (flattened C)
    #
    # The constraint is: sum_k u_k[a] * v_k[b] * w_k[c] = T[a,b,c] for all a,b,c
    # where T[a,b,c] = 1 if (a,b,c) corresponds to a valid (i,j,k) triple, else 0.
    #
    # This is a system of 729 equations (but only 27 are non-trivial) in 24*27 = 648 unknowns.
    #
    # Let me try a numerical approach with a better initialization:
    # Use the schoolbook decomposition and then try to merge pairs of terms.
    #
    # Actually, let me try a completely different structural approach:
    # Use the "block Strassen" idea more carefully.
    #
    # Write A, B, C as block matrices:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    #
    # C = A*B gives:
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x1)
    # C21 = A21*B11 + A22*B21  (1x2)
    # C22 = A21*B12 + A22*B22  (1x1)
    #
    # For C11 (2x2 = 2x2 + 2x1*1x2):
    #   A11*B11: 4 scalar mults (or 7 with Strassen on 2x2? No, Strassen needs larger blocks)
    #   A12*B21: 2*1*2 = 4 scalar mults (outer product)
    #   Total for C11: 8 scalar mults
    #
    # For C12 (2x1 = 2x2*1x2 + 2x1*1x1):
    #   A11*B12: 2*2*1 = 4 scalar mults
    #   A12*B22: 2*1*1 = 2 scalar mults
    #   Total for C12: 6 scalar mults
    #
    # For C21 (1x2 = 1x2*2x2 + 1x1*1x2):
    #   A21*B11: 1*2*2 = 4 scalar mults
    #   A22*B21: 1*1*2 = 2 scalar mults
    #   Total for C21: 6 scalar mults
    #
    # For C22 (1x1 = 1x2*1x2 + 1x1*1x1):
    #   A21*B12: 1*2*1 = 2 scalar mults
    #   A22*B22: 1 scalar mult
    #   Total for C22: 3 scalar mults
    #
    # Grand total: 8 + 6 + 6 + 3 = 23 scalar mults!
    #
    # Wait, that's exactly 23! But this is just counting the scalar multiplications in the block decomposition.
    # Each scalar multiplication corresponds to a rank-1 term in the tensor decomposition.
    #
    # So the block decomposition directly gives a rank-23 decomposition!
    #
    # Let me verify: the block structure partitions the 9x9x9 tensor into blocks,
    # and within each block, we use the standard multiplication. The total number of
    # scalar multiplications is the sum over all blocks, which is 23.
    #
    # But wait - is this correct? The block decomposition of matrix multiplication
    # doesn't reduce the number of scalar multiplications below 27 in general.
    # Let me recount.
    #
    # A is 3x3, partitioned as:
    #   Row 0: A[0,0], A[0,1], A[0,2]
    #   Row 1: A[1,0], A[1,1], A[1,2]
    #   Row 2: A[2,0], A[2,1], A[2,2]
    #
    # If we partition as 2x2 + 2x1 + 1x2 + 1x1:
    #   A11 = [[A00, A01], [A10, A11]] (2x2)
    #   A12 = [[A02], [A12]] (2x1)
    #   A21 = [[A20, A21]] (1x2)
    #   A22 = [[A22]] (1x1)
    #
    # Similarly for B.
    #
    # C11 = A11*B11 + A12*B21
    #   A11*B11: 2x2 * 2x2 = 4 scalar mults (each C element is sum of 2 products)
    #   Actually, 2x2*2x2 requires 4*2 = 8 scalar mults? No.
    #   C11[i][j] = sum_k A11[i][k]*B11[k][j] for i=0,1 and j=0,1
    #   That's 4 output entries, each with 2 terms = 8 scalar mults.
    #
    #   A12*B21: 2x1 * 1x2 = outer product, 2*1*2 = 4 scalar mults.
    #   C11[i][j] += A12[i]*B21[j], 4 terms.
    #
    #   Total for C11: 8 + 4 = 12 scalar mults.
    #
    # C12 = A11*B12 + A12*B22
    #   A11*B12: 2x2 * 2x1, C12[i] = sum_k A11[i][k]*B12[k], 2 output entries, 2 terms each = 4 scalar mults.
    #   A12*B22: 2x1 * 1x1, C12[i] += A12[i]*B22, 2 scalar mults.
    #   Total for C12: 4 + 2 = 6 scalar mults.
    #
    # C21 = A21*B11 + A22*B21
    #   A21*B11: 1x2 * 2x2, C21[j] = sum_k A21[k]*B11[k][j], 2 output entries, 2 terms each = 4 scalar mults.
    #   A22*B21: 1x1 * 1x2, C21[j] += A22*B21[j], 2 scalar mults.
    #   Total for C21: 4 + 2 = 6 scalar mults.
    #
    # C22 = A21*B12 + A22*B22
    #   A21*B12: 1x2 * 2x1, C22 = sum_k A21[k]*B12[k], 2 scalar mults.
    #   A22*B22: 1x1 * 1x1, 1 scalar mult.
    #   Total for C22: 2 + 1 = 3 scalar mults.
    #
    # Grand total: 12 + 6 + 6 + 3 = 27. Back to 27!
    #
    # So the block decomposition doesn't help. The 2x2*2x2 block still needs 8 scalar mults
    # (not 4), because each of the 4 output entries requires 2 multiplications.
    #
    # OK so I need to actually find a way to reduce the rank. Let me use a numerical approach
    # but with a much better strategy: use the known rank-23 decomposition structure.
    #
    # Let me try a different approach: use a rank-24 decomposition found via
    # "tensor rank reduction" by finding one linear dependency.
    #
    # Actually, let me just try to find a rank-24 or rank-25 decomposition numerically
    # with a much better optimization strategy: use alternating least squares (ALS)
    # which is more stable for tensor decompositions.
    #
    # ALS for tensor decomposition:
    # Given T and target rank R, alternate between:
    # 1. Fix V, W, solve for U (linear least squares)
    # 2. Fix U, W, solve for V
    # 3. Fix U, V, solve for W
    #
    # This is more stable than full gradient descent.

    # Let me implement ALS with snapping to half-integers.

    # Precompute the target tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    nz_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_set.add((a, b, c))

    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2

        # Verify exactness
        for a, b, c in nz_set:
            val = np.sum(U_snap[a] * V_snap[b] * W_snap[c])
            if abs(val - 1.0) > 1e-9:
                return None
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in nz_set:
                        val = np.sum(U_snap[a] * V_snap[b] * W_snap[c])
                        if abs(val) > 1e-9:
                            return None

        # Prune zero columns
        mask = np.any(U_snap != 0, axis=0) | np.any(V_snap != 0, axis=0) | np.any(W_snap != 0, axis=0)
        U_final = U_snap[:, mask]
        V_final = V_snap[:, mask]
        W_final = W_snap[:, mask]

        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }

    def get_fallback(rank):
        if rank != 27:
            return None
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for idx, (i, j, k) in enumerate(
            [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        ):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a, idx] = 1.0
            V[b, idx] = 1.0
            W[c, idx] = 1.0
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    best_result = get_fallback(27)

    # Try ALS for ranks 23-26
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break

        num_restarts = 10 if rank <= 24 else 5
        for r in range(num_restarts):
            if time.time() > deadline:
                break

            np.random.seed(42 + r * 1000 + rank)

            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3

            # ALS iterations
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break

                # Solve for U: for each a, U[a,:] should satisfy
                # sum_k U[a,k] * V[b,k] * W[c,k] = T[a,b,c] for all b,c
                # This is a linear system: U[a,:] @ (V[b,:] * W[c,:]) = T[a,b,c]
                # Vectorize: for fixed a, we need U[a,:] such that
                # sum_k U[a,k] * (V[b,k] * W[c,k]) = T[a,b,c] for all (b,c)
                # This is: U[a,:] @ M_a = T[a,:,:].flatten()
                # where M_a[b,c,k] = V[b,k]*W[c,k], reshaped to ((m*p)*(p*n), rank)

                for a in range(n*m):
                    # Build the matrix: rows indexed by (b,c), columns by k
                    # M[b,c,k] = V[b,k] * W[c,k]
                    M_a = np.einsum('bk,ck->bck', V, W).reshape(m*p*p*n, rank)
                    t_a = T[a, :, :].reshape(-1)
                    # Solve least squares: U[a,:] @ M_a.T = t_a
                    # i.e., M_a @ U[a,:].T = t_a
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(M_a, t_a, rcond=None)
                    except:
                        pass

                # Solve for V
                for b in range(m*p):
                    M_b = np.einsum('ak,ck->ack', U, W).reshape(n*m*p*n, rank)
                    t_b = T[:, b, :].reshape(-1)
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(M_b, t_b, rcond=None)
                    except:
                        pass

                # Solve for W
                for c in range(p*n):
                    M_c = np.einsum('ak,bk->abk', U, V).reshape(n*m*m*p, rank)
                    t_c = T[:, :, c].reshape(-1)
                    try:
                        W[c, :], _, _, _ = np.linalg.lstsq(M_c, t_c, rcond=None)
                    except:
                        pass

                # Check convergence
                err = 0.0
                for a, b, c in nz_set:
                    val = np.sum(U[a] * V[b] * W[c])
                    err += (val - 1.0)**2
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            if (a, b, c) not in nz_set:
                                val = np.sum(U[a] * V[b] * W[c])
                                err += val**2

                if err < 1e-6:
                    converged = True
                    break

            if time.time() > deadline:
                break

            # Try to snap and verify
            result = snap_and_verify(U, V, W)
            if result:
                final_rank = len(result["U"][0])
                if final_rank < (27 if best_result is None else len(best_result["U"][0])):
                    best_result = result
                    if final_rank <= rank:
                        return result

            # Also try without snapping (in case the solution is close to integers)
            result_int = snap_and_verify(np.round(U), np.round(V), np.round(W))
            if result_int:
                final_rank = len(result_int["U"][0])
                if final_rank < (27 if best_result is None else len(best_result["U"][0])):
                    best_result = result_int
                    if final_rank <= rank:
                        return result_int

    return best_result
