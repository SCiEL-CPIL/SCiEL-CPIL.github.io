import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 148.0  # 2s safety margin

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125

    def compute_errors(U, V, W):
        """Compute all (i,j,k) errors efficiently using einsum."""
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    errors[i, j, k] = val - 1.0
        return errors

    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        errors = compute_errors(U, V, W)
        return np.max(np.abs(errors))

    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_result = (U_school.copy(), V_school.copy(), W_school.copy())

    # === Deterministic block decomposition: 2x2 + 3x3 blocks ===
    # Partition indices: top-left 2x2 block uses indices {0,1}, bottom-right 3x3 uses {2,3,4}
    # Block structure:
    #   A = [[A11, A12],    B = [[B11, B12],    C = A*B = [[C11, C12],
    #        [A21, A22]]         [B21, B22]]              [C21, C22]]
    #   A11: 2x2, A12: 2x3, A21: 3x2, A22: 3x3
    #   B11: 2x2, B12: 2x3, B21: 3x2, B22: 3x3
    #   C11 = A11*B11 (2x2)
    #   C12 = A11*B12 + A12*B22 (2x3)
    #   C21 = A21*B11 + A22*B21 (3x2)
    #   C22 = A21*B12 + A22*B22 (3x3)

    # Strassen algorithm for 2x2 matrix multiplication (rank 7)
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Laderman algorithm for 3x3 matrix multiplication (rank 15)
    # This is a known rank-15 decomposition for 3x3 matrix multiplication
    # Laderman's algorithm uses 15 multiplications:
    # M1 = (A00+A11+A22)*(B00+B11+B22)
    # M2 = (A00+A22-A11)*(B00+B22-B11)
    # M3 = (A11+A22-A00)*(B11+B22-B00)
    # M4 = (A00+A11-A22)*(B00+B11-B22)
    # M5 = (A00+A22-A11)*(B00+B22-B11)  -- wait, this is same as M2?
    # Let me use the correct Laderman coefficients.
    # Actually, let me just use schoolbook for 3x3 (rank 27) for simplicity and correctness.
    # Then total rank would be: 7 + 27 + 6 + 6 = 46
    # But wait, the off-diagonal blocks also need to be handled.

    # Let me reconsider the block structure more carefully.
    # For 5x5 * 5x5 -> 5x5:
    # C[i,j] = sum_{k=0}^{4} A[i,k] * B[k,j]
    #
    # Split k into {0,1} and {2,3,4}:
    # C[i,j] = sum_{k=0}^{1} A[i,k]*B[k,j] + sum_{k=2}^{4} A[i,k]*B[k,j]
    #
    # For i in {0,1}, j in {0,1}: C11 = A11*B11 (2x2 * 2x2 -> 2x2)
    # For i in {0,1}, j in {2,3,4}: C12 = A11*B12 + A12*B22 (2x2 * 2x3 + 2x3 * 3x3 -> 2x3)
    # For i in {2,3,4}, j in {0,1}: C21 = A21*B11 + A22*B21 (3x2 * 2x2 + 3x3 * 3x2 -> 3x2)
    # For i in {2,3,4}, j in {2,3,4}: C22 = A21*B12 + A22*B22 (3x2 * 2x3 + 3x3 * 3x3 -> 3x3)

    # Each of these is a separate tensor that needs to be decomposed.
    # C11: 2x2 * 2x2 -> 2x2, rank 7 (Strassen)
    # C12: (2x2 * 2x3) + (2x3 * 3x3) -> 2x3
    #   The first part: 2x2 * 2x3, rank 6 (schoolbook)
    #   The second part: 2x3 * 3x3, rank 18 (schoolbook)
    #   Total for C12: 6 + 18 = 24
    # C21: (3x2 * 2x2) + (3x3 * 3x2) -> 3x2
    #   The first part: 3x2 * 2x2, rank 12 (schoolbook)
    #   The second part: 3x3 * 3x2, rank 18 (schoolbook)
    #   Total for C21: 12 + 18 = 30
    # C22: (3x2 * 2x3) + (3x3 * 3x3) -> 3x3
    #   The first part: 3x2 * 2x3, rank 18 (schoolbook)
    #   The second part: 3x3 * 3x3, rank 27 (schoolbook)
    #   Total for C22: 18 + 27 = 45
    #
    # Grand total: 7 + 24 + 30 + 45 = 106
    #
    # This is better than 125! Let me implement this.

    # Actually, I can do better by using Strassen for the 2x2 parts within the off-diagonal blocks too.
    # But for now, let me just get a valid decomposition with rank 106.

    # Build the decomposition
    U = np.zeros((n*m, 106))
    V = np.zeros((m*p, 106))
    W = np.zeros((p*n, 106))

    col = 0

    # C11 = A11 * B11 (2x2 * 2x2 -> 2x2) using Strassen (rank 7)
    # Strassen terms for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each Strassen term, we need to express it as a rank-1 decomposition of the sub-tensor.
    # The sub-tensor for 2x2 * 2x2 -> 2x2 has entries (i,j,k) for i,j,k in {0,1}.
    # Each Strassen multiplication M_r corresponds to a specific pattern.

    # Let me define the Strassen terms more carefully.
    # For 2x2 matrix multiplication, the tensor T[i,j,k] = 1 if the (i,j) entry of the product
    # depends on A[i,k]*B[k,j]. In schoolbook, T[i,j,k] = 1 for all i,j,k.
    # In Strassen, we have 7 terms, each of which is a product of linear combinations.

    # Actually, the Strassen algorithm gives us a decomposition of the tensor into 7 rank-1 terms.
    # Each term is of the form: (linear combination of A entries) * (linear combination of B entries) -> (linear combination of C entries)

    # Let me just use the schoolbook for all blocks for simplicity and correctness.
    # Total rank: 8 + 30 + 30 + 45 = 113 (for the four blocks)
    # Wait, that's not right either. Let me think about this differently.

    # The tensor for 5x5 * 5x5 -> 5x5 has 125 entries (i,j,k) for i,j,k in {0..4}.
    # Each entry T[i,j,k] = 1 (since C[i,j] = sum_k A[i,k]*B[k,j]).
    # We need to decompose this tensor into rank-1 terms.
    # A rank-1 term is of the form: u[i,j] * v[j,k] * w[k,i] for some vectors u, v, w.

    # Schoolbook: T[i,j,k] = e_i[i]*e_j[j] * e_j[j]*e_k[k] * e_k[k]*e_i[i] = 1
    # So each (i,j,k) gives one rank-1 term. Total rank = 125.

    # To reduce the rank, we need to find linear combinations that still give T[i,j,k] = 1 for all (i,j,k).

    # Let me try a different approach: use the fact that the tensor is symmetric in some sense
    # and try to find a lower-rank decomposition using known results.

    # For now, let me just return the schoolbook decomposition.
    # The optimization approach in the original code was not working well.

    # Let me try one more thing: use a block-diagonal structure.
    # If we can find a permutation of the indices that makes the tensor block-diagonal,
    # we can decompose each block separately.

    # Actually, let me just implement the optimization approach more carefully,
    # with a better initialization and more iterations.

    candidate_ranks = [34, 40, 46, 50, 55, 60, 70, 80, 90, 100, 110, 120]

    for rank in candidate_ranks:
        if time.time() > deadline - 25:
            break

        if rank >= rank_schoolbook:
            continue

        best_err_for_rank = float('inf')
        best_U, best_V, best_W = None, None, None

        for init_seed in range(8):
            if time.time() > deadline - 15:
                break

            rng = np.random.RandomState(init_seed * 1000 + rank + 7)

            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))

            # Initialize with schoolbook-like pattern
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = 1.0
                            col += 1

            # Add random perturbation for the remaining columns
            for r in range(col, rank):
                i_idx = rng.randint(n)
                j_idx = rng.randint(m)
                k_idx = rng.randint(p)
                U[i_idx*m+j_idx, r] = 1.0 + rng.randn() * 0.1
                V[j_idx*p+k_idx, r] = 1.0 + rng.randn() * 0.1
                W[k_idx*n+i_idx, r] = 1.0 + rng.randn() * 0.1

            # Alternating minimization
            prev_err = float('inf')
            for it in range(60):
                if time.time() > deadline - 10:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((rank, p))
                        for k in range(p):
                            B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = sol
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = sol
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((rank, m))
                        for j in range(m):
                            C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = sol
                        except:
                            pass

                # Compute error
                errors = compute_errors(U, V, W)
                max_err = np.max(np.abs(errors))

                if max_err < best_err_for_rank:
                    best_err_for_rank = max_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()

                if max_err < 1e-10:
                    break

                if max_err > prev_err * 1.5 and it > 15:
                    break
                prev_err = max_err

            if best_err_for_rank < 1e-8:
                break

        if best_err_for_rank < 1e-6 and best_U is not None:
            # Round to nearest half-integer
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            max_err = verify(U_r, V_r, W_r, rank)
            if max_err < 1e-8:
                best_rank = rank
                best_result = (U_r, V_r, W_r)
                if rank <= 50:
                    break

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
