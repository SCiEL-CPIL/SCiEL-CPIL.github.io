"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def elapsed():
        return time.time() - start_time

    def verify_numpy(U, V, W, tol=1e-9):
        """Verify decomposition using numpy. U:(nm,r), V:(mp,r), W:(pn,r)"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = to_lists(U_school, V_school, W_school)
    best_rank = schoolbook_rank

    # Strategy: Use Strassen-like decomposition
    # Partition A (4x5) as [A1 | A2] where A1 is 4x4, A2 is 4x1
    # Partition B (5x5) as [B1; B2] where B1 is 4x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    #
    # Further partition B1 (4x5) as [B1a | B1b] where B1a is 4x4, B1b is 4x1
    # A1*B1 = A1*B1a + A1*B1b
    #
    # A1*B1a: 4x4 * 4x4 -> Use Strassen (rank 8)
    # A1*B1b: 4x4 * 4x1 -> Schoolbook (rank 16)
    # A2*B2: 4x1 * 1x5 -> Schoolbook (rank 20)
    # Total: 8 + 16 + 20 = 44

    # Strassen for 4x4 * 4x4:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6

    # For 4x4*4x4, the indices are:
    # A rows: 0,1,2,3; A cols: 0,1,2,3
    # B rows: 0,1,2,3; B cols: 0,1,2,3
    # C rows: 0,1,2,3; C cols: 0,1,2,3

    # Strassen decomposition for 4x4*4x4 (rank 8):
    # We need to express this in terms of the global indices
    # In the full problem, A1 is at rows 0-3, cols 0-3 of A
    # B1a is at rows 0-3, cols 0-3 of B
    # The output C1a is at rows 0-3, cols 0-3 of C

    # Let's build the Strassen part:
    U_strassen = np.zeros((nm, 8))
    V_strassen = np.zeros((mp, 8))
    W_strassen = np.zeros((pn, 8))

    # Strassen formulas for 2x2 blocks:
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2
    # Let B = [[B11, B12], [B21, B22]] where each is 2x2
    #
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4, we partition into 2x2 blocks of 2x2:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
    # C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]

    # M1: (A11+A22)*(B11+B22)
    # U: A11 entries (rows 0,1; cols 0,1) and A22 entries (rows 2,3; cols 2,3)
    # V: B11 entries (rows 0,1; cols 0,1) and B22 entries (rows 2,3; cols 2,3)
    # W: C11 (rows 0,1; cols 0,1), C12 (rows 0,1; cols 2,3), C21 (rows 2,3; cols 0,1), C22 (rows 2,3; cols 2,3)

    # Actually, Strassen for 2x2*2x2 gives 7 multiplications for 2x2 output.
    # For 4x4*4x4, we apply Strassen to the 2x2 block structure, giving 7^2 = 49?
    # No, that's not right. Strassen for 4x4 uses 7 multiplications of 2x2 matrices,
    # which is 7 * 4 = 28 scalar multiplications? No...

    # Let me reconsider. Strassen's algorithm for n x n where n is power of 2:
    # For 2x2: 7 multiplications
    # For 4x4: 7^2 = 49 multiplications (each being 2x2 multiplications)
    # But each 2x2 multiplication is 4 scalar multiplications, so 49*4 = 196? That's worse!

    # I think I'm confusing myself. Let me look at this differently.

    # The rank of the 2x2*2x2 matrix multiplication tensor is 7 (Strassen).
    # The rank of the 4x4*4x4 tensor is at most 7^2 = 49 (by applying Strassen recursively).
    # But the schoolbook rank is 16, so 49 > 16, meaning recursive Strassen is worse!

    # So for 4x4*4x4, we should just use schoolbook (rank 16) or find a better decomposition.

    # Let me reconsider the partition:
    # A (4x5) * B (5x5) = C (4x5)
    #
    # Partition A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # Partition B = [B1; B2] where B1 is 4x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 4x4 * 4x5 -> schoolbook rank 16*5 = 80? No, 4*4*5 = 80
    # Wait, 4x4 * 4x5 has 4*4*5 = 80 scalar multiplications in schoolbook.
    # A2*B2: 4x1 * 1x5 has 4*1*5 = 20 scalar multiplications.
    # Total: 80 + 20 = 100 (same as schoolbook for the whole thing)

    # So partitioning doesn't help if we use schoolbook for each part.

    # Let me try a different approach: use the fact that some entries can be shared.

    # Actually, let me just try random search more aggressively.

    def try_random_rank(target_rank, num_trials=100, seed=42):
        """Try to find a rank-target_rank decomposition."""
        import random
        rng = random.Random(seed)
        for trial in range(num_trials):
            if elapsed() > deadline - 5:
                return None

            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))

            for c in range(target_rank):
                for _ in range(nm // 3):
                    row = rng.randint(0, nm - 1)
                    U[row, c] = rng.choice([-1, 0, 1, 0.5, -0.5])
                for _ in range(mp // 3):
                    row = rng.randint(0, mp - 1)
                    V[row, c] = rng.choice([-1, 0, 1, 0.5, -0.5])
                for _ in range(pn // 3):
                    row = rng.randint(0, pn - 1)
                    W[row, c] = rng.choice([-1, 0, 1, 0.5, -0.5])

            if verify_numpy(U, V, W, tol=1e-9):
                return to_lists(U, V, W)

        return None

    # Try ranks from 99 down to 80
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if elapsed() > deadline - 10:
            break

        result = try_random_rank(target_rank, num_trials=20, seed=target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

    return best_result
# EVOLVE-BLOCK-END
