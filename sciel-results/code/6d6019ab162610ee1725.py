"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 48

    # Known lower bounds and theoretical ranks
    # For matrix multiplication, the tensor rank is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But we need to find an exact decomposition.

    # For small matrices, we can try to find a decomposition with rank less than schoolbook.
    # The matrix multiplication tensor T_{i,j,k} for (n x m) * (m x p) has entries:
    # T[i,j,k] = sum_l A[i,l] * B[l,k]
    # In tensor form: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and the indices are:
    #   U[i*m + l, r]
    #   V[l*p + k, r]
    #   W[k*n + i, r]  (trace convention as specified)

    # The tensor T has entries T[(i, l), (l, k), (k, i)] = 1 if we consider the structure tensor.
    # Actually, the matrix multiplication tensor is:
    # M[i,j,k] = sum_l E[i,l] * F[l,k] where E is n x m, F is m x p
    # The structure tensor has entries T[i, l, k] = 1 for each triple (i, l, k)
    # Wait, let me be more careful.

    # The matrix multiplication map is:
    # (A, B) -> A*B where A is n x m, B is m x p
    # The tensor T such that (A*B)[i,k] = sum_{i,l,k} T[i,l,k] * A[i,l] * B[l,k]
    # So T[i,l,k] = 1 for all i in [0,n), l in [0,m), k in [0,p)
    # This is a tensor of shape (n, m, p) with all entries 1.

    # A rank-r decomposition means:
    # T[i,l,k] = sum_{r=0}^{r-1} U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]

    # We want to find the minimum r such that this holds.

    # For the 2x4x6 case, let's try to find a decomposition.
    # One approach: use the fact that we can combine some of the schoolbook terms.

    # Let's try a rank that is a multiple of something, or try specific values.
    # Known results: for 2x2x2, the rank is 7 (Strassen).
    # For larger matrices, we can use Strassen's algorithm recursively or other methods.

    # However, for exact decomposition with integer/half-integer entries, we need to be careful.

    # Let's try to construct a decomposition with rank < 48.
    # One simple improvement: if n, m, or p has structure, we can reduce rank.
    # For example, if we have a 1 x m x p case, the rank is m*p (no improvement possible).
    # For 2 x m x p, we might be able to do better.

    # Let's try a brute-force or structured approach for small ranks.
    # Given the time constraint, let's try a few candidate ranks.

    # Candidate ranks to try (from low to high, but we know rank >= max(n*m, m*p, p*n) = 24)
    # Actually, the border rank might be lower, but we need exact rank.

    # Let's try rank = 24, 25, ..., up to some limit.
    # But solving for exact integer/half-integer solutions is hard.

    # Alternative: use known Strassen-like decompositions.
    # Strassen's 2x2x2 uses rank 7 instead of 8.
    # We can pad/embed this into our 2x4x6 problem.

    # Let's think about it differently. The tensor T is all-ones of shape (2,4,6).
    # We can try to find a rank-r decomposition where U, V, W have entries in {0, 1/2, 1}.

    # Given the complexity, let's first return the schoolbook decomposition if we can't find better quickly.
    # But let's try a few structured approaches.

    # Approach 1: Try to use Kronecker products or block structures.
    # If we can write the 2x4x6 tensor as a sum of simpler tensors, we can reduce rank.

    # For now, let's implement a simple check: can we find a rank-r decomposition with r < 48?
    # We'll use a randomized approach with rounding to half-integers.

    def verify(U, V, W, rank):
        """Verify that U, V, W give an exact decomposition."""
        n, m, p = DIMENSIONS
        # Build the tensor from U, V, W
        # T[i,l,k] = sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
        # We want T[i,l,k] = 1 for all i, l, k
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank):
        """Try to find a rank-r decomposition with half-integer entries."""
        if time.time() > deadline - 10:
            return None

        n, m, p = DIMENSIONS
        U_dim = n * m
        V_dim = m * p
        W_dim = p * n

        # Use numpy for efficiency
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        # Random initialization with half-integers
        np.random.seed(42)
        for _ in range(100):  # Try 100 random initializations
            if time.time() > deadline - 5:
                break

            # Initialize with random half-integers in {-1, -0.5, 0, 0.5, 1}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            for r in range(rank):
                U[:, r] = np.random.choice(vals, size=U_dim)
                V[:, r] = np.random.choice(vals, size=V_dim)
                W[:, r] = np.random.choice(vals, size=W_dim)

            # Check if this works
            if verify(U.tolist(), V.tolist(), W.tolist(), rank):
                return U.tolist(), V.tolist(), W.tolist()

        return None

    # Try ranks from 24 up to schoolbook_rank - 1
    for rank in range(24, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # If no better rank found, return schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
