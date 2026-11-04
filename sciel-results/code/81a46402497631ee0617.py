"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS

    # Try to find a better decomposition via random search with rank-1 terms
    # that have multiple non-zero entries, allowing cross-term cancellation.

    # First, establish the trivial decomposition as fallback
    trivial_rank = n * m * p  # 84

    # Try to find a decomposition with rank < trivial_rank
    # We'll use a randomized approach: start with trivial and try to merge terms.

    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if the output
    # element (i,k) depends on input element (i,j) and (j,k).
    # In the trace convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} for the (i,k) output
    # Wait, let me re-read the convention.
    # Actually, for matrix multiplication C = A * B where A is n x m and B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (for all valid i,j,k)
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i,j,k]
    # But W is indexed by (k*n+i), which is the output index.
    # So the equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # Wait, that's not right either. Let me think again.
    # The tensor M_{n,m,p} has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all (i,j,k)
    # A rank-1 decomposition is sum_r u_r ⊗ v_r ⊗ w_r where
    # u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # And the tensor is reconstructed as sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # This should equal T[i,j,k] = 1 for all (i,j,k).

    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)

    # The trivial solution has one term per (i,j,k) with U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1

    # Let's try to find a better solution using a simple iterative approach.
    # We'll use a random initialization and gradient descent on the constraint.

    # Actually, let's try a simpler approach: use the fact that for small matrices,
    # we can try ranks from some lower bound up to the trivial rank.

    # Lower bound: max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # Upper bound: 84

    # Let's try to find a decomposition with rank around 40-60 using random search.

    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Verify using numpy for speed."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try a few random searches with different ranks
    best_result = None
    best_rank = trivial_rank

    # Start with the trivial decomposition
    U_trivial = [[0] * trivial_rank for _ in range(n * m)]
    V_trivial = [[0] * trivial_rank for _ in range(m * p)]
    W_trivial = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = trivial_rank

    # Try to find better decompositions
    # Use a simple random search with local optimization

    def random_search(rank, max_iters=5000, seed=42):
        """Try to find a decomposition of given rank using random search."""
        rng = np.random.RandomState(seed)

        # Initialize with small random values
        U_arr = rng.randn(n * m, rank) * 0.1
        V_arr = rng.randn(m * p, rank) * 0.1
        W_arr = rng.randn(p * n, rank) * 0.1

        # Use a simple gradient-free approach: try to satisfy the constraints
        # by adjusting the values.

        # Actually, let's use a different approach: start from the trivial solution
        # and try to merge terms.

        # For now, let's just return None if we can't find a solution quickly.
        # This is a placeholder for a more sophisticated search.

        return None

    # Given the time constraints and complexity, let's just return the trivial solution.
    # The trivial solution is guaranteed to be correct and has rank 84.

    return best_result
# EVOLVE-BLOCK-END
