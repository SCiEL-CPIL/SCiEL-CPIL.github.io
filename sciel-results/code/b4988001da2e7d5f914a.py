"""
Search for a rank-minimal tensor decomposition of 3x4x7 matrix multiplication.

Approach:
  1. Construct the multiplication tensor T (indices (i,j,k), i in 0..2, j in 0..3, k in 0..6).
  2. The decomposition requires matrices U (12 x r), V (28 x r), W (21 x r) such that
     T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c].
  3. Use a CP-like alternating minimization / tensor-train style iterative method to find
     a low-rank approximation, then verify exactness.
  4. Known lower bound for M(3,4,7) is at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28,
     but the actual tensor rank is unknown. Schoolbook gives 84.
  5. We try to find a decomposition with rank < 84 using iterative optimization.
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # seconds; leave 15s margin for evaluator overhead

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84

    # Build the multiplication tensor T[i, j, k] = 1 for all valid (i,j,k)
    # Actually, the multiplication tensor for matrix multiplication has T[i,j,k] = 1
    # for all (i, j, k) in [n] x [m] x [p]. The tensor rank of M(n,m,p) is what we want.
    #
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # This is a 3-way tensor factorization with specific index mappings.

    # For the search, we'll use a randomized initialization + alternating least squares
    # on the constraint that T = sum_c U_c ⊗ V_c ⊗ W_c (with the specific index layouts).

    # Actually, let's think about this differently.
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We want to find rank r such that T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    # This is equivalent to finding matrices U, V, W such that the 3-way array
    # A[i, j, k] = T[i,j,k] equals sum_c U_flat[i,j,c] * V_flat[j,k,c] * W_flat[k,i,c]

    # Let's use a simple iterative method:
    # - Initialize U, V, W randomly
    # - Alternate: fix two, solve for the third via linear least squares
    # - This is similar to CP decomposition but with the specific index constraints.

    # However, due to the specific index mappings, a direct ALS is tricky.
    # Let's use a different approach: treat it as a non-convex optimization with
    # random restarts and use gradient-free methods or a simple greedy approach.

    # Actually, the most reliable approach within time limits:
    # 1. Return schoolbook if no better found.
    # 2. Try a small number of random rank-r candidates and check if they can exactly
    #    decompose T via solving linear systems.

    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,j,k): T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    # This is a system of n*m*p = 84 equations in (n*m + m*p + p*n) * r = (12+28+21)*r = 61r unknowns.
    # For r=84, we have 61*84 = 5124 unknowns and 84 equations - very underdetermined,
    # and the schoolbook solution works.

    # For r < 84, it's still underdetermined for r >= 2 (61*2 = 122 > 84).
    # The question is whether exact solutions exist.

    # Strategy: use random initialization and solve via iterative projection.
    # We'll try ranks from a lower bound up to 83, and for each, do a few random restarts
    # with alternating least squares on a relaxed problem, then check exactness.

    # Lower bound: rank >= max(n*m, m*p, p*n) is NOT correct.
    # The correct lower bounds are more complex. Let's just try a range.

    # Actually, let's use a known technique: the "border rank" or "tensor rank"
    # can be attacked via polynomial systems, but that's intractable.

    # Practical approach: use a random subspace method.
    # Pick random matrices V and W, then check if T is in the span of
    # {V_c ⊗ W_c} (with the index constraints), and if so, solve for U.

    # Let's implement a cleaner version:
    # For fixed V and W, the equations become linear in U:
    # For each (i,j,k): T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Group by (i,j): for fixed i,j, we have p equations (one for each k) in m... wait.
    # For fixed i,j: T[i,j,k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) for k=0..p-1
    # Let M[i,j] be a p x r matrix where M[i,j, k, c] = V[j*p+k, c] * W[k*n+i, c]
    # Then T[i,j,:] = M[i,j] @ U[i*m+j, :] (as a vector of length r)
    # Wait, T[i,j,:] is a vector of length p, and U[i*m+j, :] is length r.
    # M[i,j] is p x r, so M[i,j] @ U[i*m+j, :] is p x 1. That works!
    # So for fixed V, W, we can solve for U via least squares:
    # For each (i,j): U[i*m+j, :] = argmin || M[i,j] @ u - T[i,j,:] ||^2

    # Similarly, for fixed U, W, solve for V, and for fixed U, V, solve for W.

    # This is ALS! Let's implement it.

    def solve_U(V, W, T):
        """Solve for U given V, W."""
        U = np.zeros((n * m, r_global))
        for i in range(n):
            for j in range(m):
                # Build M[i,j]: p x r matrix
                # M[k, c] = V[j*p + k, c] * W[k*n + i, c]
                M = np.zeros((p, r_global))
                for k in range(p):
                    for c in range(r_global):
                        M[k, c] = V[j * p + k, c] * W[k * n + i, c]
                # Solve M @ u = T[i, j, :]
                # Use least squares
                u, _, _, _ = np.linalg.lstsq(M, T[i, j, :], rcond=None)
                U[i * m + j, :] = u
        return U

    def solve_V(U, W, T):
        """Solve for V given U, W."""
        V = np.zeros((m * p, r_global))
        for j in range(m):
            for k in range(p):
                # Build M[j,k]: n x r matrix
                # M[i, c] = U[i*m + j, c] * W[k*n + i, c]
                M = np.zeros((n, r_global))
                for i in range(n):
                    for c in range(r_global):
                        M[i, c] = U[i * m + j, c] * W[k * n + i, c]
                # Solve M @ v = T[:, j, k]
                t_vec = np.array([T[i, j, k] for i in range(n)])
                v, _, _, _ = np.linalg.lstsq(M, t_vec, rcond=None)
                V[j * p + k, :] = v
        return V

    def solve_W(U, V, T):
        """Solve for W given U, V."""
        W = np.zeros((p * n, r_global))
        for k in range(p):
            for i in range(n):
                # Build M[k,i]: m x r matrix
                # M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                M = np.zeros((m, r_global))
                for j in range(m):
                    for c in range(r_global):
                        M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                # Solve M @ w = T[i, :, k]
                t_vec = np.array([T[i, j, k] for j in range(m)])
                w, _, _, _ = np.linalg.lstsq(M, t_vec, rcond=None)
                W[k * n + i, :] = w
        return W

    def check_exact(U, V, W, T, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r_global):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - T[i, j, k]) > tol:
                        return False
        return True

    # Build T: all ones
    T = np.ones((n, m, p))

    best_rank = schoolbook_rank
    best_result = None

    # Try ranks from a reasonable lower bound up to schoolbook_rank - 1
    # Lower bound: at least max(n, m, p) = 7, but realistically much higher.
    # Let's try from 28 (m*p, which is the largest of the matrix dimensions products) up to 83.
    # Actually, let's start from a higher rank and work down, stopping early if time is low.

    for r_global in range(schoolbook_rank - 1, 27, -1):
        if time.time() - start_time > DEADLINE - 30:
            break

        # Try a few random restarts for this rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 15:
                break

            np.random.seed(42 + r_global * 1000 + restart)

            # Random initialization
            U = np.random.randn(n * m, r_global)
            V = np.random.randn(m * p, r_global)
            W = np.random.randn(p * n, r_global)

            # ALS iterations
            converged = False
            for it in range(50):
                if time.time() - start_time > DEADLINE - 10:
                    break
                U_new = solve_U(V, W, T)
                V_new = solve_V(U_new, W, T)
                W_new = solve_W(U_new, V_new, T)

                # Check convergence
                change = np.max(np.abs(W_new - W))
                U, V, W = U_new, V_new, W_new

                if change < 1e-12:
                    converged = True
                    break

            # Check exactness
            if check_exact(U, V, W, T):
                best_rank = r_global
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break

        if best_result is not None:
            break

    # If no better found, return schoolbook
    if best_result is None:
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
    else:
        return best_result
# EVOLVE-BLOCK-END
