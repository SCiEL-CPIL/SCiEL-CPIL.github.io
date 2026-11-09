"""Matrix multiplication 2x5 by 5x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try to find a lower-rank decomposition using structured search
    # The tensor T has entries T[(i,j,k)] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We seek rank r such that T = sum_{t=1}^{r} u_t v_t^T w_t^T
    # where u_t in R^n, v_t in R^m, w_t in R^p
    #
    # Key insight: For small dimensions, we can use the fact that the matrix
    # multiplication tensor has a known structure. For 2x5x6, the known
    # rank is 14 (from literature). Let's try to find a rank-14 decomposition.
    #
    # Approach: Use a structured parameterization. Since n=2 is small,
    # we can enumerate certain structures.
    #
    # A rank-r decomposition means:
    # For all i in [n], j in [m], k in [p]:
    #   sum_{t=1}^{r} U[i,j,t] * V[j,k,t] * W[k,i,t] = 1
    #
    # Let's try a decomposition where the W factors have a specific structure.
    # Since n=2, W[k,i,t] for i=0,1 gives 2 values per (k,t).
    #
    # Strategy: Use a bilinear decomposition approach.
    # For each t, define matrices A_t (n x m), B_t (m x p), C_t (p x n)
    # such that sum_t A_t ⊗ B_t ⊗ C_t = T
    #
    # Actually, the standard formulation:
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # U[i*m+j, t] = a_t[i,j]
    # V[j*p+k, t] = b_t[j,k]
    # W[k*n+i, t] = c_t[k,i]
    # Constraint: for all i,j,k: sum_t a_t[i,j] * b_t[j,k] * c_t[k,i] = 1

    # Try rank 14 with a structured approach
    # Use the fact that for 2x5x6, we can try decompositions based on
    # known algorithms or structured guesses.

    # Let's try a more systematic approach: random search with verification
    # for small ranks, trying to find integer/half-integer solutions.

    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition exactly reproduces the tensor."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(mat):
        """Check if all entries are integers or half-integers (i.e., 2*x is integer)."""
        for row in mat:
            for x in row:
                if abs(2 * x - round(2 * x)) > 1e-9:
                    return False
        return True

    # Try ranks from 14 up to schoolbook
    for r in range(14, rank_schoolbook + 1):
        if time.time() > deadline - 5.0:
            break

        # For rank 14, try a structured approach
        # Use a random initialization with structured constraints
        # and try to solve via iterative refinement

        # Random search approach: initialize with random half-integer values
        # and check if they work. This is unlikely to work directly,
        # so we use a more structured approach.

        # Structured approach for rank 14:
        # Based on known results, try a decomposition where
        # the factors have a specific pattern.

        # For now, let's try a few random structured initializations
        # with a local optimization approach.

        if r == 14:
            # Try a specific structured decomposition
            # This is based on the idea that for 2x5x6, there exists
            # a rank-14 decomposition with specific structure.

            # Use a parameterized family and search for parameters
            # that satisfy the constraints.

            # One approach: fix some entries and solve for others.
            # Since we have 2*5*6 = 60 constraints and 14*(2*5 + 5*6 + 6*2) = 14*32 = 448 unknowns,
            # the system is underdetermined, so solutions should exist.

            # Let's try a random approach: initialize with small random half-integers
            # and use a Newton-like method to drive the residuals to zero.

            for attempt in range(50):
                if time.time() > deadline - 5.0:
                    break

                np.random.seed(attempt * 12345 + 67890)

                # Initialize with random values in {-1, -0.5, 0, 0.5, 1}
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                U = [[np.random.choice(vals) for _ in range(r)] for _ in range(n * m)]
                V = [[np.random.choice(vals) for _ in range(r)] for _ in range(m * p)]
                W = [[np.random.choice(vals) for _ in range(r)] for _ in range(p * n)]

                # Convert to numpy for computation
                U_np = np.array(U, dtype=np.float64)
                V_np = np.array(V, dtype=np.float64)
                W_np = np.array(W, dtype=np.float64)

                # Compute residuals
                # residual[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1
                # We want to minimize sum of squares of residuals.

                # Use a simple gradient-free approach: perturb one variable at a time
                # and keep the change if it reduces the total residual.

                def compute_residual_sum(U_arr, V_arr, W_arr):
                    total = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for t in range(r):
                                    s += U_arr[i * m + j, t] * V_arr[j * p + k, t] * W_arr[k * n + i, t]
                                total += (s - 1.0) ** 2
                    return total

                current_res = compute_residual_sum(U_np, V_np, W_np)

                # Try to improve by local search
                improved = True
                max_iters = 200
                iters = 0
                while improved and iters < max_iters:
                    if time.time() > deadline - 5.0:
                        break
                    improved = False
                    iters += 1

                    # Try perturbing each variable
                    for idx in range(n * m * r + m * p * r + p * n * r):
                        if time.time() > deadline - 5.0:
                            break

                        # Determine which matrix and element
                        if idx < n * m * r:
                            mat = U_np
                            row = idx // r
                            col = idx % r
                            mat_name = 'U'
                        elif idx < n * m * r + m * p * r:
                            idx2 = idx - n * m * r
                            mat = V_np
                            row = idx2 // r
                            col = idx2 % r
                            mat_name = 'V'
                        else:
                            idx2 = idx - n * m * r - m * p * r
                            mat = W_np
                            row = idx2 // r
                            col = idx2 % r
                            mat_name = 'W'

                        old_val = mat[row, col]
                        for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                            if abs(new_val - old_val) < 1e-9:
                                continue
                            mat[row, col] = new_val
                            new_res = compute_residual_sum(U_np, V_np, W_np)
                            if new_res < current_res - 1e-12:
                                current_res = new_res
                                improved = True
                                break
                            else:
                                mat[row, col] = old_val

                if current_res < 1e-12:
                    # Found a solution! Convert back to lists
                    U_list = U_np.tolist()
                    V_list = V_np.tolist()
                    W_list = W_np.tolist()

                    # Round to half-integers
                    U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
                    V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
                    W_list = [[round(x * 2) / 2 for x in row] for row in W_list]

                    # Verify
                    if verify_decomposition(U_list, V_list, W_list, r):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        return best_result

    return best_result
# EVOLVE-BLOCK-END
