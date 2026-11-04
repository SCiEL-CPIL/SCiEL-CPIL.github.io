"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    The decomposition must have integer or half-integer entries.
    Returns: {'U': (12, rank), 'V': (20, rank), 'W': (15, rank)}
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    total_terms = U_rows * V_rows * W_rows  # 3*4*5 * 4*5 * 5*3 = 60 * 20 * 15 = 18000 terms

    start_time = time.time()
    deadline = 135.0  # 15s safety margin

    # Start with the schoolbook decomposition (rank = n*m*p = 60)
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(U_rows)]
    V = [[0] * rank_school for _ in range(V_rows)]
    W = [[0] * rank_school for _ in range(W_rows)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U, V, W

    # Build the tensor: T[i,j,k] corresponds to U[i*m+j], V[j*p+k], W[k*n+i]
    # The decomposition satisfies:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T is the matrix multiplication tensor.

    # For 3x4*4x5, the tensor has a specific structure.
    # Let's try to find a lower-rank decomposition using a search approach.
    #
    # Strategy: Use the known structure of small matrix multiplication tensors.
    # For 3x4*4x5, we can try to find a decomposition of rank < 60.
    #
    # The tensor can be written as:
    #   T = sum_{i,j,k} e_i^U * e_j^V * e_k^W
    # where e_i^U has 1 at position i*m+j, etc.
    #
    # A lower-rank decomposition would group these terms.
    # Let's try a randomized search for a lower-rank decomposition.

    # First, let's verify the schoolbook decomposition works
    # Then try to find a lower-rank one

    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The tensor T has shape (n*m, m*p, p*n) and T[i*m+j, j*p+k, k*n+i] = 1
    # Actually, the tensor is T[a, b, c] where a indexes U, b indexes V, c indexes W
    # T[a, b, c] = 1 if a, b, c are consistent (i.e., there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i), else 0.

    # Let's build the tensor
    T = np.zeros((U_rows, V_rows, W_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    # Now we want to find rank-r matrices U, V, W such that
    #   sum_c U[a,c] * V[b,c] * W[c_index, c] = T[a,b,c]
    # where c_index is the row index of W.

    # This is a non-convex problem. Let's try a few approaches:
    # 1. Try to find a decomposition with rank < 60
    # 2. Use random initialization and gradient descent (but this is non-convex)
    # 3. Use the known lower bounds and try to find a matching decomposition

    # For now, let's try a simple approach: try to find a lower-rank decomposition
    # by randomly initializing U, V, W with small entries and trying to fit the tensor.

    # Actually, let's think about this differently. The tensor for matrix multiplication
    # has a special structure. For 3x4*4x5, the minimal rank is known to be...
    # Let's just try to find a good decomposition.

    # One approach: use the fact that the tensor is "structured" and try to find
    # a decomposition that exploits this structure.

    # Let's try a randomized search:
    # 1. Start with a random initialization of U, V, W with rank r
    # 2. Use coordinate descent or gradient-based methods to fit the tensor
    # 3. Check if the result has integer/half-integer entries

    # But this is a non-convex problem and may not converge to a good solution.

    # Alternative: use the known structure of small matrix multiplication tensors.
    # For 2x2*2x2, the minimal rank is 7 (Strassen).
    # For 3x3*3x3, the minimal rank is 23.
    # For 3x4*4x5, the minimal rank is... let's try to find it.

    # Let's try a different approach: use the fact that the tensor can be written
    # as a sum of rank-1 tensors, and try to find a sparse decomposition.

    # Actually, let's just try to find a decomposition with rank as low as possible
    # using a heuristic approach.

    # Heuristic: Start with the schoolbook decomposition and try to merge terms.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists a linear
    # combination that preserves the tensor structure.

    # This is complex. Let's try a simpler approach: use a randomized search
    # with a fixed number of iterations.

    # Let's try to find a decomposition with rank < 60 using a randomized approach.
    # We'll use a random initialization and try to fit the tensor using least squares.

    # But the constraint is that the entries must be integer or half-integer.
    # This is a hard constraint.

    # Let's try a different approach: use the known lower bound for the rank of
    # the matrix multiplication tensor. For 3x4*4x5, the rank is at least...
    # The rank is at least n*m*p / max(n, m, p) = 60 / 5 = 12 (by the cut bound).
    # But this is a weak lower bound.

    # Let's just try to find a good decomposition and see what rank we can achieve.

    # Approach: Use a randomized search with a time limit.
    # 1. Try different ranks starting from a lower bound
    # 2. For each rank, try a few random initializations
    # 3. Use a local search to improve the decomposition
    # 4. Check if the entries are close to integer or half-integer

    # Let's implement this.

    def check_integer_half_integer(arr, tol=1e-6):
        """Check if all entries are close to integer or half-integer."""
        for val in arr:
            if val is None:
                return False
            val = float(val)
            # Check if val is close to an integer or half-integer
            int_part = int(val)
            half_part = int_part + 0.5
            if abs(val - int_part) < tol or abs(val - half_part) < tol:
                pass
            else:
                return False
        return True

    def compute_residual(U_mat, V_mat, W_mat, T_tensor):
        """Compute the residual of the tensor decomposition."""
        # U_mat: (U_rows, rank), V_mat: (V_rows, rank), W_mat: (W_rows, rank)
        # T_tensor: (U_rows, V_rows, W_rows)
        # Residual[a,b,c] = sum_c U[a,c] * V[b,c] * W[c_index, c] - T[a,b,c]
        rank = U_mat.shape[1]
        residual = np.zeros_like(T_tensor)
        for c in range(rank):
            residual += np.einsum('a,b->abc', U_mat[:, c], V_mat[:, c]) * W_mat[:, c]
        residual -= T_tensor
        return residual

    def objective(U_mat, V_mat, W_mat, T_tensor):
        """Compute the objective function (sum of squared residuals)."""
        residual = compute_residual(U_mat, V_mat, W_mat, T_tensor)
        return np.sum(residual ** 2)

    # Try to find a lower-rank decomposition
    # Start from a rank below the schoolbook rank
    for target_rank in range(50, 10, -1):
        if time.time() > deadline - 30:
            break

        # Try a few random initializations
        for init in range(3):
            if time.time() > deadline - 20:
                break

            # Random initialization with small values
            np.random.seed(init + target_rank * 1000)
            U_mat = np.random.randn(U_rows, target_rank) * 0.1
            V_mat = np.random.randn(V_rows, target_rank) * 0.1
            W_mat = np.random.randn(W_rows, target_rank) * 0.1

            # Try to fit the tensor using a simple local search
            # We'll use a coordinate descent approach
            best_obj = objective(U_mat, V_mat, W_mat, T)
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break

                # Try to update U
                for c in range(target_rank):
                    # Fix V[:, c] and W[:, c], optimize U[:, c]
                    # residual = T - sum_{c' != c} U[:, c'] * V[:, c'] * W[:, c']
                    # We want to minimize ||U[:, c] * (V[:, c] * W[:, c]) - residual||^2
                    # This is a linear least squares problem
                    # But the product V[:, c] * W[:, c] is not a matrix, it's a tensor product

                    # Actually, this is more complex. Let's skip the coordinate descent
                    # and just use a simple approach: random perturbations.

                    # Try a random perturbation of U[:, c]
                    U_old = U_mat[:, c].copy()
                    U_mat[:, c] += np.random.randn(U_rows) * 0.01
                    obj = objective(U_mat, V_mat, W_mat, T)
                    if obj < best_obj:
                        best_obj = obj
                    else:
                        U_mat[:, c] = U_old

                if time.time() > deadline - 10:
                    break

                # Try to update V
                for c in range(target_rank):
                    V_old = V_mat[:, c].copy()
                    V_mat[:, c] += np.random.randn(V_rows) * 0.01
                    obj = objective(U_mat, V_mat, W_mat, T)
                    if obj < best_obj:
                        best_obj = obj
                    else:
                        V_mat[:, c] = V_old

                if time.time() > deadline - 10:
                    break

                # Try to update W
                for c in range(target_rank):
                    W_old = W_mat[:, c].copy()
                    W_mat[:, c] += np.random.randn(W_rows) * 0.01
                    obj = objective(U_mat, V_mat, W_mat, T)
                    if obj < best_obj:
                        best_obj = obj
                    else:
                        W_mat[:, c] = W_old

                if time.time() > deadline - 10:
                    break

            # Check if the result has integer/half-integer entries
            # and if the residual is small
            if best_obj < 1e-6:
                # Check if entries are close to integer or half-integer
                if (check_integer_half_integer(U_mat, 1e-4) and
                        check_integer_half_integer(V_mat, 1e-4) and
                        check_integer_half_integer(W_mat, 1e-4)):
                    # Round to nearest integer or half-integer
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2

                    # Verify the rounded solution
                    residual_rounded = compute_residual(U_rounded, V_rounded, W_rounded, T)
                    if np.max(np.abs(residual_rounded)) < 1e-4:
                        # Found a valid decomposition
                        best_rank = target_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        break

        if best_rank == target_rank:
            break

    # Convert to lists of integers or half-integers
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()

    # Ensure the entries are exactly integer or half-integer
    def round_to_half(x):
        return round(x * 2) / 2

    U_final = [[round_to_half(x) for x in row] for row in U_final]
    V_final = [[round_to_half(x) for x in row] for row in V_final]
    W_final = [[round_to_half(x) for x in row] for row in W_final]

    # Convert to integers if possible (multiply by 2 and check if integer)
    # Actually, the problem says "integer/half-integer entries", so we can keep them as floats
    # or convert to a common representation.

    # Let's keep them as floats but ensure they are exactly representable
    # 0.5 is exactly representable in binary floating point.

    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
