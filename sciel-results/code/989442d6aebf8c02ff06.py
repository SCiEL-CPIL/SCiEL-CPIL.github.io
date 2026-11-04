import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    def make_schoolbook():
        rank = n * m * p
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

    def make_strassen():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1

            # M1: (A00 + A11) * (B0,k0 + B1,k1)
            U[0 * 3 + 0][col] += 1
            U[1 * 3 + 1][col] += 1
            V[0 * 10 + k0][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            # M2: (A10 + A11) * B0,k0
            U[1 * 3 + 0][col] += 1
            U[1 * 3 + 1][col] += 1
            V[0 * 10 + k0][col] += 1
            W[k0 * 2 + 1][col] += 1
            W[k1 * 2 + 1][col] -= 1
            col += 1

            # M3: A00 * (B0,k1 - B1,k1)
            U[0 * 3 + 0][col] += 1
            V[0 * 10 + k1][col] += 1
            V[1 * 10 + k1][col] -= 1
            W[k1 * 2 + 0][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            # M4: A11 * (B1,k0 - B0,k0)
            U[1 * 3 + 1][col] += 1
            V[1 * 10 + k0][col] += 1
            V[0 * 10 + k0][col] -= 1
            W[k0 * 2 + 0][col] += 1
            W[k0 * 2 + 1][col] += 1
            col += 1

            # M5: (A00 + A01) * B1,k1
            U[0 * 3 + 0][col] += 1
            U[0 * 3 + 1][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] -= 1
            W[k1 * 2 + 0][col] += 1
            col += 1

            # M6: (A10 - A00) * (B0,k0 + B0,k1)
            U[1 * 3 + 0][col] += 1
            U[0 * 3 + 0][col] -= 1
            V[0 * 10 + k0][col] += 1
            V[0 * 10 + k1][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            # M7: (A01 - A11) * (B1,k0 + B1,k1)
            U[0 * 3 + 1][col] += 1
            U[1 * 3 + 1][col] -= 1
            V[1 * 10 + k0][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1
            col += 1

            # W00: A02 * B2,k0
            U[0 * 3 + 2][col] += 1
            V[2 * 10 + k0][col] += 1
            W[k0 * 2 + 0][col] += 1
            col += 1

            # W01: A02 * B2,k1
            U[0 * 3 + 2][col] += 1
            V[2 * 10 + k1][col] += 1
            W[k1 * 2 + 0][col] += 1
            col += 1

            # W10: A12 * B2,k0
            U[1 * 3 + 2][col] += 1
            V[2 * 10 + k0][col] += 1
            W[k0 * 2 + 1][col] += 1
            col += 1

            # W11: A12 * B2,k1
            U[1 * 3 + 2][col] += 1
            V[2 * 10 + k1][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        return True

    def try_reduce_rank(U, V, W, max_rank_reduction=5):
        """Attempt to reduce the rank by removing dependent tensor factors."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        current_rank = U_arr.shape[1]
        
        # Build the tensor in matrix form
        # T is of shape (n*m, m*p, p*n)
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        
        # Try to find a linear dependence among the columns
        # If there exists a vector c such that sum_r c[r] * (U[:,r] * V[:,r]) = 0
        # and sum_r c[r] * W[:,r] = 0, then we can remove that component.
        
        # This is equivalent to checking if the columns of the unfolded tensor are linearly dependent
        # in a specific way.
        
        # A simpler heuristic: check if we can express one column as a linear combination of others
        # by solving a least squares problem for the "outer product" vectors.
        
        # Let's try to find a non-trivial null space vector for the combined system.
        # We want c such that:
        # sum_r c[r] * U[:,r] * V[:,r] = 0  (vector of size n*m * m*p)
        # sum_r c[r] * W[:,r] = 0          (vector of size p*n)
        
        # This is a linear system. Let's build the matrix A where row i corresponds to a constraint.
        # Actually, it's easier to check the rank of the matrix formed by the columns of U, V, W.
        
        # Let's try a direct elimination: check if any column is redundant.
        # We can do this by checking if the tensor T can be represented with fewer terms.
        
        # A robust way is to use SVD or rank-revealing QR on the "mode-n" unfoldings.
        # But let's try a simpler iterative approach:
        # Check if we can find a vector c in the null space of the map c -> (U*V, W) tensor.
        
        # Let's form the matrix M of size (n*m*m*p + p*n) x current_rank
        # where the first n*m*m*p rows come from the outer product U*V, and the last p*n from W.
        
        # This might be too large if current_rank is large, but 55 is manageable.
        # n*m*m*p = 2*3*3*10 = 180. p*n = 20. Total 200 rows.
        
        M = np.zeros((n*m*m*p + p*n, current_rank))
        
        for r in range(current_rank):
            # Fill the U*V part
            outer = (U_arr[:, r] * V_arr[:, r]).flatten()
            M[:n*m*m*p, r] = outer
            # Fill the W part
            M[n*m*m*p:, r] = W_arr[:, r]
        
        # Find the null space of M
        U_null, S, Vh = np.linalg.svd(M, full_matrices=True)
        
        # The null space is spanned by the right singular vectors corresponding to zero singular values
        tol = 1e-10
        null_indices = np.where(S < tol)[0]
        
        if len(null_indices) == 0:
            return None
        
        # Take the first null space vector
        c = Vh[null_indices[0], :].conj().T
        
        # Check if c is truly in the null space
        residual = np.linalg.norm(M @ c)
        if residual > 1e-6:
            return None
        
        # Now we can use this dependence to eliminate one term.
        # However, eliminating a term from a tensor decomposition is not straightforward
        # because it's not just a linear combination of rows, it's a bilinear/trilinear structure.
        
        # If c is in the null space, it means that the combination of tensor factors is zero.
        # This implies that the rank can be reduced if we can rewrite the decomposition.
        
        # A simpler approach: just check if we can find a linear dependency among the columns
        # of U, V, or W individually that allows reduction.
        
        # Let's try a different tactic: use the fact that we have a specific structure.
        # The current decomposition is block-diagonal-ish (5 independent 2x2x2 blocks + 1x1x2 blocks).
        # There might be dependencies between the blocks.
        
        # For now, return None if we can't easily reduce.
        return None

    try:
        result = make_strassen()
        if result is not None and validate(result['U'], result['V'], result['W']):
            # Try to reduce rank
            reduced = try_reduce_rank(result['U'], result['V'], result['W'])
            if reduced is not None and validate(reduced['U'], reduced['V'], reduced['W']):
                return reduced
            return result
    except Exception:
        pass

    return make_schoolbook()
