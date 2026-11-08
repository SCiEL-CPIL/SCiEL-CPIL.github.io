"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 32

    # We are looking for a tensor decomposition of the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: T = sum_{c=1}^{rank} U[:, c] * V[:, c] * W[:, c]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The score is the rank. We want to minimize rank.

    # Known lower bound for 2x4x4 matrix multiplication:
    # The tensor rank is at least max(n*m, m*p, p*n) = 16.
    # Actually, for matrix multiplication, the rank is at least the max of the dimensions.
    # There is a result by Strassen that the rank of 2x2x2 is 7.
    # For 2x4x4, we can potentially do better than 32.
    # Let's try to find a decomposition with rank < 32.

    # We can use a heuristic search. Since the tensor is small, we can try to find a low-rank decomposition.
    # The tensor has 2*4*4 = 32 non-zero entries.
    # Let's try rank 16 first (the theoretical lower bound for this shape is likely higher, but let's see).
    # Actually, the rank of the tensor is the minimum number of rank-1 tensors that sum to it.

    # Let's use a randomized initialization and gradient descent to minimize the residual.
    # If we can find a decomposition with rank R, we return it.

    # We'll try ranks from 16 upwards.
    # For each rank, we'll try several random initializations.

    def residual(rank, U, V, W):
        # Compute the tensor from the decomposition
        # T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
        # We need to match T[i*m+j, j*p+k, k*n+i] = 1
        # Let's construct the target tensor T_target of shape (n*m, m*p, p*n)
        # T_target[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent, else 0.
        # Actually, the tensor is defined by T[i, j, k] = 1 if j is the shared index.
        # In the flattened form:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # So for a given column c, the contribution is U[:, c] * V[:, c] * W[:, c] (outer product)
        # The sum of these outer products should equal the target tensor.

        # Let's compute the sum of outer products.
        # This is a tensor of shape (n*m, m*p, p*n)
        # We can do this efficiently using einsum or loops.
        # Since the dimensions are small, loops are fine.

        T = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])

        # Now compare with the target tensor.
        # The target tensor has 1s at positions where the indices are consistent.
        # Let's create the target tensor.
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1

        # The residual is the difference between T and T_target.
        # We want to minimize the sum of squares of the residual.
        # But we also need to ensure that the entries are close to 0 or 1 (or half-integers).
        # For now, let's just minimize the sum of squares.
        diff = T - T_target
        return np.sum(diff**2)

    # We'll use a simple local search.
    # For a given rank, we'll initialize U, V, W randomly and then use gradient descent.
    # To make it faster, we'll use the fact that the tensor is sparse.

    # Let's try to find a decomposition with rank 16.
    # If that fails, try 17, 18, etc.

    # We'll use a time limit for each rank.
    time_per_rank = 10  # seconds

    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None

    # Initialize with the schoolbook decomposition
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W

    # Try to find a lower rank
    for rank in range(16, rank_schoolbook):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Try several random initializations
        for trial in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break

            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)

            # Use gradient descent to minimize the residual
            # We'll use a simple SGD approach
            lr = 0.01
            for step in range(1000):
                if time.time() - start_time > DEADLINE - 5:
                    break

                # Compute gradient
                # d/dU[i, c] = sum_{j, k} (T[i, j, k] - T_target[i, j, k]) * V[j, c] * W[k, c]
                # This is expensive to compute directly. Let's use a simpler approach.
                # We'll compute the residual and then the gradient using finite differences or a more efficient method.
                # For now, let's just do a simple update.

                # Actually, let's use the fact that the tensor is small and compute the gradient directly.
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)

                # Compute T = sum_c U[:, c] * V[:, c] * W[:, c]
                T = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])

                # Compute the difference
                T_target = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i*m+j, j*p+k, k*n+i] = 1

                diff = T - T_target

                # Compute gradients
                # d/dU[i, c] = sum_{j, k} diff[i, j, k] * V[j, c] * W[k, c]
                for c in range(rank):
                    grad_U[:, c] = np.einsum('ijk,jc,kc->ic', diff, V[:, c], W[:, c])
                    grad_V[:, c] = np.einsum('ijk,ic,kc->jc', diff, U[:, c], W[:, c])
                    grad_W[:, c] = np.einsum('ijk,ic,jc->kc', diff, U[:, c], V[:, c])

                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

            # Check the residual
            res = residual(rank, U, V, W)
            if res < 1e-6:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break

    # Return the best result found
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
