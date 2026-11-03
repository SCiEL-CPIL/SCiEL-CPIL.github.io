"""Tensor decomposition search for 3x5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank_school = n * m * p  # 75

    # Start with schoolbook decomposition
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    best_rank = rank_school

    # Try to find a lower rank decomposition using random search and local optimization
    # The tensor condition is: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    def compute_error(U, V, W, target_rank):
        """Compute the error in the tensor decomposition."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (s - 1.0) ** 2
        return error

    # Random search for lower rank
    random.seed(42)
    np.random.seed(42)

    # Try ranks from schoolbook down to a reasonable minimum
    # For 3x5x5, the theoretical lower bound is around 3*5*5 - something, but we'll search
    # Let's try to find a decomposition with rank < 75

    # Strategy: Start with schoolbook, then try to merge/eliminate terms
    # Use a greedy approach: try to remove one term at a time and see if we can adjust others

    # First, let's try a simple random initialization and optimization
    # For each candidate rank, initialize randomly and try to optimize

    max_attempts = 20
    time_limit = 90  # seconds

    start_time = time.time()

    for target_rank in range(rank_school - 1, 50, -1):
        if time.time() - start_time > time_limit:
            break

        # Try a few random initializations for this rank
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break

            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1

            # Add some structure: try to preserve some of the schoolbook terms
            # Pick a subset of the schoolbook terms
            indices = random.sample(range(rank_school), min(target_rank, rank_school))
            for c, idx in enumerate(indices):
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0

            # Use scipy minimize to reduce the error
            def objective(params):
                U_flat = params[:n * m * target_rank].reshape(n * m, target_rank)
                V_flat = params[n * m * target_rank:n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                W_flat = params[n * m * target_rank + m * p * target_rank:].reshape(p * n, target_rank)
                return compute_error(U_flat, V_flat, W_flat, target_rank)

            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            result = minimize(objective, params0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-6})

            if result.success and result.fun < 1e-6:
                # Found a valid decomposition
                U_opt = result.x[:n * m * target_rank].reshape(n * m, target_rank)
                V_opt = result.x[n * m * target_rank:n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                W_opt = result.x[n * m * target_rank + m * p * target_rank:].reshape(p * n, target_rank)

                # Check if entries are close to integers or half-integers
                # For now, just accept it
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U_opt.copy()
                    best_V = V_opt.copy()
                    best_W = W_opt.copy()
                    break  # Move to next lower rank

    # Convert to lists of lists with integer/half-integer entries if possible
    # For the final answer, round to nearest integer or half-integer
    def round_to_half_int(x):
        """Round to nearest integer or half-integer."""
        return round(x * 2) / 2

    U_final = np.array([[round_to_half_int(x) for x in row] for row in best_U])
    V_final = np.array([[round_to_half_int(x) for x in row] for row in best_V])
    W_final = np.array([[round_to_half_int(x) for x in row] for row in best_W])

    # Verify the decomposition
    error = compute_error(U_final, V_final, W_final, best_rank)
    if error > 1e-4:
        # Fallback to schoolbook
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
