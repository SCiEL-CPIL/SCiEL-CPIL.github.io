import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Build target tensor indices
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets.append((a, b, c))
    num_targets = len(targets)

    size_U = n * m
    size_V = m * p
    size_W = p * n

    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W (as lists of lists) form a valid decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for a, b, c in targets:
            val = U_arr[a] @ V_arr[b] @ W_arr[c]
            if abs(val - 1.0) > tol:
                return False
        # Check half-integer property
        for mat in [U_arr, V_arr, W_arr]:
            rounded = np.round(mat * 2) / 2
            if np.max(np.abs(mat - rounded)) > 1e-6:
                return False
        return True

    def try_random_search(R, num_trials=200):
        """Try to find a rank-R decomposition by random search with local refinement."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            # Initialize with random values in [-3, 3]
            x = np.random.uniform(-3, 3, R * (size_U + size_V + size_W))

            for step in range(50):
                if time.time() > deadline:
                    return None

                U = x[:size_U * R].reshape(size_U, R)
                V = x[size_U * R:size_U * R + size_V * R].reshape(size_V, R)
                W = x[size_U * R + size_V * R:].reshape(size_W, R)

                # Compute residuals and gradients
                res = np.zeros(num_targets)
                dU = np.zeros((size_U, R))
                dV = np.zeros((size_V, R))
                dW = np.zeros((size_W, R))

                for idx, (a, b, c) in enumerate(targets):
                    val = U[a] @ V[b] @ W[c]
                    res[idx] = val - 1.0

                    # Gradients
                    # d/dU[a,r] = V[b,r] * W[c,r]
                    # d/dV[b,r] = U[a,r] * W[c,r]
                    # d/dW[c,r] = U[a,r] * V[b,r]
                    vw = V[b] * W[c]
                    uw = U[a] * W[c]
                    uv = U[a] * V[b]

                    dU[a] += vw * res[idx]
                    dV[b] += uw * res[idx]
                    dW[c] += uv * res[idx]

                # Check if we've converged
                max_res = np.max(np.abs(res))
                if max_res < 1e-10:
                    # Found solution! Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2

                    # Verify rounded solution
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                    else:
                        # Rounded version doesn't work, but exact solution exists
                        # Try to nudge towards half-integers
                        for _ in range(20):
                            U = U + 0.01 * np.sign(U_r - U) * np.random.uniform(0, 0.5)
                            V = V + 0.01 * np.sign(V_r - V) * np.random.uniform(0, 0.5)
                            W = W + 0.01 * np.sign(W_r - W) * np.random.uniform(0, 0.5)
                            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

                # Gradient descent step with adaptive learning rate
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-15:
                    break

                lr = 0.1 / (1 + grad_norm * 0.1)
                x -= lr * grad

                # Occasionally add noise to escape local minima
                if trial % 10 == 0 and step > 10:
                    x += np.random.normal(0, 0.1, len(x))

        return None

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_solution = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook

    # Try ranks from high to low
    # Theoretical lower bounds suggest we might find solutions in the 20s-30s range
    for R in range(35, 19, -1):
        if time.time() > deadline:
            break

        remaining_time = deadline - time.time()
        # Allocate more trials for higher ranks (easier to find)
        num_trials = int(50 * remaining_time / 30)
        num_trials = max(10, min(200, num_trials))

        sol = try_random_search(R, num_trials=num_trials)
        if sol is not None:
            best_solution = sol
            best_rank = R
            # Don't break immediately; try lower ranks if time permits
            if R <= 22:
                break

    return best_solution
