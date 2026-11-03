import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    U_best, V_best, W_best = get_schoolbook()
    best_rank = len(U_best[0])

    def verify(U, V, W, tol=1e-8):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)

        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    total = np.dot(U_arr[ui], V_arr[vk] * W_arr[wk])
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def objective_flat(x, R):
        """Objective: sum of squared residuals for the tensor decomposition."""
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:3*n*m*R].reshape(p*n, R)

        residual_sum = 0.0
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    val = np.dot(U[ui], V[vk] * W[wk])
                    residual_sum += (val - 1.0) ** 2
        return residual_sum

    def try_optimize(R, max_trials=5, method='L-BFGS-B'):
        """Try to find a rank-R decomposition using optimization."""
        if time.time() > deadline - 15.0:
            return None

        total_vars = 3 * n * m * R  # n*m = m*p = p*n = 25 for 5x5x5
        best_solution = None
        best_obj = float('inf')

        for trial in range(max_trials):
            if time.time() > deadline - 15.0:
                break

            # Initialize with random values
            x0 = np.random.randn(total_vars) * 0.1

            # Add some structure to help convergence
            if trial % 3 == 0:
                # Block-diagonal initialization
                x0 = np.zeros(total_vars)
                U_part = x0[:n*m*R].reshape(n*m, R)
                V_part = x0[n*m*R:2*n*m*R].reshape(m*p, R)
                W_part = x0[2*n*m*R:].reshape(p*n, R)

                block = max(R // 5, 1)
                for i in range(n):
                    for j in range(m):
                        for r in range(block * min(i+j, 4), block * min(i+j+1, 5)):
                            if r < R:
                                U_part[i*m+j, r] = random.uniform(-0.5, 0.5)
                for j in range(m):
                    for k in range(p):
                        for r in range(block * min(j+k, 4), block * min(j+k+1, 5)):
                            if r < R:
                                V_part[j*p+k, r] = random.uniform(-0.5, 0.5)
                for k in range(p):
                    for i in range(n):
                        for r in range(block * min(k+i, 4), block * min(k+i+1, 5)):
                            if r < R:
                                W_part[k*n+i, r] = random.uniform(-0.5, 0.5)

            elif trial % 3 == 1:
                # Sparse initialization
                x0 = np.random.randn(total_vars)
                mask = np.random.random(total_vars) < 0.3
                x0 = x0 * mask
                x0[mask] *= 2  # Scale up non-zero entries

            else:
                # Integer-like initialization
                x0 = np.random.choice([-1, -0.5, 0, 0.5, 1], size=total_vars)

            from scipy.optimize import minimize
            result = minimize(
                objective_flat,
                x0,
                args=(R,),
                method=method,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )

            if result.fun < best_obj:
                best_obj = result.fun
                best_solution = result.x

            # Check if we found a valid solution
            if result.fun < 1e-6:
                x = result.x
                U = x[:n*m*R].reshape(n*m, R).tolist()
                V = x[n*m*R:2*n*m*R].reshape(m*p, R).tolist()
                W = x[2*n*m*R:].reshape(p*n, R).tolist()

                # Snap to half-integers
                U_snapped = [[round(u * 2) / 2 for u in row] for row in U]
                V_snapped = [[round(v * 2) / 2 for v in row] for row in V]
                W_snapped = [[round(w * 2) / 2 for w in row] for row in W]

                if verify(U_snapped, V_snapped, W_snapped):
                    return U_snapped, V_snapped, W_snapped

        return None

    # Try decreasing ranks from 124 down to known bounds
    # Known: Strassen gives rank 7 for 2x2x2, so 5x5x5 should be much less than 125
    # Try a range of ranks
    for R in range(120, 20, -1):
        if time.time() > deadline - 15.0:
            break

        result = try_optimize(R, max_trials=3)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = R
            print(f"Found rank {R} decomposition!")
            return {"U": U_best, "V": V_best, "W": W_best}

    # If no improvement found, return schoolbook
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
