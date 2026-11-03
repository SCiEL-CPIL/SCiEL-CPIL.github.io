"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    # Build the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))

    T_target = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Function to check validity and snap to half-integers
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2

        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)

        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None

        # Convert back to lists
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]

        return {"U": U_out, "V": V_out, "W": W_out}

    # Optimization function with better initialization and more robust search
    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None

        np.random.seed(seed)

        # Multiple initialization strategies
        inits = []

        # Strategy 1: Random initialization
        U1 = np.random.randn(n * m, r) * 0.5
        V1 = np.random.randn(m * p, r) * 0.5
        W1 = np.random.randn(p * n, r) * 0.5
        inits.append((U1, V1, W1))

        # Strategy 2: Sparse initialization with some structure
        U2 = np.zeros((n * m, r))
        V2 = np.zeros((m * p, r))
        W2 = np.zeros((p * n, r))
        for col in range(r):
            # Randomly assign non-zero entries
            for _ in range(3):
                idx = np.random.randint(0, n * m)
                U2[idx, col] = np.random.choice([-1, 1])
            for _ in range(3):
                idx = np.random.randint(0, m * p)
                V2[idx, col] = np.random.choice([-1, 1])
            for _ in range(3):
                idx = np.random.randint(0, p * n)
                W2[idx, col] = np.random.choice([-1, 1])
        inits.append((U2, V2, W2))

        # Strategy 3: Block-structured initialization inspired by Strassen
        # For 2x5x5, we can think of it as 2x(5x5) which has structure
        U3 = np.zeros((n * m, r))
        V3 = np.zeros((m * p, r))
        W3 = np.zeros((p * n, r))
        # Initialize with some diagonal-like structure
        for col in range(min(r, n * m * m * p // 3)):
            i = col % n
            j = (col // n) % m
            k = (col // (n * m)) % p
            U3[i * m + j, col] = 1.0
            V3[j * p + k, col] = 1.0
            W3[k * n + i, col] = 1.0
        # Add some random perturbations
        U3 += np.random.randn(n * m, r) * 0.1
        V3 += np.random.randn(m * p, r) * 0.1
        W3 += np.random.randn(p * n, r) * 0.1
        inits.append((U3, V3, W2))  # Note: using W2 here is a bug, let me fix

        U3 = np.zeros((n * m, r))
        V3 = np.zeros((m * p, r))
        W3 = np.zeros((p * n, r))
        for col in range(min(r, n * m * m * p // 3)):
            i = col % n
            j = (col // n) % m
            k = (col // (n * m)) % p
            U3[i * m + j, col] = 1.0
            V3[j * p + k, col] = 1.0
            W3[k * n + i, col] = 1.0
        U3 += np.random.randn(n * m, r) * 0.1
        V3 += np.random.randn(m * p, r) * 0.1
        W3 += np.random.randn(p * n, r) * 0.1
        inits[2] = (U3, V3, W3)

        def loss(params):
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err

        best_result = None
        best_loss = np.inf

        for init_idx, (U_init, V_init, W_init) in enumerate(inits):
            if time.time() > deadline:
                break

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                # Use L-BFGS-B with more iterations
                res = minimize(loss, x0, method='L-BFGS-B',
                               options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12})
            except Exception:
                continue

            if res.fun < best_loss:
                best_loss = res.fun
                if res.fun < 1e-6:
                    size_U = (n * m) * r
                    size_V = (m * p) * r
                    U_sol = res.x[:size_U].reshape((n * m), r)
                    V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
                    W_sol = res.x[size_U + size_V:].reshape((p * n), r)

                    result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
                    if result:
                        return result
                    best_result = (U_sol, V_sol, W_sol)

        # If we didn't find an exact solution but got close, try polishing
        if best_result is not None and best_loss < 1e-3:
            U_sol, V_sol, W_sol = best_result
            x0 = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
            try:
                res = minimize(loss, x0, method='L-BFGS-B',
                               options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-14})
                if res.fun < 1e-8:
                    size_U = (n * m) * r
                    size_V = (m * p) * r
                    U_sol = res.x[:size_U].reshape((n * m), r)
                    V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
                    W_sol = res.x[size_U + size_V:].reshape((p * n), r)

                    result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
                    if result:
                        return result
            except Exception:
                pass

        return None

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find lower rank
    # Try ranks from 49 down to 30 with multiple seeds
    ranks_to_try = list(range(49, 29, -1))

    seeds = [42, 123, 999, 111, 222, 333, 444, 555, 666, 777, 888, 1000, 2000, 3000, 4000]

    for r in ranks_to_try:
        if time.time() > deadline:
            break
        found = False
        for seed in seeds:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result:
                best_result = result
                best_rank = r
                found = True
                break
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
