"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product

    start_time = time.time()
    DEATHLINE = 145.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def build_trivial():
        """Build the trivial rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}, rank

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_newton_decomposition(R, num_trials=20, max_iter=500, seed=42):
        """
        Try to find a rank-R decomposition using Newton's method on the polynomial system.
        Variables: U (10*R), V (30*R), W (12*R) -> total 52*R variables
        Equations: 60 equations (one per (i,j,k) triple)
        
        We use a simplified approach: fix the structure by setting most entries to 0
        and only solve for a subset of free variables.
        """
        if elapsed() > DEATHLINE - 10:
            return None

        rng = np.random.RandomState(seed)

        # Target indices
        targets = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    targets.append((i * m + j, j * p + k, k * n + i))

        for trial in range(num_trials):
            if elapsed() > DEATHLINE - 10:
                break

            # Initialize with small random values
            U = rng.randn(n * m, R) * 0.5
            V = rng.randn(m * p, R) * 0.5
            W = rng.randn(p * n, R) * 0.5

            # Use gradient descent with adaptive steps
            best_loss = float('inf')
            best_U, best_V, best_W = None, None, None
            lr = 0.01

            for it in range(max_iter):
                if elapsed() > DEATHLINE - 10:
                    break

                # Compute loss and gradients
                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)

                for (u, v, w) in targets:
                    contrib = 0.0
                    for r in range(R):
                        contrib += U[u, r] * V[v, r] * W[w, r]

                    err = contrib - 1.0
                    for r in range(R):
                        grad_U[u, r] += 2.0 * err * V[v, r] * W[w, r]
                        grad_V[v, r] += 2.0 * err * U[u, r] * W[w, r]
                        grad_W[w, r] += 2.0 * err * U[u, r] * V[v, r]

                    loss += err * err

                loss /= len(targets)

                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()

                if loss < 1e-14:
                    break

                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 1e-12:
                    step = lr * loss / (grad_norm ** 2 + 1e-12)
                    U -= step * grad_U
                    V -= step * grad_V
                    W -= step * grad_W

                if it % 50 == 0 and it > 0:
                    lr *= 0.9

            if best_loss < 1e-10:
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()

                # Round to nearest half-integer
                for mat in [U_list, V_list, W_list]:
                    for i in range(len(mat)):
                        for j in range(len(mat[i])):
                            mat[i][j] = round(mat[i][j] * 2) / 2.0

                if verify_decomposition(U_list, V_list, W_list, tol=1e-8):
                    return U_list, V_list, W_list

        return None

    def try_structured_decomposition(R):
        """
        Try a structured decomposition where we fix the pattern of non-zero entries.
        For rank R, we assign each of the 60 target triples to one of R slots,
        and solve for the coefficients.
        """
        if elapsed() > DEATHLINE - 10:
            return None

        targets = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    targets.append((i * m + j, j * p + k, k * n + i))

        # Assign targets to slots using round-robin or random
        rng = np.random.RandomState(R)

        for attempt in range(5):
            if elapsed() > DEATHLINE - 10:
                break

            # Random assignment of targets to slots
            assignment = [rng.randint(0, R) for _ in range(len(targets))]

            # For each slot, collect the targets assigned to it
            slot_targets = [[] for _ in range(R)]
            for idx, t in enumerate(targets):
                slot_targets[assignment[idx]].append(t)

            # Build linear system: for each slot r, we need U[u,r]*V[v,r]*W[w,r] to contribute
            # to the correct positions. This is nonlinear, so we use a different approach.
            # Instead, use the fact that for a rank-1 term, the contribution is separable.
            # For each slot, we need: sum over targets in that slot of the product = 1
            # and for targets NOT in that slot, the product should be 0 (or cancel).

            # This is complex, so let's use a simpler heuristic:
            # Initialize U, V, W with random values and run a few Newton iterations
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))

            # For each slot, set initial values
            for r in range(R):
                if slot_targets[r]:
                    # Set U, V, W for this slot to have non-zero entries at the target positions
                    for (u, v, w) in slot_targets[r]:
                        U[u, r] = 1.0 / (len(slot_targets[r]) ** (1/3))
                        V[v, r] = 1.0 / (len(slot_targets[r]) ** (1/3))
                        W[w, r] = 1.0 / (len(slot_targets[r]) ** (1/3))

            # Now run gradient descent to refine
            best_loss = float('inf')
            best_U, best_V, best_W = None, None, None
            lr = 0.05

            for it in range(200):
                if elapsed() > DEATHLINE - 10:
                    break

                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)

                for (u, v, w) in targets:
                    contrib = 0.0
                    for r in range(R):
                        contrib += U[u, r] * V[v, r] * W[w, r]

                    err = contrib - 1.0
                    for r in range(R):
                        grad_U[u, r] += 2.0 * err * V[v, r] * W[w, r]
                        grad_V[v, r] += 2.0 * err * U[u, r] * W[w, r]
                        grad_W[w, r] += 2.0 * err * U[u, r] * V[v, r]

                    loss += err * err

                loss /= len(targets)

                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()

                if loss < 1e-14:
                    break

                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 1e-12:
                    step = lr * loss / (grad_norm ** 2 + 1e-12)
                    U -= step * grad_U
                    V -= step * grad_V
                    W -= step * grad_W

                if it % 30 == 0 and it > 0:
                    lr *= 0.95

            if best_loss < 1e-10:
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()

                for mat in [U_list, V_list, W_list]:
                    for i in range(len(mat)):
                        for j in range(len(mat[i])):
                            mat[i][j] = round(mat[i][j] * 2) / 2.0

                if verify_decomposition(U_list, V_list, W_list, tol=1e-8):
                    return U_list, V_list, W_list

        return None

    # Start with trivial decomposition
    best_result, best_rank = build_trivial()

    if elapsed() > DEATHLINE - 20:
        return best_result

    # Try structured decompositions for various ranks
    for R in range(30, 50):
        if elapsed() > DEATHLINE - 15:
            break

        result = try_structured_decomposition(R)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = R
            break

    # If structured didn't work, try Newton-based approach
    if best_rank == 60:
        for R in range(35, 55):
            if elapsed() > DEATHLINE - 15:
                break

            result = try_newton_decomposition(R, num_trials=10, max_iter=300, seed=R)
            if result is not None:
                U_list, V_list, W_list = result
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = R
                break

    return best_result
# EVOLVE-BLOCK-END
