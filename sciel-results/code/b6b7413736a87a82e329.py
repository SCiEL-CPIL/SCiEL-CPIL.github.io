import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen decomposition using structured coefficients
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]

    def strassen_block():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0
        for a in range(2):  # A block column: 2a, 2a+1
            for b in range(4):  # B column block: 2b, 2b+1
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1

        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def error_metric(U, V, W, rank):
        """Compute sum of squared errors for all tensor entries."""
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank] * V[j*p+k, :rank], W[k*n+i, :rank])
                    total_err += (val - 1.0) ** 2
        return total_err

    def optimize_rank(U_init, V_init, W_init, target_rank, max_iters=3000, lr=0.05, seed=None):
        """
        Gradient-based optimization to find a valid decomposition at target_rank.
        Uses smooth surrogate: minimize sum of squared errors.
        """
        if time.time() > deadline - 15:
            return None

        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))

        # Initialize: use first target_rank columns from best solution
        init_cols = min(target_rank, U_init.shape[1])
        U[:, :init_cols] = U_init[:, :init_cols]
        V[:, :init_cols] = V_init[:, :init_cols]
        W[:, :init_cols] = W_init[:, :init_cols]

        # If we're reducing rank, add small random perturbation to break symmetry
        if target_rank < init_cols:
            U[:, init_cols:] = np.random.randn(nm, target_rank - init_cols) * 0.1
            V[:, init_cols:] = np.random.randn(mp, target_rank - init_cols) * 0.1
            W[:, init_cols:] = np.random.randn(pn, target_rank - init_cols) * 0.1

        # Compute initial error
        current_err = error_metric(U, V, W, target_rank)

        # Gradient descent with adaptive learning rate
        best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), current_err

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m+j, :]
                        v_row = V[j*p+k, :]
                        w_row = W[k*n+i, :]
                        val = np.dot(u_row * v_row, w_row)
                        err = val - 1.0

                        # d/dU: err * V[j*p+k,:] * W[k*n+i,:]
                        dU[i*m+j, :] += err * v_row * w_row
                        # d/dV: err * U[i*m+j,:] * W[k*n+i,:]
                        dV[j*p+k, :] += err * u_row * w_row
                        # d/dW: err * U[i*m+j,:] * V[j*p+k,:]
                        dW[k*n+i, :] += err * u_row * v_row

            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            new_err = error_metric(U, V, W, target_rank)

            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Adaptive learning rate
            if new_err > current_err * 1.5:
                lr *= 0.5
            elif it % 100 == 0 and it > 0:
                lr *= 0.95

            current_err = new_err

            if best_err < 1e-10:
                break

        if best_err < 1e-9:
            return best_U, best_V, best_W, target_rank
        return None

    def optimize_with_restarts(U_init, V_init, W_init, target_rank, n_restarts=5, max_iters=2000, lr=0.05):
        """
        Try multiple random restarts to find a valid decomposition at target_rank.
        """
        if time.time() > deadline - 15:
            return None

        best_result = None
        best_err = float('inf')

        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break

            # Create perturbed initialization
            U_pert = U_init[:, :min(target_rank, U_init.shape[1])].copy()
            V_pert = V_init[:, :min(target_rank, V_init.shape[1])].copy()
            W_pert = W_init[:, :min(target_rank, W_init.shape[1])].copy()

            if target_rank < U_init.shape[1]:
                U_pert = np.zeros((nm, target_rank))
                V_pert = np.zeros((mp, target_rank))
                W_pert = np.zeros((pn, target_rank))
                U_pert[:, :min(target_rank, U_init.shape[1])] = U_init[:, :min(target_rank, U_init.shape[1])]
                V_pert[:, :min(target_rank, V_init.shape[1])] = V_init[:, :min(target_rank, V_init.shape[1])]
                W_pert[:, :min(target_rank, W_init.shape[1])] = W_init[:, :min(target_rank, W_init.shape[1])]
                # Add random noise to break symmetry
                U_pert += np.random.randn(nm, target_rank) * 0.05
                V_pert += np.random.randn(mp, target_rank) * 0.05
                W_pert += np.random.randn(pn, target_rank) * 0.05
            else:
                # Extend with random columns
                if target_rank > U_init.shape[1]:
                    U_pert = np.zeros((nm, target_rank))
                    V_pert = np.zeros((mp, target_rank))
                    W_pert = np.zeros((pn, target_rank))
                    U_pert[:, :U_init.shape[1]] = U_init
                    V_pert[:, :V_init.shape[1]] = V_init
                    W_pert[:, :W_init.shape[1]] = W_init
                    U_pert[:, U_init.shape[1]:] = np.random.randn(nm, target_rank - U_init.shape[1]) * 0.1
                    V_pert[:, V_init.shape[1]:] = np.random.randn(mp, target_rank - V_init.shape[1]) * 0.1
                    W_pert[:, W_init.shape[1]:] = np.random.randn(pn, target_rank - W_init.shape[1]) * 0.1

            result = optimize_rank(U_pert, V_pert, W_pert, target_rank, max_iters=max_iters, lr=lr)
            if result is not None:
                return result

            # Track best error even if not converged
            current_err = error_metric(U_pert, V_pert, W_pert, target_rank)
            if current_err < best_err:
                best_err = current_err
                best_result = (U_pert.copy(), V_pert.copy(), W_pert.copy(), target_rank)

        return best_result

    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_block()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    # Try to find lower rank via gradient-based optimization with restarts
    # Theoretical lower bound: ceil(64/2) = 32, but realistically 40-48
    target_ranks = [48, 44, 40, 36, 32]

    for target in target_ranks:
        if time.time() > deadline - 25:
            break
        if target >= rank_best:
            continue

        # Try multiple restarts for each target rank
        result = optimize_with_restarts(U_best, V_best, W_best, target, n_restarts=3, max_iters=1500, lr=0.03)
        if result is not None:
            U_test, V_test, W_test, r_test = result
            if verify(U_test, V_test, W_test, r_test):
                U_best, V_best, W_best, rank_best = U_test, V_test, W_test, r_test
                continue
            else:
                # If verification fails, don't use this result
                pass

    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
