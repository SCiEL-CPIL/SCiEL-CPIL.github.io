import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe

    n, m, p = DIMENSIONS

    # Build the target tensor T
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def compute_loss(U, V, W):
        try:
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            return np.sum(diff * diff)
        except:
            return 1e9

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    def validate(U, V, W):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)

        if not (np.allclose(U, U_snap, atol=1e-6) and
                np.allclose(V, V_snap, atol=1e-6) and
                np.allclose(W, W_snap, atol=1e-6)):
            return False, None, None, None

        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=1e-6):
            return True, U_snap, V_snap, W_snap
        return False, None, None

    # Strategy: Use the block matrix multiplication identity
    # The 2x5x5 tensor can be decomposed as the tensor product of:
    # - 2x1x1 scalar multiplication tensor (rank 2)
    # - 1x5x5 vector-matrix multiplication tensor (rank 5)
    # This gives a constructive rank-10 decomposition via Kronecker products

    # First, try to find a rank-5 decomposition for 1x5x5 (row vector times matrix)
    # T_1[j, k, 0] = 1 for the 1x5x5 case, but actually the output index varies

    # For 1x5x5: T_1[a, b, c] where a=j (0..4), b=j*5+k (0..24), c=k (0..4)
    # Actually for 1xm x mxp: T_1[j, j*p+k, k] = 1
    # This is rank at most m*p = 25, but let's see if it's lower

    # Actually, let's think about this differently.
    # The 2x5x5 tensor T can be written as:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # = sum over (i,j,k) of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}

    # Let's try a direct search for low rank decompositions
    # but with better initialization

    best_solution = None
    best_rank = n * m * p  # 50

    # Try ranks from 10 up to 49
    for target_rank in range(10, min(50, best_rank)):
        if time.time() > deadline - 20:
            break

        # Try multiple restarts for this rank
        num_restarts = 8

        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break

            R = target_rank

            # Better initialization: use structured random
            # Initialize with small random values
            np.random.seed(42 + restart)
            U = np.random.randn(U_dim, R) * 0.3
            V = np.random.randn(V_dim, R) * 0.3
            W = np.random.randn(W_dim, R) * 0.3

            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def objective(x):
                u = x[:U_dim * R].reshape(U_dim, R)
                v = x[U_dim * R:U_dim * R + V_dim * R].reshape(V_dim, R)
                w = x[U_dim * R + V_dim * R:].reshape(W_dim, R)
                return compute_loss(u, v, w)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 800, 'disp': False, 'maxls': 80})

                if res.fun < 1e-8:
                    u_opt = res.x[:U_dim * R].reshape(U_dim, R)
                    v_opt = res.x[U_dim * R:U_dim * R + V_dim * R].reshape(V_dim, R)
                    w_opt = res.x[U_dim * R + V_dim * R:].reshape(W_dim, R)

                    valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        break

            except Exception as e:
                pass

        if best_solution:
            break

    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
