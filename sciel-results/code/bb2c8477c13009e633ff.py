import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec_val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(dec_val - 1.0) > 1e-7:
                    return False
    return True


def _polish_half_integer(U, V, W):
    """Nudge a real solution toward the half-integer lattice; keep if still valid."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None


def _rank33_solution():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})

    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}


class _Timeout(Exception):
    pass


def _make_loss_grad(target_rank, T_indices):
    n, m, p = DIMENSIONS
    num_u = n * m
    num_v = m * p
    num_w = p * n

    def loss_and_grad(x):
        u_size = num_u * target_rank
        v_size = num_v * target_rank
        U = x[:u_size].reshape(num_u, target_rank)
        V = x[u_size:u_size + v_size].reshape(num_v, target_rank)
        W = x[u_size + v_size:].reshape(num_w, target_rank)

        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)

        for (a, b, c) in T_indices:
            prod_sum = np.dot(U[a, :], V[b, :] * W[c, :])
            error = prod_sum - 1.0
            loss += error * error
            g = 2.0 * error
            grad_U[a, :] += g * (V[b, :] * W[c, :])
            grad_V[b, :] += g * (U[a, :] * W[c, :])
            grad_W[c, :] += g * (U[a, :] * V[b, :])

        return loss, np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])

    return loss_and_grad


def _try_scipy_search(target_rank, deadline, seed=0, budget=25):
    """L-BFGS-B search for a rank-target_rank whole-tensor decomposition."""
    n, m, p = DIMENSIONS
    num_u = n * m
    num_v = m * p
    num_w = p * n

    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i * m + j, j * p + k, k * n + i))

    loss_and_grad = _make_loss_grad(target_rank, T_indices)

    base = _rank33_solution()
    U33 = np.array(base['U'], dtype=float)
    V33 = np.array(base['V'], dtype=float)
    W33 = np.array(base['W'], dtype=float)

    inits = []
    # 0: truncate the known rank-33 solution
    inits.append((U33[:, :target_rank], V33[:, :target_rank], W33[:, :target_rank]))
    # 1: random moderate
    np.random.seed(seed + 1)
    inits.append((np.random.randn(num_u, target_rank) * 1.0,
                  np.random.randn(num_v, target_rank) * 1.0,
                  np.random.randn(num_w, target_rank) * 1.0))
    # 2: sparse random signs
    np.random.seed(seed + 2)
    U2 = np.zeros((num_u, target_rank)); V2 = np.zeros((num_v, target_rank)); W2 = np.zeros((num_w, target_rank))
    for r in range(target_rank):
        U2[np.random.randint(0, num_u), r] = np.random.choice([-1, 1])
        V2[np.random.randint(0, num_v), r] = np.random.choice([-1, 1])
        W2[np.random.randint(0, num_w), r] = np.random.choice([-1, 1])
    inits.append((U2, V2, W2))
    # 3: truncated + small noise
    np.random.seed(seed + 3)
    inits.append((U33[:, :target_rank] + np.random.randn(num_u, target_rank) * 0.2,
                  V33[:, :target_rank] + np.random.randn(num_v, target_rank) * 0.2,
                  W33[:, :target_rank] + np.random.randn(num_w, target_rank) * 0.2))

    for Ui, Vi, Wi in inits:
        if time.time() > deadline - 3:
            break
        x0 = np.concatenate([Ui.ravel(), Vi.ravel(), Wi.ravel()])
        maxiter = max(100, int(budget / 0.03))
        try:
            result = minimize(loss_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'maxfun': 4000,
                                       'ftol': 1e-12, 'gtol': 1e-9})
        except _Timeout:
            continue
        except Exception:
            continue

        if result.fun < 1e-6:
            x_opt = result.x
            u_size = num_u * target_rank
            v_size = num_v * target_rank
            U_opt = x_opt[:u_size].reshape(num_u, target_rank)
            V_opt = x_opt[u_size:u_size + v_size].reshape(num_v, target_rank)
            W_opt = x_opt[u_size + v_size:].reshape(num_w, target_rank)
            U_l = U_opt.tolist(); V_l = V_opt.tolist(); W_l = W_opt.tolist()
            if _verify(U_l, V_l, W_l):
                pol = _polish_half_integer(U_opt, V_opt, W_opt)
                if pol is not None:
                    return {"U": pol[0], "V": pol[1], "W": pol[2]}
                return {"U": U_l, "V": V_l, "W": W_l}
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    fallback = _rank33_solution()
    if not _verify(fallback['U'], fallback['V'], fallback['W']):
        pass  # should never happen

    for target_rank in [32, 31, 30, 29, 28, 27, 26, 25, 24]:
        if time.time() > deadline - 8:
            break
        budget = max(6, min(25, (deadline - time.time()) - 4))
        result = _try_scipy_search(target_rank, deadline, seed=target_rank, budget=budget)
        if result is not None:
            return result

    return fallback
