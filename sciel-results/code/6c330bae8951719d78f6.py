import numpy as np
import time
from scipy.optimize import minimize

def _simulate_aco(prizes, distance_matrix, max_len, eta, seed):
    """Fixed ACO protocol; returns best total prize over all iterations/ants."""
    np.random.seed(seed)
    N = len(prizes)
    tau = np.ones((N, N))
    best_prize_global = 0.0
    d_j0 = distance_matrix[:, 0]

    for _ in range(30):
        best_prize_iter = 0.0
        for _ in range(20):
            current = 0
            path_len = 0.0
            prize_total = 0.0
            visited = np.zeros(N, dtype=bool)
            visited[0] = True
            while True:
                remaining = max_len - path_len
                costs = distance_matrix[current, :] + d_j0
                eligible_mask = (~visited) & (costs <= remaining)
                if not np.any(eligible_mask):
                    break
                eligible_idx = np.where(eligible_mask)[0]
                weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                total_w = weights.sum()
                if total_w <= 0:
                    break
                probs = weights / total_w
                r = np.random.rand()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(eligible_idx):
                    idx = len(eligible_idx) - 1
                next_node = eligible_idx[idx]
                path_len += distance_matrix[current, next_node]
                prize_total += prizes[next_node]
                visited[next_node] = True
                current = next_node
            if prize_total > best_prize_iter:
                best_prize_iter = prize_total
        if best_prize_iter > best_prize_global:
            best_prize_global = best_prize_iter
        tau = tau * 0.9
        tau += best_prize_iter / 10.0

    return best_prize_global

def _clean_eta(eta, N):
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (N, N):
        eta = np.ones((N, N))
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def _log_eta_params_to_eta(params, N, base):
    """params: (N*N,) log-values -> positive eta (logit form, no exp overflow)."""
    return np.exp(params)  # params already on log scale

def _rbf_predict(X_train, y_train, X_new, gamma):
    """RBF (Gaussian) kernel regression with ridge. Returns prediction for X_new."""
    X_train = np.atleast_2d(X_train)
    X_new = np.atleast_2d(X_new)
    # pairwise squared distances
    d2 = (np.sum(X_train ** 2, axis=1)[:, None]
          + np.sum(X_new ** 2, axis=1)[None, :]
          - 2.0 * X_train @ X_new.T)
    d2 = np.maximum(d2, 0.0)
    K = np.exp(-gamma * d2)
    n = X_train.shape[0]
    lam = 1e-3
    A = K.T @ K + lam * np.eye(n)
    b = K.T @ y_train
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return K @ w

def _make_simple_prize_heuristic(prizes, distance_matrix, max_len):
    """Cheap fallback: prize[j] / (d[i,j]+d[j,0]+eps)."""
    N = len(prizes)
    p = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]
    eta = p[None, :] / (C + 1e-2)
    return _clean_eta(eta, N)

def heuristics(prizes, distance_matrix, max_len):
    start = time.time()
    DEADLINE = 300.0
    def left():
        return DEADLINE - (time.time() - start)

    N = len(prizes)
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]  # cost i->j->0
    slack = np.maximum(1.0 - C / max_len, 0.0)
    p = np.maximum(prizes, 1e-9)

    # ---------- Build a low-dim feature basis for the heuristic ----------
    # Each candidate eta = exp( sum_k w_k * f_k ), f_k are fixed basis matrices.
    # We learn w via surrogate, then do a short L-BFGS on the real objective.
    basis = []
    # prize magnitude
    basis.append(np.log(p[None, :]))
    # negative cost
    basis.append(-C / max_len)
    # slack
    basis.append(np.log(slack + 1e-3))
    # prize * slack
    basis.append(np.log(p[None, :]) + np.log(slack + 1e-3))
    # prize / cost
    basis.append(np.log(p[None, :]) - C / max_len)
    # slack^2
    basis.append(2.0 * np.log(slack + 1e-3))
    # constant
    basis.append(np.ones((N, N)))
    # prize rank (learned-free structural signal)
    ranks = np.argsort(np.argsort(p))
    basis.append(np.log(ranks[None, :] + 1.0))
    # negative distance to 0 for j (return cost)
    basis.append(-d_j0[None, :])
    # prize * (1 - d[j,0]/max_len)
    basis.append(np.log(p[None, :]) + np.log(np.maximum(1.0 - d_j0[None, :] / max_len, 1e-3)))

    B = np.stack(basis, axis=0)  # (K, N, N)
    K = B.shape[0]
    Bv = B.reshape(K, -1)  # (K, N*N)

    def w_to_eta(w):
        eta = np.exp(Bv @ w).reshape(N, N)
        return _clean_eta(eta, N)

    # ---------- Seed the surrogate with diverse initial evaluations ----------
    rng = np.random.RandomState(0)
    X_train = []
    y_train = []

    def eval_and_store(w):
        eta = w_to_eta(w)
        s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
        X_train.append(w.copy())
        y_train.append(float(s))
        return s

    # initial: zeros, random, and a few structured starts
    eval_and_store(np.zeros(K))
    for _ in range(12):
        if left() < 90:
            break
        w = rng.normal(0.0, 0.5, size=K)
        eval_and_store(w)

    # structured starts: emphasize prize, or emphasize slack, or prize/cost
    for scale, idx in [(1.5, 0), (1.5, 2), (1.5, 3), (-1.0, 1), (1.0, 6)]:
        if left() < 90:
            break
        w = np.zeros(K)
        w[idx] = scale
        eval_and_store(w)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_mean = y_train.mean()
    y_std = max(y_train.std(), 1e-6)
    y_norm = (y_train - y_mean) / y_std

    best_w = X_train[np.argmax(y_train)].copy()
    best_score = float(y_train.max())

    # ---------- Surrogate-guided optimization (RBF) ----------
    def surrogate_neg(w):
        pred = _rbf_predict(X_train, y_norm, w.reshape(1, -1), gamma=2.0)
        return -pred[0]

    # local search around current best with L-BFGS on the surrogate
    for _ in range(6):
        if left() < 70:
            break
        res = minimize(surrogate_neg, best_w, method='L-BFGS-B',
                       options={'maxiter': 40, 'gtol': 1e-4})
        w_cand = res.x
        if left() < 60:
            break
        s = _simulate_aco(prizes, distance_matrix, max_len, w_to_eta(w_cand), 0)
        X_train = np.vstack([X_train, w_cand.reshape(1, -1)])
        y_train = np.append(y_train, s)
        y_norm = (y_train - y_mean) / y_std  # recompute cheaply
        if s > best_score:
            best_score = float(s)
            best_w = w_cand.copy()
        # also try a small random perturbation of the surrogate argmin
        w_pert = w_cand + rng.normal(0.0, 0.3, size=K)
        if left() < 60:
            break
        s2 = _simulate_aco(prizes, distance_matrix, max_len, w_to_eta(w_pert), 0)
        X_train = np.vstack([X_train, w_pert.reshape(1, -1)])
        y_train = np.append(y_train, s2)
        y_norm = (y_train - y_mean) / y_std
        if s2 > best_score:
            best_score = float(s2)
            best_w = w_pert.copy()

    # ---------- Direct L-BFGS on the REAL objective around best_w ----------
    if left() > 40:
        def real_neg(w):
            eta = w_to_eta(w)
            return -_simulate_aco(prizes, distance_matrix, max_len, eta, 0)
        try:
            res = minimize(real_neg, best_w, method='L-BFGS-B',
                           options={'maxiter': 15, 'gtol': 1e-3})
            w_cand = res.x
            s = -real_neg(w_cand)
            if s > best_score:
                best_score = float(s)
                best_w = w_cand.copy()
        except Exception:
            pass

    # ---------- Validate with extra seeds, keep best ----------
    if left() > 25:
        eta_best = w_to_eta(best_w)
        # quick multi-seed confirmation; if a neighbor generalizes better, take it
        for _ in range(3):
            if left() < 20:
                break
            w_try = best_w + rng.normal(0.0, 0.2, size=K)
            eta_try = w_to_eta(w_try)
            a = _simulate_aco(prizes, distance_matrix, max_len, eta_best, 1)
            b = _simulate_aco(prizes, distance_matrix, max_len, eta_try, 1)
            if b > a:
                best_w = w_try
                eta_best = eta_try

    eta = w_to_eta(best_w)

    # ---------- Final safety ----------
    eta = _clean_eta(eta, N)
    if not np.all(np.isfinite(eta)) or eta.shape != (N, N):
        eta = _make_simple_prize_heuristic(prizes, distance_matrix, max_len)
    return eta
