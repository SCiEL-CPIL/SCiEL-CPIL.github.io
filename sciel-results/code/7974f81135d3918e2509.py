# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_entropies(P: np.ndarray) -> tuple:
    """
    Compute D = H_{-1} and M = max(H_0, H_1, H_2) for a 10x10 grid P[y, x].
    """
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y,x] = x, Y[y,x] = y

    # Slope -1: x - y
    vals_m1 = X - Y
    vals_m1 = vals_m1.flatten()
    p_flat = P.flatten()
    # Pool by value
    unique_m1, inverse_m1 = np.unique(vals_m1, return_inverse=True)
    dist_m1 = np.zeros(len(unique_m1))
    np.add.at(dist_m1, inverse_m1, p_flat)
    H_m1 = _shannon_entropy(dist_m1)

    # Slope 0: x
    vals_0 = X
    unique_0, inverse_0 = np.unique(vals_0, return_inverse=True)
    dist_0 = np.zeros(len(unique_0))
    np.add.at(dist_0, inverse_0, p_flat)
    H_0 = _shannon_entropy(dist_0)

    # Slope 1: x + y
    vals_1 = X + Y
    unique_1, inverse_1 = np.unique(vals_1, return_inverse=True)
    dist_1 = np.zeros(len(unique_1))
    np.add.at(dist_1, inverse_1, p_flat)
    H_1 = _shannon_entropy(dist_1)

    # Slope 2: x + 2y
    vals_2 = X + 2 * Y
    unique_2, inverse_2 = np.unique(vals_2, return_inverse=True)
    dist_2 = np.zeros(len(unique_2))
    np.add.at(dist_2, inverse_2, p_flat)
    H_2 = _shannon_entropy(dist_2)

    M = max(H_0, H_1, H_2)
    return H_m1, M


def _smooth_max_surrogate(P: np.ndarray, temperature: float = 0.01) -> float:
    """
    Smooth surrogate for max(H_0, H_1, H_2) using log-sum-exp.
    Returns a smooth approximation to the max.
    """
    H_m1, M = _compute_entropies(P)
    # We need individual entropies for the smooth max
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    p_flat = P.flatten()

    vals_0 = X.flatten()
    unique_0, inverse_0 = np.unique(vals_0, return_inverse=True)
    dist_0 = np.zeros(len(unique_0))
    np.add.at(dist_0, inverse_0, p_flat)
    H_0 = _shannon_entropy(dist_0)

    vals_1 = (X + Y).flatten()
    unique_1, inverse_1 = np.unique(vals_1, return_inverse=True)
    dist_1 = np.zeros(len(unique_1))
    np.add.at(dist_1, inverse_1, p_flat)
    H_1 = _shannon_entropy(dist_1)

    vals_2 = (X + 2 * Y).flatten()
    unique_2, inverse_2 = np.unique(vals_2, return_inverse=True)
    dist_2 = np.zeros(len(unique_2))
    np.add.at(dist_2, inverse_2, p_flat)
    H_2 = _shannon_entropy(dist_2)

    Hs = np.array([H_0, H_1, H_2], dtype=np.float64)
    max_H = np.max(Hs)
    if max_H < 1e-12:
        return 0.0
    # Log-sum-exp for smooth max
    shifted = Hs - max_H
    log_sum_exp = max_H + np.log(np.sum(np.exp(shifted)))
    return log_sum_exp


def _objective(params: np.ndarray, temperature: float = 0.01) -> float:
    """
    Objective to minimize: -D / M_smooth (we want to maximize D/M).
    """
    P = np.exp(params)
    P = P / np.sum(P)
    P = np.clip(P, 1e-15, None)
    P = P / np.sum(P)

    H_m1, M = _compute_entropies(P)
    if M < 1e-9:
        return 1e6

    # Use smooth max surrogate for better gradients
    M_smooth = _smooth_max_surrogate(P, temperature)
    if M_smooth < 1e-9:
        return 1e6

    ratio = H_m1 / M_smooth
    return -ratio


def _objective_grad(params: np.ndarray, temperature: float = 0.01) -> tuple:
    """
    Objective and numerical gradient (since analytical is complex).
    """
    val = _objective(params, temperature)
    # Use finite differences for gradient
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _objective(params_plus, temperature)
        grad[i] = (val_plus - val) / eps
    return val, grad


def _normalize_and_validate(P: np.ndarray) -> np.ndarray:
    """Clip, normalize, and validate the distribution."""
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s <= 1e-9:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 580  # Safety margin

    # Start from uniform
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_ratio = _compute_entropies(best_P)[0] / _compute_entropies(best_P)[1]

    # Parameterize as log-probabilities for unconstrained optimization
    log_P0 = np.log(P_uniform)
    params0 = log_P0.flatten()

    # Try multiple starting points and optimization runs
    rng = np.random.default_rng(42)

    # Define several initial candidates
    candidates = [params0.copy()]

    # Add some structured initial guesses
    # Diagonal-heavy
    P_diag = np.zeros((n, n))
    for i in range(n):
        P_diag[i, i] = 1.0 / n
    P_diag = P_diag / P_diag.sum()
    candidates.append(np.log(P_diag).flatten())

    # Anti-diagonal heavy (x - y = 0)
    P_anti = np.zeros((n, n))
    for i in range(n):
        P_anti[i, i] = 1.0 / n
    P_anti = P_anti / P_anti.sum()
    candidates.append(np.log(P_anti).flatten())

    # Random perturbations of uniform
    for _ in range(5):
        P_rand = np.random.dirichlet(np.ones(n * n), size=1)[0]
        P_rand = P_rand.reshape(n, n)
        candidates.append(np.log(P_rand).flatten())

    # Also try sparse-like starts
    for _ in range(5):
        P_sparse = np.zeros((n, n))
        # Pick 10-20 random cells
        num_cells = rng.integers(10, 21)
        cells = rng.choice(n * n, size=num_cells, replace=False)
        P_sparse_flat = np.zeros(n * n)
        P_sparse_flat[cells] = 1.0 / num_cells
        P_sparse = P_sparse_flat.reshape(n, n)
        P_sparse = P_sparse / P_sparse.sum()
        candidates.append(np.log(np.clip(P_sparse, 1e-10, None)).flatten())

    temperature = 0.05

    for idx, params_start in enumerate(candidates):
        if time.time() > deadline - 60:
            break

        # L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                params_start,
                args=(temperature,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < -best_ratio:
                P_opt = np.exp(result.x)
                P_opt = P_opt / np.sum(P_opt)
                P_opt = np.clip(P_opt, 1e-15, None)
                P_opt = P_opt / np.sum(P_opt)
                H_m1, M = _compute_entropies(P_opt)
                if M > 1e-9:
                    ratio = H_m1 / M
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_P = P_opt
        except Exception:
            pass

    # Polish pass with tighter temperature
    if time.time() < deadline - 30:
        temperature_tight = 0.01
        try:
            result = minimize(
                _objective,
                np.log(np.clip(best_P, 1e-15, None)).flatten(),
                args=(temperature_tight,),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            P_polished = np.exp(result.x)
            P_polished = P_polished / np.sum(P_polished)
            P_polished = np.clip(P_polished, 1e-15, None)
            P_polished = P_polished / np.sum(P_polished)
            H_m1, M = _compute_entropies(P_polished)
            if M > 1e-9:
                ratio = H_m1 / M
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_polished
        except Exception:
            pass

    return _normalize_and_validate(best_P)


# EVOLVE-BLOCK-END
