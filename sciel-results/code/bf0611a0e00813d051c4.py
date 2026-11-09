# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on 10x10 grid.
    Maximizes H(x-y) / max(H(x), H(x+y), H(x+2y)).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s kill

    # Precompute projection indices and masks
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y,x] = x, Y[y,x] = y

    proj_minus1 = X - Y      # x - y, range -9..9
    proj_0 = X               # x, range 0..9
    proj_1 = X + Y           # x + y, range 0..18
    proj_2 = X + 2 * Y       # x + 2y, range 0..27

    # Precompute unique values and masks for each projection
    def precompute_projection(proj):
        vals = np.unique(proj)
        masks = {}
        for v in vals:
            masks[v] = (proj == v)
        return vals, masks

    vals_m1, masks_m1 = precompute_projection(proj_minus1)
    vals_0, masks_0 = precompute_projection(proj_0)
    vals_1, masks_1 = precompute_projection(proj_1)
    vals_2, masks_2 = precompute_projection(proj_2)

    def compute_entropy_from_masks(masks, P):
        """Compute Shannon entropy using precomputed masks."""
        probs = np.array([np.sum(P[mask]) for mask in masks.values()])
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def compute_all_entropies(P):
        """Compute all four entropies efficiently."""
        D = compute_entropy_from_masks(masks_m1, P)
        H0 = compute_entropy_from_masks(masks_0, P)
        H1 = compute_entropy_from_masks(masks_1, P)
        H2 = compute_entropy_from_masks(masks_2, P)
        return D, H0, H1, H2

    def compute_ratio(params):
        """Compute the ratio D/M from log-weights."""
        log_w = params.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        D, H0, H1, H2 = compute_all_entropies(P)
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0
        return D / M

    def objective(params):
        """Negative ratio to minimize."""
        log_w = params.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        D, H0, H1, H2 = compute_all_entropies(P)
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 1e9
        return -(D / M)

    def objective_smooth(params, temperature=0.1):
        """Smoothed objective using soft-max for the max term."""
        log_w = params.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        D, H0, H1, H2 = compute_all_entropies(P)
        # Soft max using log-sum-exp
        Hs = np.array([H0, H1, H2])
        max_H = np.max(Hs)
        if max_H < 1e-9:
            return 1e9
        log_sum_exp = max_H + np.log(np.sum(np.exp((Hs - max_H) / temperature))) * temperature
        M_smooth = log_sum_exp
        if M_smooth < 1e-9:
            return 1e9
        return -(D / M_smooth)

    # Initialize
    best_params = np.zeros(n * n)
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)

    def update_best(params, score):
        nonlocal best_params, best_score, best_P
        if score > best_score:
            best_score = score
            best_params = params.copy()
            log_w = best_params.reshape((n, n))
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            best_P = w / np.sum(w)

    # Generate diverse initializations
    np.random.seed(42)
    initializations = []

    # Uniform
    initializations.append(np.zeros(n * n))

    # Diagonal bias to flatten x-y
    init_diag = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            init_diag[i, j] = 1.0 / (1.0 + d)
    init_diag = init_diag - np.mean(init_diag)
    initializations.append(init_diag.flatten())

    # Anti-diagonal bias
    init_anti = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i + j - (n - 1))
            init_anti[i, j] = 1.0 / (1.0 + d)
    init_anti = init_anti - np.mean(init_anti)
    initializations.append(init_anti.flatten())

    # Concentrate on corners
    init_corner = np.zeros((n, n))
    init_corner[0, 0] = 1.0
    init_corner[0, -1] = 1.0
    init_corner[-1, 0] = 1.0
    init_corner[-1, -1] = 1.0
    initializations.append(init_corner)

    # Concentrate on x=0 column
    init_col0 = np.zeros((n, n))
    init_col0[:, 0] = 1.0
    initializations.append(init_col0)

    # Concentrate on y=0 row
    init_row0 = np.zeros((n, n))
    init_row0[0, :] = 1.0
    initializations.append(init_row0)

    # Random initializations with varying scales
    for scale in [0.3, 0.5, 1.0, 2.0, 3.0]:
        initializations.append(np.random.randn(n * n) * scale)

    # Structured: weight by distance from center
    init_center = np.zeros((n, n))
    cx, cy = (n - 1) / 2.0, (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dist = np.sqrt((i - cx) ** 2 + (j - cx) ** 2)
            init_center[i, j] = np.exp(-dist / 3.0)
    init_center = init_center - np.mean(init_center)
    initializations.append(init_center.flatten())

    # Run optimization from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 60:
            break

        # Nelder-Mead
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-9})
            score = -result.fun
            if score > 0:
                update_best(result.x, score)
        except Exception:
            pass

        # L-BFGS-B
        if time.time() > deadline - 60:
            break
        try:
            result2 = minimize(objective, init, method='L-BFGS-B',
                              options={'maxiter': 5000, 'ftol': 1e-12})
            score = -result2.fun
            if score > 0:
                update_best(result2.x, score)
        except Exception:
            pass

        # Smooth objective with L-BFGS-B (helps with the kink at max)
        if time.time() > deadline - 45:
            break
        try:
            result3 = minimize(lambda p: objective_smooth(p, 0.05), init, method='L-BFGS-B',
                              options={'maxiter': 3000, 'ftol': 1e-12})
            # Evaluate with exact objective
            exact_score = -objective(result3.x)
            if exact_score > 0:
                update_best(result3.x, exact_score)
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 30 and best_score > 0:
        # Multiple polish attempts with different methods
        polish_inits = [best_params]
        # Add small perturbations
        for _ in range(5):
            perturb = best_params + np.random.randn(n * n) * 0.01
            polish_inits.append(perturb)

        for p_init in polish_inits:
            if time.time() > deadline - 15:
                break

            # Very tight Nelder-Mead
            try:
                result = minimize(objective, p_init, method='Nelder-Mead',
                                 options={'maxiter': 30000, 'xatol': 1e-11, 'fatol': 1e-11})
                score = -result.fun
                if score > 0:
                    update_best(result.x, score)
            except Exception:
                pass

            # Tight L-BFGS-B
            if time.time() > deadline - 15:
                break
            try:
                result2 = minimize(objective, p_init, method='L-BFGS-B',
                                  options={'maxiter': 10000, 'ftol': 1e-14})
                score = -result2.fun
                if score > 0:
                    update_best(result2.x, score)
            except Exception:
                pass

            # Smooth polish
            if time.time() > deadline - 15:
                break
            try:
                result3 = minimize(lambda p: objective_smooth(p, 0.01), p_init, method='L-BFGS-B',
                                  options={'maxiter': 5000, 'ftol': 1e-14})
                exact_score = -objective(result3.x)
                if exact_score > 0:
                    update_best(result3.x, exact_score)
            except Exception:
                pass

    # Ensure valid output
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total

    return best_P


# EVOLVE-BLOCK-END
