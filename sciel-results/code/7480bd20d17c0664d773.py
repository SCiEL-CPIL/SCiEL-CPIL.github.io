import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    time_limit = 620.0  # Safety margin before 630s kill

    # Precompute bin indices
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2*Y

    idx_m1 = (V_m1 + 9).flatten()
    idx_0 = V_0.flatten()
    idx_1 = (V_1).flatten()
    idx_2 = (V_2).flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def get_probabilities(params):
        """Softmax over 100 log-weights."""
        # Subtract max for numerical stability
        shifted = params - np.max(params)
        P = np.exp(shifted)
        total = P.sum()
        if total < 1e-15:
            return None
        return P / total

    def compute_entropies(P_flat):
        counts_m1 = np.bincount(idx_m1, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(idx_2, weights=P_flat, minlength=num_bins_2)

        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))

        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)
        return H_m1, H_0, H_1, H_2

    def compute_ratio(params):
        P_flat = get_probabilities(params)
        if P_flat is None:
            return 0.0
        H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def objective(params):
        return -compute_ratio(params)

    def objective_smooth(params, beta):
        P_flat = get_probabilities(params)
        if P_flat is None:
            return 1e10
        H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
        Hs = np.array([H_0, H_1, H_2])
        max_H = np.max(Hs)
        # Stable log-sum-exp for soft-max
        M = (max_H + np.log(np.sum(np.exp(beta * (Hs - max_H))))) / beta
        if M < 1e-9:
            return 1e10
        return -H_m1 / M

    # Generate diverse starting points
    rng = np.random.default_rng(12345)
    start_points = []

    # 1. Uniform (all zeros)
    start_points.append(np.zeros(100))

    # 2. Random Gaussian starts
    for scale in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        for _ in range(5):
            start_points.append(rng.normal(0, scale, 100))

    # 3. Anti-diagonal concentration (x+y = constant)
    for target in [3, 5, 7, 9, 11]:
        p = np.full(100, -5.0)
        for x in range(10):
            y = target - x
            if 0 <= y < 10:
                p[y*10+x] = 0.0
        start_points.append(p)

    # 4. Diagonal concentration (x-y = constant)
    for d in [-4, -2, 0, 2, 4]:
        p = np.full(100, -5.0)
        for x in range(10):
            y = x - d
            if 0 <= y < 10:
                p[y*10+x] = 0.0
        start_points.append(p)

    # 5. Row concentration
    for row in [2, 4, 5, 7]:
        p = np.full(100, -5.0)
        p[row*10:(row+1)*10] = 0.0
        start_points.append(p)

    # 6. Column concentration
    for col in [2, 4, 5, 7]:
        p = np.full(100, -5.0)
        p[::10, col] = 0.0 if False else 0.0
        p[np.arange(0, 100, 10) + col] = 0.0
        start_points.append(p)

    # 7. Corner concentration
    corners = [(0,0), (0,9), (9,0), (9,9), (4,4), (4,5), (5,4), (5,5)]
    for (cy, cx) in corners:
        p = np.full(100, -5.0)
        p[cy*10+cx] = 0.0
        start_points.append(p)

    # 8. Two-anti-diagonal (spread x-y while keeping x+y moderate)
    for d1, d2 in [(-3, 3), (-4, 4), (-2, 2), (-5, 5)]:
        p = np.full(100, -6.0)
        for x in range(10):
            for d in [d1, d2]:
                y = x - d
                if 0 <= y < 10:
                    p[y*10+x] = 0.0
        start_points.append(p)

    # 9. Structured: high weight on cells where x-y is extreme
    p = np.zeros(100)
    for x in range(10):
        for y in range(10):
            d = x - y
            p[y*10+x] = -abs(d) * 0.3
    start_points.append(p)

    # 10. Structured: maximize x-y spread (favor corners)
    p = np.zeros(100)
    for x in range(10):
        for y in range(10):
            d = x - y
            p[y*10+x] = abs(d) * 0.2
    start_points.append(p)

    # 11. Log-linear on (x, y)
    for ax in [-1, 0, 1]:
        for ay in [-1, 0, 1]:
            p = np.zeros(100)
            for x in range(10):
                for y in range(10):
                    p[y*10+x] = ax * x + ay * y
            start_points.append(p)

    # 12. Quadratic structures
    for cx in [-1, 0, 1]:
        for cy in [-1, 0, 1]:
            p = np.zeros(100)
            for x in range(10):
                for y in range(10):
                    p[y*10+x] = cx * (x-4.5)**2 + cy * (y-4.5)**2
            start_points.append(p)

    # 13. Random sparse (few active cells)
    for _ in range(10):
        p = np.full(100, -8.0)
        n_active = rng.integers(3, 8)
        active = rng.choice(100, size=n_active, replace=False)
        p[active] = rng.normal(0, 2, n_active)
        start_points.append(p)

    # 14. Random with anti-diagonal bias
    for _ in range(10):
        p = rng.normal(0, 1, 100)
        for x in range(10):
            for y in range(10):
                p[y*10+x] -= 0.3 * (x + y - 9)**2
        start_points.append(p)

    # 15. Random with diagonal bias
    for _ in range(10):
        p = rng.normal(0, 1, 100)
        for x in range(10):
            for y in range(10):
                p[y*10+x] += 0.3 * (x - y)
        start_points.append(p)

    # Evaluate all starting points
    best_score = 0.0
    best_params = None

    for sp in start_points:
        score = compute_ratio(sp)
        if score > best_score:
            best_score = score
            best_params = sp.copy()

    # Optimization: multi-start with annealing
    betas_schedule = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    # Sort starts by initial score (best first)
    scored_starts = [(compute_ratio(sp), sp) for sp in start_points]
    scored_starts.sort(key=lambda x: -x[0])

    max_starts_to_optimize = 40
    for initial_score, sp in scored_starts[:max_starts_to_optimize]:
        if time.time() - start_time > time_limit - 80:
            break

        current_params = sp.copy()
        current_score = initial_score

        # Annealing with smooth objective
        for beta in betas_schedule:
            if time.time() - start_time > time_limit - 80:
                break

            if beta >= 200.0:
                # Use exact objective
                def obj(p):
                    return objective(p)
            else:
                def obj(p):
                    return objective_smooth(p, beta)

            try:
                res = minimize(obj, current_params, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < obj(current_params):
                    current_params = res.x
            except Exception:
                pass

        # Evaluate exact score
        score = compute_ratio(current_params)
        if score > best_score:
            best_score = score
            best_params = current_params.copy()

    # Additional optimization rounds on best found
    if best_params is not None and time.time() - start_time < time_limit - 60:
        # Nelder-Mead for non-smooth max
        try:
            res_nm = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 10000, 'fatol': 1e-14, 'xatol': 1e-13})
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    # COBYLA (derivative-free, handles non-smooth)
    if best_params is not None and time.time() - start_time < time_limit - 40:
        try:
            res_cobyla = minimize(objective, best_params, method='COBYLA',
                                  options={'maxiter': 5000, 'rhobeg': 1.0})
            if -res_cobyla.fun > best_score:
                best_score = -res_cobyla.fun
                best_params = res_cobyla.x
        except Exception:
            pass

    # Final L-BFGS-B polish
    if best_params is not None and time.time() - start_time < time_limit - 20:
        try:
            res_final = minimize(objective, best_params, method='L-BFGS-B',
                                 options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if -res_final.fun > best_score:
                best_score = -res_final.fun
                best_params = res_final.x
        except Exception:
            pass

    # Another Nelder-Mead from slightly perturbed best
    if best_params is not None and time.time() - start_time < time_limit - 10:
        for _ in range(3):
            perturbed = best_params + rng.normal(0, 0.5, 100)
            try:
                res_pert = minimize(objective, perturbed, method='Nelder-Mead',
                                    options={'maxiter': 5000, 'fatol': 1e-14, 'xatol': 1e-13})
                if -res_pert.fun > best_score:
                    best_score = -res_pert.fun
                    best_params = res_pert.x
            except Exception:
                pass

    # Reconstruct P
    if best_params is None:
        P = np.ones((10, 10)) / 100.0
    else:
        P_flat = get_probabilities(best_params)
        if P_flat is None:
            P = np.ones((10, 10)) / 100.0
        else:
            P = P_flat.reshape(10, 10)

    # Normalize and validate
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-9:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s

    return P

if __name__ == "__main__":
    pass
