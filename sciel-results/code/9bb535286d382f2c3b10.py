# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _precompute_pooling_matrices(n=10):
    """Precompute pooling matrices for slopes -1, 0, 1, 2.
    Each matrix A_r has shape (k_r, n*n) where k_r is number of unique values.
    A_r[v, idx] = 1 if cell idx maps to value v under slope r.
    """
    y_idx, x_idx = np.indices((n, n))
    flat_idx = np.arange(n * n)
    
    matrices = {}
    for r in [-1, 0, 1, 2]:
        vals = (x_idx + r * y_idx).flatten()
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        k = len(unique_vals)
        A = np.zeros((k, n * n), dtype=np.float64)
        A[inverse, flat_idx] = 1.0
        matrices[r] = (A, unique_vals, inverse)
    
    return matrices


def _compute_all_entropies(P_flat, matrices):
    """Compute H_{-1}, H_0, H_1, H_2 from flattened P."""
    entropies = {}
    for r in [-1, 0, 1, 2]:
        A, _, _ = matrices[r]
        probs = A @ P_flat
        probs = np.maximum(probs, 0)
        mask = probs > 1e-15
        if mask.sum() == 0:
            entropies[r] = 0.0
        else:
            p = probs[mask]
            entropies[r] = -np.sum(p * np.log2(p))
    return entropies


def _compute_ratio(P_flat, matrices):
    """Compute D/M ratio."""
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    if M < 1e-9:
        return 0.0, D, M
    return D / M, D, M


def _soft_max_entropies(P_flat, matrices, beta=10.0):
    """Compute soft-max of H_0, H_1, H_2 using log-sum-exp with temperature 1/beta."""
    ent = _compute_all_entropies(P_flat, matrices)
    H_vals = np.array([ent[0], ent[1], ent[2]])
    max_H = np.max(H_vals)
    exp_vals = np.exp(beta * (H_vals - max_H))
    soft_max = max_H + (1.0 / beta) * np.log(np.sum(exp_vals))
    return soft_max, H_vals


def _gradient_soft_log_ratio(P_flat, matrices, beta=10.0):
    """Compute gradient of log(D) - log(soft_max(H_0, H_1, H_2)) w.r.t. P_flat."""
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    
    if D < 1e-15:
        return np.zeros_like(P_flat)
    
    H_vals = np.array([ent[0], ent[1], ent[2]])
    max_H = np.max(H_vals)
    exp_vals = np.exp(beta * (H_vals - max_H))
    sum_exp = np.sum(exp_vals)
    soft_max_val = max_H + (1.0 / beta) * np.log(sum_exp)
    
    if soft_max_val < 1e-15:
        return np.zeros_like(P_flat)
    
    w = exp_vals / sum_exp
    
    A_num, _, inv_num = matrices[-1]
    q_num = A_num @ P_flat
    q_num = np.maximum(q_num, 1e-15)
    log_q_num = np.log2(q_num)
    dD_dP = -(log_q_num[inv_num] + 1.0 / np.log(2))
    dlogD_dP = dD_dP / D
    
    dlogM_dP = np.zeros_like(P_flat)
    for i, r in enumerate([0, 1, 2]):
        if w[i] < 1e-15:
            continue
        A_den, _, inv_den = matrices[r]
        q_den = A_den @ P_flat
        q_den = np.maximum(q_den, 1e-15)
        log_q_den = np.log2(q_den)
        dH_den_dP = -(log_q_den[inv_den] + 1.0 / np.log(2))
        dlogM_dP += w[i] * dH_den_dP / soft_max_val
    
    return dlogD_dP - dlogM_dP


def _project_simplex(P, n=10):
    """Project P onto simplex: non-negative, sum to 1."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s > 1e-15:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
    return P


def _gradient_ascent(P, matrices, n, max_iters=2000, time_limit=300, beta=10.0, 
                      step_size=None, polish=False):
    """Run gradient ascent on log-ratio objective with adaptive step size."""
    start_time = time.time()
    P_flat = P.flatten().copy()
    
    best_P_flat = P_flat.copy()
    best_ratio, _, _ = _compute_ratio(P_flat, matrices)
    
    if step_size is None:
        step_size = 0.005 if polish else 0.02
    
    patience = 0
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        grad = _gradient_soft_log_ratio(P_flat, matrices, beta=beta)
        
        P_new_flat = P_flat + step_size * grad
        P_new = _project_simplex(P_new_flat.reshape(n, n), n)
        P_new_flat = P_new.flatten()
        
        new_ratio, _, _ = _compute_ratio(P_new_flat, matrices)
        
        if new_ratio > best_ratio + 1e-12:
            best_ratio = new_ratio
            best_P_flat = P_new_flat.copy()
            patience = 0
            no_improve_count = 0
        else:
            patience += 1
            no_improve_count += 1
        
        if patience > 30:
            step_size *= 0.8
            patience = 0
        
        if no_improve_count > (200 if polish else 300):
            break
        
        P_flat = P_new_flat
    
    return best_P_flat.reshape(n, n), best_ratio


def _random_init(n, rng):
    """Generate a random initial distribution on the simplex."""
    P = rng.dirichlet(np.ones(n * n))
    return P.reshape(n, n)


def _structured_init(n, rng):
    """Generate structured initial distributions that may have good ratio properties."""
    inits = []
    
    inits.append(np.ones((n, n)) / (n * n))
    
    for diag_weight in [0.1, 0.3, 0.5]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, i] = diag_weight
        P = P / P.sum()
        inits.append(P)
    
    for anti_weight in [0.1, 0.3, 0.5]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, n - 1 - i] = anti_weight
        P = P / P.sum()
        inits.append(P)
    
    for corner_weight in [0.2, 0.4]:
        P = np.ones((n, n)) * 0.01
        P[0, 0] = corner_weight
        P[n-1, n-1] = corner_weight
        P[0, n-1] = corner_weight
        P[n-1, 0] = corner_weight
        P = P / P.sum()
        inits.append(P)
    
    for _ in range(5):
        P = np.zeros((n, n))
        num_nonzero = rng.integers(10, 50)
        idx = rng.choice(n * n, size=num_nonzero, replace=False)
        P.flat[idx] = rng.dirichlet(np.ones(num_nonzero))
        P = P / P.sum()
        inits.append(P)
    
    return inits


def _scipy_optimize(P_init, matrices, n, time_limit=60):
    """Use scipy L-BFGS-B to optimize the log-ratio objective."""
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 0.0
        P_flat = np.maximum(params, 0)
        s = P_flat.sum()
        if s < 1e-15:
            return 0.0
        P_flat = P_flat / s
        ratio, _, _ = _compute_ratio(P_flat, matrices)
        return -ratio  # minimize negative ratio
    
    def gradient(params):
        if time.time() - start_time > time_limit:
            return np.zeros_like(params)
        P_flat = np.maximum(params, 0)
        s = P_flat.sum()
        if s < 1e-15:
            return np.zeros_like(params)
        P_flat = P_flat / s
        grad = _gradient_soft_log_ratio(P_flat, matrices, beta=10.0)
        return -grad  # negative because we minimize
    
    P_flat_init = P_init.flatten().copy()
    
    result = minimize(
        objective,
        P_flat_init,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': 5000, 'maxfun': 5000, 'gtol': 1e-8, 'disp': False}
    )
    
    P_opt = np.maximum(result.x, 0)
    s = P_opt.sum()
    if s > 1e-15:
        P_opt = P_opt / s
    else:
        P_opt = np.ones(n * n) / (n * n)
    
    ratio_opt, _, _ = _compute_ratio(P_opt, matrices)
    return P_opt.reshape(n, n), ratio_opt


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize P to maximize H_{-1} / max(H_0, H_1, H_2) using precomputed
    pooling matrices, gradient ascent, and scipy L-BFGS-B.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 620
    
    matrices = _precompute_pooling_matrices(n)
    
    rng = np.random.default_rng(42)
    
    candidates = _structured_init(n, rng)
    
    for _ in range(15):
        candidates.append(_random_init(n, rng))
    
    best_P = None
    best_ratio = -np.inf
    
    candidate_ratios = []
    for P_init in candidates:
        if time.time() > deadline - 120:
            break
        P_flat = P_init.flatten()
        ratio, _, _ = _compute_ratio(P_flat, matrices)
        candidate_ratios.append((ratio, P_init))
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P_init.copy()
    
    candidate_ratios.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 1: Gradient ascent from top candidates
    for ratio, P_init in candidate_ratios[:10]:
        if time.time() > deadline - 180:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 60:
            break
        
        attempt_time = min(remaining_time / 3, 80)
        
        P_opt, ratio_opt = _gradient_ascent(
            P_init, matrices, n,
            max_iters=5000,
            time_limit=attempt_time,
            beta=10.0
        )
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt.copy()
    
    # Phase 2: Scipy L-BFGS-B from best solution
    if best_P is not None and time.time() < deadline - 100:
        remaining_time = deadline - time.time() - 20
        if remaining_time > 30:
            P_scipy, ratio_scipy = _scipy_optimize(
                best_P, matrices, n,
                time_limit=min(remaining_time, 120)
            )
            if ratio_scipy > best_ratio:
                best_ratio = ratio_scipy
                best_P = P_scipy.copy()
    
    # Phase 3: Polish with gradient ascent (higher beta, smaller steps)
    if best_P is not None and time.time() < deadline - 30:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 15:
            P_final, ratio_final = _gradient_ascent(
                best_P, matrices, n,
                max_iters=10000,
                time_limit=remaining_time * 0.5,
                beta=10.0,
                polish=False
            )
            if ratio_final > best_ratio:
                best_ratio = ratio_final
                best_P = P_final.copy()
        
        if time.time() < deadline - 15:
            remaining_time = deadline - time.time() - 10
            if remaining_time > 10:
                P_final2, ratio_final2 = _gradient_ascent(
                    best_P, matrices, n,
                    max_iters=50000,
                    time_limit=remaining_time,
                    beta=50.0,
                    polish=True,
                    step_size=0.001
                )
                if ratio_final2 > best_ratio:
                    best_ratio = ratio_final2
                    best_P = P_final2.copy()
    
    # Phase 4: Final scipy polish
    if best_P is not None and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            P_final3, ratio_final3 = _scipy_optimize(
                best_P, matrices, n,
                time_limit=remaining_time
            )
            if ratio_final3 > best_ratio:
                best_ratio = ratio_final3
                best_P = P_final3.copy()
    
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-15:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)
    
    return best_P


# EVOLVE-BLOCK-END
