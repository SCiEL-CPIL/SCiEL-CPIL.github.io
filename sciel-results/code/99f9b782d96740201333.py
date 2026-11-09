import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0

    def compute_dists(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists

    def get_unique_dists(dists):
        iu = np.triu_indices(8, k=1)
        return dists[iu]

    def compute_score_log(points, alpha=1000.0):
        dists = compute_dists(points)
        d_pairs = get_unique_dists(dists)
        
        if np.min(d_pairs) < 1e-10:
            return -np.inf, None, None

        log_prod = np.sum(np.log(d_pairs))
        
        if np.isinf(alpha):
            D = np.max(d_pairs)
            log_D = np.log(D)
        else:
            max_d = np.max(d_pairs)
            exp_terms = np.exp(alpha * (d_pairs - max_d))
            S = np.sum(exp_terms)
            log_S = np.log(S)
            log_D = (np.log(S) + alpha * max_d) / alpha
            
        score_log = log_prod - 28.0 * log_D
        return score_log, d_pairs, log_D

    def objective_soft(params, alpha):
        points = params.reshape(8, 2)
        score_log, _, _ = compute_score_log(points, alpha)
        if np.isinf(score_log):
            return 1e10
        return -score_log

    def gradient_soft(params, alpha):
        points = params.reshape(8, 2)
        dists = compute_dists(points)
        d_pairs = get_unique_dists(dists)
        
        if np.min(d_pairs) < 1e-10:
            return np.zeros(16)

        i_idx, j_idx = np.triu_indices(8, k=1)
        d_ij = dists[i_idx, j_idx]
        
        grad_log_prod = np.zeros((8, 2))
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d = d_ij[k]
            if d < 1e-10: continue
            diff = points[i] - points[j]
            term = diff / (d * d)
            grad_log_prod[i] += term
            grad_log_prod[j] -= term

        if np.isinf(alpha):
            alpha_eff = 1000.0
        else:
            alpha_eff = alpha

        max_d = np.max(d_ij)
        exp_terms = np.exp(alpha_eff * (d_ij - max_d))
        S = np.sum(exp_terms)
        weights = exp_terms / S
        
        grad_log_D = np.zeros((8, 2))
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d = d_ij[k]
            if d < 1e-10: continue
            w = weights[k]
            diff = points[i] - points[j]
            term = w * diff / d
            grad_log_D[i] += term
            grad_log_D[j] -= term

        grad = -(grad_log_prod - 28.0 * grad_log_D)
        
        return grad.flatten()

    def refine_alpha(points, alpha_start=10.0, alpha_end=5000.0, steps=8):
        alphas = np.logspace(np.log10(alpha_start), np.log10(alpha_end), steps)
        current_points = points
        for alpha in alphas:
            res = minimize(objective_soft, current_points.flatten(), 
                           method='L-BFGS-B', 
                           jac=lambda x: gradient_soft(x, alpha),
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < 1e9:
                current_points = res.x.reshape(8, 2)
        return current_points

    def polish_exact(points):
        # Use high alpha soft-max for smooth polishing
        def obj_high_alpha(params):
            points = params.reshape(8, 2)
            score_log, _, _ = compute_score_log(points, 10000.0)
            if np.isinf(score_log):
                return 1e10
            return -score_log
        
        def grad_high_alpha(params):
            return gradient_soft(params, 10000.0)

        res = minimize(obj_high_alpha, points.flatten(), method='L-BFGS-B',
                       jac=grad_high_alpha,
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < 1e9:
            return res.x.reshape(8, 2)
        return points

    # Initial guesses
    inits = []
    
    # Regular octagon
    angles = np.arange(8) * (np.pi / 4)
    oct_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(oct_pts)

    # Two squares
    for ratio in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        R1 = 1.0
        R2 = ratio
        s = R2 / np.sqrt(2)
        pts = np.array([
            [R1, 0], [0, R1], [-R1, 0], [0, -R1],
            [s, s], [-s, s], [-s, -s], [s, -s]
        ])
        inits.append(pts)
        
    # Random perturbations of best symmetric
    rng = np.random.default_rng(42)
    best_sym = None
    best_sym_score = -np.inf
    for pts in inits:
        score_log, _, _ = compute_score_log(pts, np.inf)
        if score_log > best_sym_score:
            best_sym_score = score_log
            best_sym = pts

    for _ in range(15):
        pert = rng.normal(0, 0.2, size=best_sym.shape)
        inits.append(best_sym + pert)
        
    # Add a few random points
    for _ in range(5):
        pts = rng.normal(0, 1, size=(8, 2))
        inits.append(pts)

    best_points = None
    best_score = -np.inf

    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break

        # Refine with annealed alpha
        try:
            refined = refine_alpha(init, alpha_start=5.0, alpha_end=5000.0, steps=8)
        except Exception:
            refined = init

        # Polish with high-alpha soft objective
        try:
            polished = polish_exact(refined)
        except Exception:
            polished = refined

        score_log, _, _ = compute_score_log(polished, np.inf)
        if score_log > best_score:
            best_score = score_log
            best_points = polished

    # Final check and normalization
    if best_points is None:
        best_points = oct_pts

    # Normalize to diameter 2
    dists = compute_dists(best_points)
    D = np.max(dists)
    if D > 0:
        best_points = best_points * (2.0 / D)
        
    # Verify
    dists = compute_dists(best_points)
    iu = np.triu_indices(8, k=1)
    if np.min(dists[iu]) < 1e-6 or np.any(np.abs(best_points) >= 1e6):
        # Fallback
        angles = np.arange(8) * (np.pi / 4)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_points = best_points * (2.0 / (2 * np.cos(np.pi/8)))

    return best_points.tolist()
