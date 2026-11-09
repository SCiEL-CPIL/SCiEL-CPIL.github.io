import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    time_limit = 135.0  # safety margin before 150s kill

    def compute_dists(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists

    def get_unique_dists(dists):
        iu = np.triu_indices(8, k=1)
        return dists[iu]

    def compute_score_log(points):
        dists = compute_dists(points)
        d_pairs = get_unique_dists(dists)
        if np.min(d_pairs) < 1e-12:
            return -np.inf
        log_prod = np.sum(np.log(d_pairs))
        D = np.max(d_pairs)
        log_D = np.log(D)
        score_log = log_prod - 28.0 * log_D
        return score_log

    def compute_score(points):
        sl = compute_score_log(points)
        if np.isinf(sl) or np.isnan(sl):
            return 0.0
        return np.exp(sl)

    def objective_soft(params, alpha):
        points = params.reshape(8, 2)
        dists = compute_dists(points)
        d_pairs = get_unique_dists(dists)
        if np.min(d_pairs) < 1e-12:
            return 1e10
        log_prod = np.sum(np.log(d_pairs))
        max_d = np.max(d_pairs)
        exp_terms = np.exp(alpha * (d_pairs - max_d))
        S = np.sum(exp_terms)
        log_D = (np.log(S) + alpha * max_d) / alpha
        score_log = log_prod - 28.0 * log_D
        if np.isinf(score_log) or np.isnan(score_log):
            return 1e10
        return -score_log

    def gradient_soft(params, alpha):
        points = params.reshape(8, 2)
        dists = compute_dists(points)
        i_idx, j_idx = np.triu_indices(8, k=1)
        d_ij = dists[i_idx, j_idx]
        if np.min(d_ij) < 1e-12:
            return np.zeros(16)
        
        grad_log_prod = np.zeros((8, 2))
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d = d_ij[k]
            if d < 1e-12:
                continue
            diff = points[i] - points[j]
            term = diff / (d * d)
            grad_log_prod[i] += term
            grad_log_prod[j] -= term

        max_d = np.max(d_ij)
        exp_terms = np.exp(alpha * (d_ij - max_d))
        S = np.sum(exp_terms)
        weights = exp_terms / S
        
        grad_log_D = np.zeros((8, 2))
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d = d_ij[k]
            if d < 1e-12:
                continue
            w = weights[k]
            diff = points[i] - points[j]
            term = w * diff / d
            grad_log_D[i] += term
            grad_log_D[j] -= term

        grad = -(grad_log_prod - 28.0 * grad_log_D)
        return grad.flatten()

    def objective_exact(params):
        points = params.reshape(8, 2)
        score_log = compute_score_log(points)
        if np.isinf(score_log) or np.isnan(score_log):
            return 1e10
        return -score_log

    def refine_annealed(points, alpha_start=5.0, alpha_end=50000.0, steps=12):
        alphas = np.logspace(np.log10(alpha_start), np.log10(alpha_end), steps)
        current_points = points
        for alpha in alphas:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 8:
                break
            try:
                res = minimize(objective_soft, current_points.flatten(),
                               method='L-BFGS-B',
                               jac=lambda x: gradient_soft(x, alpha),
                               options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
                if res.fun < 1e9:
                    current_points = res.x.reshape(8, 2)
            except Exception:
                pass
        return current_points

    def polish_nelder_mead(points):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            return points
        try:
            res = minimize(objective_exact, points.flatten(),
                           method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < 1e9:
                return res.x.reshape(8, 2)
        except Exception:
            pass
        return points

    def polish_lbfgs_high_alpha(points, alpha=100000.0):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 8:
            return points
        try:
            res = minimize(objective_soft, points.flatten(),
                           method='L-BFGS-B',
                           jac=lambda x: gradient_soft(x, alpha),
                           options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < 1e9:
                return res.x.reshape(8, 2)
        except Exception:
            pass
        return points

    # Initial guesses
    inits = []
    
    # Regular octagon
    angles = np.arange(8) * (np.pi / 4)
    oct_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(oct_pts)

    # Two squares with different radii
    for ratio in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
        R1 = 1.0
        R2 = ratio
        s = R2 / np.sqrt(2)
        pts = np.array([
            [R1, 0], [0, R1], [-R1, 0], [0, -R1],
            [s, s], [-s, s], [-s, -s], [s, -s]
        ])
        inits.append(pts)

    # Rotated two squares
    for ratio in [0.6, 0.8, 1.0, 1.2, 1.5]:
        for rot in [np.pi/8, np.pi/16, 3*np.pi/16]:
            R1 = 1.0
            R2 = ratio
            s = R2 / np.sqrt(2)
            pts1 = np.array([
                [R1, 0], [0, R1], [-R1, 0], [0, -R1]
            ])
            pts2 = np.array([
                [s, s], [-s, s], [-s, -s], [s, -s]
            ])
            c, s_r = np.cos(rot), np.sin(rot)
            rot_mat = np.array([[c, -s_r], [s_r, c]])
            pts2_rot = pts2 @ rot_mat.T
            pts = np.vstack([pts1, pts2_rot])
            inits.append(pts)

    # Random convex polygons
    rng = np.random.default_rng(12345)
    for _ in range(20):
        gaps = rng.dirichlet(np.ones(8) * 0.5) * 2 * np.pi
        angles = np.cumsum(np.append(0, gaps[:-1]))
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        inits.append(pts)

    # Random perturbations of regular octagon
    for _ in range(15):
        pert = rng.normal(0, 0.3, size=(8, 2))
        pts = oct_pts + pert
        inits.append(pts)

    # Random points in disk
    for _ in range(10):
        pts = rng.normal(0, 1, size=(8, 2))
        inits.append(pts)

    # Sort inits by exact score, keep best ones
    scored_inits = []
    for pts in inits:
        s = compute_score_log(pts)
        scored_inits.append((s, pts))
    scored_inits.sort(key=lambda x: -x[0])
    inits = [pts for _, pts in scored_inits[:20]]

    best_points = None
    best_score = -np.inf

    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break

        try:
            refined = refine_annealed(init, alpha_start=5.0, alpha_end=50000.0, steps=12)
        except Exception:
            refined = init

        try:
            polished = polish_nelder_mead(refined)
        except Exception:
            polished = refined

        try:
            polished = polish_lbfgs_high_alpha(polished, alpha=100000.0)
        except Exception:
            pass

        try:
            polished = polish_nelder_mead(polished)
        except Exception:
            pass

        score_log = compute_score_log(polished)
        if score_log > best_score:
            best_score = score_log
            best_points = polished

    # Differential evolution for global search if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            def de_objective(params):
                points = params.reshape(8, 2)
                score_log = compute_score_log(points)
                if np.isinf(score_log) or np.isnan(score_log):
                    return 1e10
                return -score_log
            
            bounds = [(-2, 2)] * 16
            de_result = differential_evolution(de_objective, bounds, 
                                               maxiter=100, popsize=15,
                                               seed=42, tol=1e-12,
                                               polish=False)
            if de_result.fun < 1e9:
                de_points = de_result.x.reshape(8, 2)
                de_points = refine_annealed(de_points, alpha_start=10.0, alpha_end=50000.0, steps=8)
                de_points = polish_nelder_mead(de_points)
                de_points = polish_lbfgs_high_alpha(de_points, alpha=100000.0)
                de_points = polish_nelder_mead(de_points)
                
                de_score_log = compute_score_log(de_points)
                if de_score_log > best_score:
                    best_score = de_score_log
                    best_points = de_points
        except Exception:
            pass

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
        angles = np.arange(8) * (np.pi / 4)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_points = best_points * (2.0 / (2 * np.cos(np.pi/8)))

    return best_points.tolist()
