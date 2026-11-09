import numpy as np
from scipy.optimize import minimize, minimize as _minimize_nm
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_score(pts):
        """Compute Delta/n^n after scaling to diameter 2."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D < 1e-15:
            return 0.0
        scaled = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        log_prod = np.sum(np.log(np.maximum(scaled[mask], 1e-300)))
        return np.exp(log_prod) / (n ** n)

    def log_sum_dist(pts):
        """Sum of log distances over unique pairs (for optimization)."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d = dists[mask]
        if np.any(d < 1e-10):
            return -1e10
        return np.sum(np.log(d))

    def max_dist(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return np.max(dists[mask])

    # Best tracking
    best_points = regular_octagon()
    best_score = compute_score(best_points)

    # === Smooth epigraph formulation using log-sum-exp soft-max ===
    # Objective: maximize sum(log(d_ij)) - n * log(D) where D = max(d_ij)
    # Equivalent to: maximize sum(log(d_ij)) subject to d_ij <= 2 for all i,j
    # We use an auxiliary variable t (soft diameter) with log-sum-exp:
    #   t = (1/beta) * log(sum(exp(beta * d_ij^2)))  approximates max(d_ij) as beta -> inf
    # Then maximize: sum(log(d_ij)) - n * log(t)
    # This is smooth and differentiable for any finite beta.

    def soft_diam_obj(x, beta):
        """Smooth objective using log-sum-exp approximation of diameter."""
        pts = x.reshape(8, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d = dists[mask]
        if np.any(d < 1e-12):
            return 1e10
        # Log-sum-exp for soft max of distances
        d_max = np.max(d)
        log_sum_exp = d_max + np.log(np.sum(np.exp(beta * (d - d_max))))
        soft_D = log_sum_exp / beta
        # Objective: sum(log(d)) - n * log(soft_D)
        obj = np.sum(np.log(d)) - n * np.log(soft_D)
        return -obj

    def soft_diam_grad(x, beta):
        """Gradient of soft_diam_obj for L-BFGS-B."""
        pts = x.reshape(8, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d = dists[mask]
        if np.any(d < 1e-12):
            return np.zeros_like(x)
        
        # Soft diameter via log-sum-exp
        d_max = np.max(d)
        exp_vals = np.exp(beta * (d - d_max))
        log_sum_exp = d_max + np.log(np.sum(exp_vals))
        soft_D = log_sum_exp / beta
        
        # Gradient of sum(log(d_ij)) w.r.t. points
        grad_pts = np.zeros((8, 2))
        for i in range(8):
            for j in range(i + 1, 8):
                diff = pts[i] - pts[j]
                dij = np.linalg.norm(diff)
                if dij < 1e-12:
                    continue
                # d(log(dij))/d(pts[i]) = (pts[i] - pts[j]) / dij^2
                grad_pts[i] += diff / (dij * dij)
                grad_pts[j] -= diff / (dij * dij)
        
        # Gradient of -n*log(soft_D) w.r.t. points
        # soft_D = (1/beta) * log(sum(exp(beta * d_k)))
        # d(soft_D)/d(pts) = (1/beta) * (1/sum(exp(beta*d_k))) * sum(beta * exp(beta*d_k) * d(d_k)/d(pts))
        # = sum(exp(beta*d_k) * d(d_k)/d(pts)) / sum(exp(beta*d_k))
        # = weighted average of d(d_k)/d(pts) with weights proportional to exp(beta*d_k)
        
        weights = exp_vals / np.sum(exp_vals)
        grad_soft_D = np.zeros((8, 2))
        idx = 0
        for i in range(8):
            for j in range(i + 1, 8):
                diff = pts[i] - pts[j]
                dij = np.linalg.norm(diff)
                if dij < 1e-12:
                    idx += 1
                    continue
                w = weights[idx]
                grad_soft_D[i] += w * diff / dij
                grad_soft_D[j] -= w * diff / dij
                idx += 1
        
        # Total gradient: grad(sum(log(d))) - n * grad(log(soft_D))
        # grad(log(soft_D)) = grad(soft_D) / soft_D
        total_grad_pts = grad_pts - n * grad_soft_D / soft_D
        return total_grad_pts.flatten()

    # === Strategy 1: Soft-diameter optimization with L-BFGS-B ===
    full_bounds = [(-2.0, 2.0), (-2.0, 2.0)] * 8
    oct_pts = regular_octagon()
    
    if time.time() - start_time < deadline:
        for beta in [10, 30, 100, 300]:
            if time.time() - start_time > deadline:
                break
            x0 = oct_pts.flatten()
            try:
                res = minimize(soft_diam_obj, x0, args=(beta,), method='L-BFGS-B',
                               bounds=full_bounds, jac=lambda x: soft_diam_grad(x, beta),
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.success or res.fun < -0.5:
                    pts = res.x.reshape(8, 2)
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
            except Exception:
                pass

    # === Nelder-Mead polish on the exact (non-smooth) objective ===
    def nm_obj(x):
        pts = x.reshape(8, 2)
        return -compute_score(pts)

    def nm_polish(pts0, maxiter=3000):
        try:
            x0 = pts0.flatten()
            res = _minimize_nm(nm_obj, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-12,
                                        'fatol': 1e-12, 'adaptive': True})
            pts = res.x.reshape(8, 2)
            d2 = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            m = np.triu(np.ones((n, n), dtype=bool), k=1)
            if np.min(d2[m]) < 1e-8:
                return None
            return pts
        except Exception:
            return None

    if time.time() - start_time < deadline:
        polished = nm_polish(best_points, maxiter=5000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 2: Symmetric with x-axis mirror ===
    def build_sym(x):
        pts = np.zeros((8, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [x[0], x[1]]
        pts[3] = [x[0], -x[1]]
        pts[4] = [x[2], x[3]]
        pts[5] = [x[2], -x[3]]
        pts[6] = [x[4], x[5]]
        pts[7] = [x[4], -x[5]]
        return pts

    def obj_sym(x):
        pts = build_sym(x)
        return -log_sum_dist(pts)

    def sym_constraint(i, j):
        def c(x):
            pts = build_sym(x)
            return 2.0 - np.linalg.norm(pts[i] - pts[j])
        return c

    sym_constraints = []
    for i in [2, 4, 6]:
        sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(0, i)})
        sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(1, i)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 4)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 6)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(4, 6)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 5)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 7)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(4, 7)})
    sym_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - x[1]})
    sym_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - x[3]})
    sym_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - x[5]})

    sym_bounds = [(-1.5, 1.5), (1e-4, 1.0),
                  (-1.5, 1.5), (1e-4, 1.0),
                  (-1.5, 1.5), (1e-4, 1.0)]

    sym_inits = [
        np.array([0.707, 0.707, 0.0, 1.0, -0.707, 0.707]),
        np.array([0.5, 0.8, -0.3, 0.9, -0.8, 0.5]),
        np.array([0.0, 1.0, 0.8, 0.6, -0.8, 0.6]),
        np.array([0.9, 0.3, 0.0, 1.0, -0.9, 0.3]),
        np.array([0.3, 0.95, -0.5, 0.85, -0.95, 0.4]),
    ]

    for init in sym_inits:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(obj_sym, init, method='SLSQP',
                           bounds=sym_bounds, constraints=sym_constraints,
                           options={'maxiter': 300, 'ftol': 1e-12})
            if res.success:
                pts = build_sym(res.x)
                d = max_dist(pts)
                if 1e-5 < np.min(np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)[np.triu(np.ones((n,n), dtype=bool), k=1)]):
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
        except Exception:
            pass

    if time.time() - start_time < deadline:
        polished = nm_polish(best_points, maxiter=4000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 3: Full 16-DOF SLSQP optimization ===
    def obj_full(x):
        pts = x.reshape(8, 2)
        return -log_sum_dist(pts)

    full_constraints = []
    for i in range(8):
        for j in range(i+1, 8):
            def c(x, i=i, j=j):
                pts = x.reshape(8, 2)
                return 2.0 - np.linalg.norm(pts[i] - pts[j])
            full_constraints.append({'type': 'ineq', 'fun': c})

    if time.time() - start_time < deadline:
        try:
            res = minimize(obj_full, full_init, method='SLSQP',
                           bounds=full_bounds, constraints=full_constraints,
                           options={'maxiter': 400, 'ftol': 1e-11})
            if res.success:
                pts = res.x.reshape(8, 2)
                d = max_dist(pts)
                if d > 1e-5:
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
        except Exception:
            pass

    # === Strategy 4: Random perturbations + local optimize ===
    np.random.seed(123)
    if time.time() - start_time < deadline:
        for trial in range(15):
            if time.time() - start_time > deadline:
                break
            x0 = np.random.uniform(-0.9, 0.9, 3)
            y0 = np.random.uniform(0.15, 0.95, 3)
            init_rand = np.concatenate([x0, y0])

            try:
                res = minimize(obj_sym, init_rand, method='SLSQP',
                               bounds=sym_bounds, constraints=sym_constraints,
                               options={'maxiter': 200, 'ftol': 1e-11})
                if res.success:
                    pts = build_sym(res.x)
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
            except Exception:
                pass

    if time.time() - start_time < deadline:
        polished = nm_polish(best_points, maxiter=5000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 5: Perturbed octagon full optimization ===
    if time.time() - start_time < deadline:
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for _ in range(5):
            if time.time() - start_time > deadline:
                break
            pert = angles + np.random.uniform(-0.3, 0.3, 8)
            pts_init = np.column_stack([np.cos(pert), np.sin(pert)])
            x_init = pts_init.flatten()

            try:
                res = minimize(obj_full, x_init, method='SLSQP',
                               bounds=full_bounds, constraints=full_constraints,
                               options={'maxiter': 300, 'ftol': 1e-11})
                if res.success:
                    pts = res.x.reshape(8, 2)
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
            except Exception:
                pass

    if time.time() - start_time < deadline:
        polished = nm_polish(best_points, maxiter=5000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 6: High-precision final polish ===
    if time.time() - start_time < deadline:
        try:
            x0 = best_points.flatten()
            res = minimize(obj_full, x0, method='SLSQP',
                           bounds=full_bounds, constraints=full_constraints,
                           options={'maxiter': 2000, 'ftol': 1e-15})
            pts = res.x.reshape(8, 2)
            s = compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
            polished = nm_polish(pts, maxiter=8000)
            if polished is not None:
                s2 = compute_score(polished)
                if s2 > best_score:
                    best_score = s2
                    best_points = polished
        except Exception:
            pass

    # Final validation
    diffs = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.min(dists[mask]) < 1e-6:
        best_points = regular_octagon()

    if np.any(np.abs(best_points) > 1e6):
        D = max_dist(best_points)
        if D > 0:
            best_points = best_points * (1e5 / D)

    return best_points.tolist()
