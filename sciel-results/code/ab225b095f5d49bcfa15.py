import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random

def _pdist_pair_indices(n):
    """Return list of (i, j) pairs in pdist order."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    timeout = 600.0  # Safety margin from 630s limit
    
    best_score = 0.0
    best_points = None
    pairs = _pdist_pair_indices(n)
    m = len(pairs)  # 120

    # Helper to compute score
    def compute_score(pts):
        if pts.shape != (16, 2):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Smooth max via log-sum-exp: log(sum exp(d_i^2 / tau)) * tau / 2 approximates max(d_i)
    def smooth_max(dists, tau):
        """Smooth approximation of max(dists) using log-sum-exp of squared distances."""
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15:
            return 0.0
        # log-sum-exp for numerical stability
        log_sum = m + np.log(np.sum(np.exp(d2 - m)))
        return np.sqrt(log_sum) * np.sqrt(tau) / np.sqrt(tau)  # = sqrt(log_sum)
        # Actually: max approx = sqrt(log(sum(exp(d^2/tau))) * tau)
        # Let's use: soft_max = sqrt(tau * log(sum(exp(d^2/tau))))
        # With tau small, this approaches max(d)

    def soft_max_dist(dists, tau=0.01):
        """Smooth max of distances: sqrt(tau * log(sum(exp(d^2/tau))))"""
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15:
            return 0.0
        log_sum = m / tau + np.log(np.sum(np.exp((d2 - m) / tau)))
        return np.sqrt(tau * log_sum)

    def soft_min_dist(dists, tau=0.01):
        """Smooth min of distances: -sqrt(tau * log(sum(exp(-d^2/tau))))"""
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15:
            return 0.0
        log_sum = -m / tau + np.log(np.sum(np.exp((d2 - m) / tau)))
        return -np.sqrt(tau * log_sum)

    # SCP: minimize soft_max(dists) subject to soft_min(dists) >= t
    # Using penalty method: minimize soft_max(dists) + mu * max(0, t - soft_min(dists))^2
    def scp_objective(x, t, mu, tau):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        smax = soft_max_dist(dists, tau)
        smin = soft_min_dist(dists, tau)
        penalty = mu * max(0.0, t - smin) ** 2
        return smax + penalty

    def scp_gradient(x, t, mu, tau):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        grad = np.zeros_like(x)
        
        # Gradient of soft_max = sqrt(tau * log(sum(exp(d^2/tau))))
        # d/dx_i = (1/(2*soft_max)) * tau * sum_j exp(d_ij^2/tau) * 2*d_ij * (x_i - x_j) / tau
        #        = sum_j exp(d_ij^2/tau) * d_ij * (x_i - x_j) / soft_max
        # More carefully:
        # f = sqrt(tau * L) where L = log(sum exp(d^2/tau))
        # df/dx = (tau * dL/dx) / (2*f)
        # dL/dx_i = sum_j [exp(d_ij^2/tau) * 2*d_ij/tau * (x_i-x_j)] / sum exp(d^2/tau)
        # So df/dx_i = tau/(2*f) * sum_j exp(d_ij^2/tau) * 2*d_ij/tau * (x_i-x_j) / sum_exp
        #            = 1/f * sum_j exp(d_ij^2/tau) * d_ij * (x_i-x_j) / sum_exp
        
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15:
            return grad
        exp_d2 = np.exp((d2 - m) / tau)
        sum_exp = np.sum(exp_d2)
        smax = soft_max_dist(dists, tau)
        
        if smax < 1e-15:
            return grad
        
        # Weight for each pair in soft_max gradient
        weights_max = exp_d2 * dists / sum_exp  # proportional to contribution
        
        for idx, (i, j) in enumerate(pairs):
            vec = pts[i] - pts[j]
            w = weights_max[idx] / smax
            grad[2*i:2*i+2] += w * vec
            grad[2*j:2*j+2] -= w * vec
        
        # Gradient of penalty term: mu * max(0, t - smin)^2
        smin = soft_min_dist(dists, tau)
        if t > smin and mu > 0:
            # d/dx of (t - smin)^2 = -2*(t - smin) * dsmin/dx
            # smin = -sqrt(tau * log(sum(exp(-d^2/tau))))
            # dsmin/dx_i = -1/(2*smin_abs) * tau * d/dx_i [log(sum(exp(-d^2/tau)))]  ... 
            # Let's compute numerically-friendly:
            # smin = -sqrt(tau * Lmin) where Lmin = log(sum(exp(-d^2/tau)))
            # Actually soft_min = -sqrt(tau * log(sum(exp(-d^2/tau))))
            # d(smin)/dx_i: smin = -sqrt(tau*L), d(smin)/dx = -tau*dL/dx / (2*sqrt(tau*L)) = -dL/dx/(2*sqrt(L/tau))
            # L = log(sum(exp(-d^2/tau)))
            # dL/dx_i = sum_j exp(-d_ij^2/tau) * (-2*d_ij/tau) * (x_i-x_j) / sum exp(-d^2/tau)
            # So d(smin)/dx_i = -1/(2*sqrt(L/tau)) * sum_j exp(-d^2/tau)*(-2*d_ij/tau)*(x_i-x_j)/sum_exp_neg
            #                = 1/(sqrt(tau*L)) * sum_j exp(-d^2/tau)*d_ij*(x_i-x_j)/sum_exp_neg
            # Note: smin = -sqrt(tau*L) so sqrt(tau*L) = -smin
            
            d2_neg = -d2
            m_neg = np.max(d2_neg)  # = -min(d2)
            exp_neg = np.exp((d2_neg - m_neg) / tau)
            sum_exp_neg = np.sum(exp_neg)
            Lmin = m_neg / tau + np.log(sum_exp_neg)
            
            if Lmin > 1e-15:
                sqrt_tau_L = np.sqrt(tau * Lmin)
                weights_min = exp_neg * dists / sum_exp_neg
                penalty_factor = -2.0 * mu * (t - smin) / sqrt_tau_L
                
                for idx, (i, j) in enumerate(pairs):
                    vec = pts[i] - pts[j]
                    w = penalty_factor * weights_min[idx]
                    grad[2*i:2*i+2] += w * vec
                    grad[2*j:2*j+2] -= w * vec
        
        return grad

    def normalize_pts(pts):
        """Center and scale points so that d_max = 1."""
        pts = pts - np.mean(pts, axis=0)
        dm = np.max(pdist(pts))
        if dm > 0:
            pts = pts / dm
        return pts

    # SCP-based optimization: bisection on t (min distance lower bound)
    # For a given t, solve: min soft_max(dists) s.t. soft_min(dists) >= t
    # Then bisect on t to find the largest feasible t, which gives optimal ratio.
    
    def solve_scp(t, x0, max_iter=200, tau=0.01):
        """Solve the SCP problem for a given t using L-BFGS-B with penalty."""
        x = x0.flatten().copy()
        mu = 1.0
        best_x = x.copy()
        best_val = scp_objective(x, t, mu, tau)
        
        for it in range(max_iter):
            if time.time() - start_time > timeout - 15:
                break
            
            try:
                res = minimize(scp_objective, x, method='L-BFGS-B',
                              jac=lambda xx: scp_gradient(xx, t, mu, tau),
                              args=(t, mu, tau),
                              options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12})
                x = res.x
            except Exception:
                break
            
            val = scp_objective(x, t, mu, tau)
            if val < best_val:
                best_val = val
                best_x = x.copy()
            
            # Check constraint satisfaction
            pts = x.reshape(-1, 2)
            dists = pdist(pts)
            smin = soft_min_dist(dists, tau)
            smax = soft_max_dist(dists, tau)
            
            if smin >= t - 1e-6:
                # Constraint satisfied, can tighten
                break
            else:
                # Increase penalty
                mu *= 2.0
                if mu > 1e6:
                    break
        
        return best_x, best_val
    
    def scp_bisection(x0, t_lo, t_hi, max_bisect=20, tau=0.01):
        """Bisect on t to find optimal min/max ratio."""
        best_x = x0.flatten().copy()
        best_score = 0.0
        
        # First, find a feasible t
        # Start with t_lo, increase until infeasible
        t = t_lo
        while t < t_hi:
            if time.time() - start_time > timeout - 10:
                break
            x_sol, _ = solve_scp(t, best_x, max_iter=100, tau=tau)
            pts = x_sol.reshape(-1, 2)
            dists = pdist(pts)
            smin = soft_min_dist(dists, tau)
            if smin < t - 1e-4:
                # Infeasible, t too high
                t_hi = t
                break
            else:
                t_lo = t
                best_x = x_sol
                # Update score
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max > 0:
                    sc = (d_min / d_max) ** 2
                    if sc > best_score:
                        best_score = sc
                t = (t_lo + t_hi) / 2 if t_lo < t_hi else t * 1.05
        
        # Refine with bisection
        for _ in range(max_bisect):
            if time.time() - start_time > timeout - 10:
                break
            t_mid = (t_lo + t_hi) / 2
            x_sol, _ = solve_scp(t_mid, best_x, max_iter=150, tau=tau)
            pts = x_sol.reshape(-1, 2)
            dists = pdist(pts)
            smin = soft_min_dist(dists, tau)
            if smin >= t_mid - 1e-5:
                t_lo = t_mid
                best_x = x_sol
            else:
                t_hi = t_mid
            
            if t_hi - t_lo < 1e-4:
                break
        
        # Final solution at t_lo
        if time.time() - start_time > timeout - 10:
            x_final, _ = solve_scp(t_lo, best_x, max_iter=200, tau=tau)
            best_x = x_final
        
        return best_x, best_score

    # Objective 2: Minimize -sum(log(dists)) to encourage uniform spread (repulsive force)
    # This is a smooth surrogate that helps escape local minima where d_min is determined by a single "bad" pair.
    def objective_spread(x):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        dists = np.maximum(dists, 1e-9)
        return -np.sum(np.log(dists))

    def gradient_spread(x):
        pts = x.reshape(-1, 2)
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(i + 1, n):
                vec = pts[i] - pts[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    vec = np.array([1e-5, 1e-5])
                    dist = np.linalg.norm(vec)
                # Gradient of -sum(log d) is -sum(grad log d)
                # grad log d_ij w.r.t x_i is (x_i - x_j) / d^2
                # So grad obj w.r.t x_i is - (x_i - x_j) / d^2
                grad[2*i:2*i+2] -= vec / (dist * dist)
                grad[2*j:2*j+2] += vec / (dist * dist)
        return grad

    # Direct ratio objective with soft-min/soft-max for smoothness
    def objective_soft_ratio(x, tau=0.01):
        """Minimize soft_max / soft_min to maximize the ratio smoothly."""
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        smin = soft_min_dist(dists, tau)
        smax = soft_max_dist(dists, tau)
        if smin < 1e-15 or smax < 1e-15:
            return 0.0
        return smax / smin  # minimize this = maximize smin/smax

    def gradient_soft_ratio(x, tau=0.01):
        """Gradient of soft_max / soft_min."""
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        smin = soft_min_dist(dists, tau)
        smax = soft_max_dist(dists, tau)
        if smin < 1e-15 or smax < 1e-15:
            return np.zeros_like(x)
        
        # f = smax / smin
        # df = (dsmax * smin - smax * dsmin) / smin^2
        # = dsmax / smin - smax * dsmin / smin^2
        # = (1/smin) * dsmax - (smax/smin^2) * dsmin
        
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15:
            return np.zeros_like(x)
        
        grad = np.zeros_like(x)
        
        # dsmax/dx
        exp_d2 = np.exp((d2 - m) / tau)
        sum_exp = np.sum(exp_d2)
        weights_max = exp_d2 * dists / sum_exp / smax
        
        # dsmin/dx
        d2_neg = -d2
        m_neg = np.max(d2_neg)
        exp_neg = np.exp((d2_neg - m_neg) / tau)
        sum_exp_neg = np.sum(exp_neg)
        Lmin = m_neg / tau + np.log(sum_exp_neg)
        if Lmin > 1e-15:
            sqrt_tau_L = np.sqrt(tau * Lmin)
            weights_min = exp_neg * dists / sum_exp_neg / sqrt_tau_L
        else:
            weights_min = np.zeros_like(dists)
        
        coeff_max = 1.0 / smin
        coeff_min = -smax / (smin * smin)
        
        for idx, (i, j) in enumerate(pairs):
            vec = pts[i] - pts[j]
            w = coeff_max * weights_max[idx] + coeff_min * weights_min[idx]
            grad[2*i:2*i+2] += w * vec
            grad[2*j:2*j+2] -= w * vec
        
        return grad

    # Simulated Annealing
    def simulated_annealing(init_pts, max_iters=5000, T_start=0.1, T_end=0.001):
        """Run SA to improve the min/max distance ratio."""
        pts = init_pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        n_iters = max_iters
        T = T_start
        
        for it in range(n_iters):
            if time.time() - start_time > timeout - 30:
                break
                
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
            
            # Random perturbation
            delta = np.random.randn(n, 2) * T
            new_pts = pts + delta
            
            new_score = compute_score(new_pts)
            cur_score = compute_score(pts)
            
            # Accept if better, or with probability exp((new - cur) / T)
            if new_score >= cur_score:
                pts = new_pts
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            else:
                accept_prob = np.exp((new_score - cur_score) / max(T, 1e-10))
                if np.random.random() < accept_prob:
                    pts = new_pts
        
        return best_pts, best_score

    # Gradient-based local optimization
    def gradient_optimize(init_pts, max_iter=200, use_spread=False, use_soft_ratio=False):
        """Use L-BFGS-B with the ratio or spread objective."""
        x0 = init_pts.flatten()
        try:
            if use_spread:
                res = minimize(objective_spread, x0, method='L-BFGS-B', jac=gradient_spread,
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            elif use_soft_ratio:
                res = minimize(objective_soft_ratio, x0, method='L-BFGS-B', jac=gradient_soft_ratio,
                               args=(0.01,),
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            else:
                res = minimize(objective_soft_ratio, x0, method='L-BFGS-B', jac=gradient_soft_ratio,
                               args=(0.05,),
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_pts = res.x.reshape(-1, 2)
            return opt_pts
        except Exception:
            return init_pts

    # Combined approach: SA from good initializations, then gradient polish + SCP
    def improve_from_init(init_pts):
        """Improve a starting configuration using SA + gradient polish + SCP bisection."""
        if time.time() - start_time > timeout - 30:
            return init_pts, compute_score(init_pts)
        
        # Normalize
        pts = normalize_pts(init_pts)
        score = compute_score(pts)
        
        # Phase 1: Simulated Annealing
        sa_pts, sa_score = simulated_annealing(pts, max_iters=3000, T_start=0.05, T_end=0.0005)
        if sa_score > score:
            pts = sa_pts
            score = sa_score
        
        # Phase 2: Gradient polish (soft ratio objective)
        if time.time() - start_time > timeout - 20:
            return pts, score
        
        grad_pts = gradient_optimize(pts, max_iter=150, use_spread=False, use_soft_ratio=True)
        grad_pts = normalize_pts(grad_pts)
        grad_score = compute_score(grad_pts)
        
        if grad_score > score:
            pts = grad_pts
            score = grad_score
        
        # Phase 2.5: Gradient polish (Spread objective) - helps if stuck in local min of ratio
        if time.time() - start_time > timeout - 15:
            spread_pts = gradient_optimize(pts, max_iter=100, use_spread=True)
            spread_pts = normalize_pts(spread_pts)
            spread_score = compute_score(spread_pts)
            if spread_score > score:
                pts = spread_pts
                score = spread_score
        
        # Phase 2.7: SCP bisection to tighten the min-distance constraint
        if time.time() - start_time > timeout - 25:
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max > 0 and d_min > 0:
                t_lo = d_min * 0.95
                t_hi = d_max * 0.95
                scp_pts_flat, scp_score = scp_bisection(pts, t_lo, t_hi, max_bisect=15, tau=0.01)
                scp_pts = normalize_pts(scp_pts_flat.reshape(-1, 2))
                scp_actual_score = compute_score(scp_pts)
                if scp_actual_score > score:
                    pts = scp_pts
                    score = scp_actual_score
        
        # Phase 3: Another SA round from the improved point
        if time.time() - start_time > timeout - 15:
            sa2_pts, sa2_score = simulated_annealing(pts, max_iters=2000, T_start=0.02, T_end=0.0002)
            if sa2_score > score:
                pts = sa2_pts
                score = sa2_score
        
        # Final: SCP polish with tighter tau
        if time.time() - start_time > timeout - 12:
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max > 0 and d_min > 0:
                t_lo = d_min * 0.98
                t_hi = d_max * 0.98
                scp_pts_flat, _ = scp_bisection(pts, t_lo, t_hi, max_bisect=10, tau=0.005)
                scp_pts = normalize_pts(scp_pts_flat.reshape(-1, 2))
                scp_actual_score = compute_score(scp_pts)
                if scp_actual_score > score:
                    pts = scp_pts
                    score = scp_actual_score
        
        # Final gradient polish
        if time.time() - start_time > timeout - 10:
            final_pts = gradient_optimize(pts, max_iter=80, use_spread=False, use_soft_ratio=True)
            final_pts = normalize_pts(final_pts)
            final_score = compute_score(final_pts)
            
            if final_score > score:
                pts = final_pts
                score = final_score
        
        return pts, score

    # Generate diverse initializations
    initializations = []
    
    np.random.seed(42)
    
    # 1. Hexagonal lattice (structurally close to optimal packing)
    points_hex = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points_hex.append([x, y])
    points_hex = np.array(points_hex)
    dists_center = np.linalg.norm(points_hex, axis=1)
    idx = np.argsort(dists_center)[:n]
    initializations.append(points_hex[idx])
    
    # 2. Multiple hexagonal subsets with different centering
    for offset in [(0.3, 0.2), (-0.2, 0.4), (0.1, -0.3), (-0.4, -0.1)]:
        pts_h = points_hex + np.array(offset)
        dc = np.linalg.norm(pts_h, axis=1)
        idx_h = np.argsort(dc)[:n]
        initializations.append(pts_h[idx_h])
    
    # 3. Regular polygon (circle)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 4. Perturbed circles
    for _ in range(5):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.random.randn(n) * 0.1
        radii = 1 + 0.15 * np.random.randn(n)
        initializations.append(np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii]))
    
    # 5. Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    initializations.append(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    
    # 6. Random points
    for _ in range(8):
        initializations.append(np.random.randn(n, 2))
    
    # 7. Two-ring structure (inner 6, outer 10)
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initializations.append(np.vstack([inner_pts, outer_pts]))
    
    # 8. Three-ring structure
    r1_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    r1_pts = np.column_stack([0.3*np.cos(r1_angles), 0.3*np.sin(r1_angles)])
    r2_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r2_pts = np.column_stack([0.65*np.cos(r2_angles), 0.65*np.sin(r2_angles)])
    r3_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r3_pts = np.column_stack([np.cos(r3_angles), np.sin(r3_angles)])
    initializations.append(np.vstack([r1_pts, r2_pts, r3_pts]))

    # Process each initialization
    for init in initializations:
        if time.time() - start_time > timeout - 25:
            break
        
        init_norm = normalize_pts(init)
        init_score = compute_score(init_norm)
        
        # Track best initialization
        if init_score > best_score:
            best_score = init_score
            best_points = init_norm.copy()
        
        # Improve
        improved_pts, improved_score = improve_from_init(init_norm)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # Final: if we have time, do SCP refinement from the best point
    if best_points is not None and time.time() - start_time < timeout - 15:
        dists = pdist(best_points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max > 0 and d_min > 0:
            t_lo = d_min * 0.99
            t_hi = d_max * 0.99
            scp_pts_flat, _ = scp_bisection(best_points, t_lo, t_hi, max_bisect=20, tau=0.003)
            scp_pts = normalize_pts(scp_pts_flat.reshape(-1, 2))
            scp_score = compute_score(scp_pts)
            if scp_score > best_score:
                best_score = scp_score
                best_points = scp_pts.copy()
        
        # One more gradient polish
        if time.time() - start_time < timeout - 10:
            final_pts = gradient_optimize(best_points, max_iter=100, use_spread=False, use_soft_ratio=True)
            final_pts = normalize_pts(final_pts)
            final_score = compute_score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()

    # If no points were found or all failed, return a default
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, 2)
        best_points = normalize_pts(best_points)
        
    return best_points
