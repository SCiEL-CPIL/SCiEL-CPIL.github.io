import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, 1)
    n_pairs = len(iu[0])
    
    def objective(pts_flat):
        """Exact objective: (max dist / min dist)^2"""
        pts = pts_flat.reshape(N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        d_sq = dists_sq[iu]
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        
        if d_min_sq < 1e-12:
            return 1e10
        return d_max_sq / d_min_sq
    
    def soft_objective(pts_flat, beta):
        """
        Smooth surrogate using log-sum-exp.
        We want to minimize (max d_sq / min d_sq).
        
        Let S_max = logsumexp(d_sq) ~ log(sum(exp(d_sq))) ~ log(exp(max)) = max (approx for large beta)
        Actually, to approximate max, we use (1/beta) * logsumexp(beta * x).
        To approximate min, we use -(1/beta) * logsumexp(-beta * x).
        
        So soft_max_sq = (1/beta) * logsumexp(beta * d_sq)
        soft_min_sq = -(1/beta) * logsumexp(-beta * d_sq)
        
        Objective = soft_max_sq / soft_min_sq
        """
        pts = pts_flat.reshape(N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        d_sq = dists_sq[iu]
        
        # Numerical stability for logsumexp
        def logsumexp(x, axis=None):
            m = np.max(x, axis=axis, keepdims=True)
            # Handle case where m is -inf or +inf
            s = np.sum(np.exp(x - m), axis=axis, keepdims=True)
            return np.log(s) + m.squeeze(axis)
        
        # For large beta * x, exp might overflow.
        # d_sq is non-negative.
        # max term: beta * d_sq. Max is beta * max(d_sq).
        # min term: -beta * d_sq. Max is -beta * min(d_sq).
        
        max_d = np.max(d_sq)
        min_d = np.min(d_sq)
        
        if min_d < 1e-12:
            return 1e10
        
        # Approximation for stability
        # soft_max = (1/beta) * (beta*max_d + log(sum(exp(beta*(d_sq - max_d)))))
        exp_max_terms = np.exp(beta * (d_sq - max_d))
        soft_max_sq = (1.0/beta) * (beta * max_d + np.log(np.sum(exp_max_terms)))
        
        # soft_min = -(1/beta) * (-beta*min_d + log(sum(exp(-beta*(d_sq - min_d)))))
        # Note: -beta * d_sq. Max of this is -beta * min_d.
        exp_min_terms = np.exp(-beta * (d_sq - min_d))
        soft_min_sq = -(1.0/beta) * (-beta * min_d + np.log(np.sum(exp_min_terms)))
        
        if soft_min_sq < 1e-12:
            return 1e10
            
        return soft_max_sq / soft_min_sq

    def hexagonal_init():
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        pts = []
        # 4x4 block of hex lattice
        # Centers:
        # (-1, 0), (0, 0), (1, 0), (2, 0) ?
        # Let's just take a compact cluster from the infinite lattice
        for i in range(-2, 3):
            for j in range(-2, 3):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        # Sort by distance from origin to get the most central 16
        dists = np.sum(pts**2, axis=1)
        sorted_idx = np.argsort(dists)
        return pts[sorted_idx[:16]]

    def square_init():
        side = 4
        x = np.arange(side, dtype=float)
        y = np.arange(side, dtype=float)
        grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        grid -= grid.mean(axis=0)
        return grid

    def circle_init():
        # Points on a circle, slightly perturbed
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        r = 1.0
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        return pts

    def random_init():
        return np.random.rand(N, 2) - 0.5

    def run_optimization(init_pts, beta_schedule):
        """
        Run optimization with annealed beta (softness).
        """
        current = init_pts.flatten().copy()
        best_score_local = objective(current)
        best_pts_local = current.copy()
        
        for beta in beta_schedule:
            if time.time() > deadline - 5:
                break
            
            # Optimize the soft objective
            try:
                res = minimize(
                    lambda x: soft_objective(x, beta),
                    current,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < 1e9: # Sanity check
                    current = res.x
                    # Evaluate on exact objective
                    score = objective(current)
                    if score < best_score_local:
                        best_score_local = score
                        best_pts_local = current.copy()
            except Exception:
                pass
                
        return best_score_local, best_pts_local

    best_score_global = np.inf
    best_pts_global = None

    # Define beta schedules: Start soft, get stiff
    beta_schedule = [10, 50, 200, 1000, 5000]
    
    initial_guesses = [
        hexagonal_init(),
        square_init(),
        circle_init(),
        random_init(),
        random_init()
    ]

    for init_pts in initial_guesses:
        if time.time() > deadline - 15:
            break
            
        score, pts = run_optimization(init_pts, beta_schedule)
        
        if score < best_score_global:
            best_score_global = score
            best_pts_global = pts

        # Refine with exact objective using Nelder-Mead or Powell
        if time.time() > deadline - 5:
            continue
            
        try:
            res = minimize(
                objective,
                best_pts_global,
                method='Powell',
                options={'maxiter': 200, 'disp': False}
            )
            if res.fun < best_score_global:
                best_score_global = res.fun
                best_pts_global = res.x.copy()
        except Exception:
            pass
        
        try:
            res = minimize(
                objective,
                best_pts_global,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            if res.fun < best_score_global:
                best_score_global = res.fun
                best_pts_global = res.x.copy()
        except Exception:
            pass

    if best_pts_global is None:
        # Fallback
        best_pts_global = hexagonal_init().flatten()

    return best_pts_global.reshape(N, D)
