import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 625.0 # Safety margin from 630s
    N = 16

    def get_true_cost(params):
        """
        Calculate (d_max / d_min)^2. Lower is better.
        Returns inf if d_min is 0.
        """
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        d_max_sq = np.max(dist_sq)
        d_min_sq = np.min(dist_sq)
        
        if d_min_sq < 1e-12:
            return 1e12
        if d_max_sq < 1e-12:
            return 1e12
            
        return d_max_sq / d_min_sq

    def get_smooth_cost(params, tau=0.5):
        """
        Calculate smooth approximation of (d_max / d_min)^2 using log-sum-exp.
        """
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Soft max and min of squared distances
        # To avoid overflow, we can shift, but for tau=0.5 and normalized points, it should be safe.
        # We normalize points inside the wrapper to keep scale in check.
        
        # Soft max
        m = np.max(dist_sq)
        soft_max = m + tau * np.log(np.sum(np.exp((dist_sq - m) / tau)))
        
        # Soft min
        m_min = np.min(dist_sq)
        soft_min = m_min - tau * np.log(np.sum(np.exp((m_min - dist_sq) / tau)))
        
        if soft_min < 1e-12:
            return 1e12
            
        return soft_max / soft_min

    def normalize(params):
        """
        Normalize points to center at origin and have max distance 1 (diameter).
        This helps with scale invariance.
        """
        pts = params.reshape(N, 2)
        # Center
        pts = pts - np.mean(pts, axis=0)
        # Scale: find max distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        d_max_sq = np.max(dist_sq)
        if d_max_sq > 1e-12:
            pts = pts / np.sqrt(d_max_sq)
        return pts

    def get_cost_normalized(params, tau=None):
        """
        Wrapper that normalizes points before calculating cost.
        """
        pts = normalize(params)
        if tau is None:
            # Calculate true cost on normalized points
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            d_max_sq = np.max(dist_sq)
            d_min_sq = np.min(dist_sq)
            if d_min_sq < 1e-12:
                return 1e12
            return d_max_sq / d_min_sq
        else:
            # Calculate smooth cost on normalized points
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            
            m = np.max(dist_sq)
            soft_max = m + tau * np.log(np.sum(np.exp((dist_sq - m) / tau)))
            
            m_min = np.min(dist_sq)
            soft_min = m_min - tau * np.log(np.sum(np.exp((m_min - dist_sq) / tau)))
            
            if soft_min < 1e-12:
                return 1e12
            return soft_max / soft_min

    best_result = None
    best_true_cost = np.inf

    # Initial guess: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_result = init_pts.flatten()
    best_true_cost = get_cost_normalized(best_result)

    # Phase 1: Differential Evolution with smooth cost
    # We use normalized cost to keep the search in a bounded region effectively
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    try:
        de_res = differential_evolution(
            lambda x: get_cost_normalized(x, tau=0.5), 
            bounds, 
            maxiter=500, 
            popsize=30, 
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        
        if de_res.fun < best_true_cost:
             # Check true cost
             candidate = de_res.x
             true_c = get_cost_normalized(candidate)
             if true_c < best_true_cost:
                 best_true_cost = true_c
                 best_result = candidate.copy()
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)

    # Phase 2: Local Refinement with Powell
    # Use a very small tau for smooth cost to guide gradient, then true cost
    time_elapsed = time.time() - start_time
    if time_elapsed < TIME_LIMIT - 60:
        try:
            # Polish with smooth cost (tau=0.01)
            pw_res = minimize(
                lambda x: get_cost_normalized(x, tau=0.01), 
                best_result, 
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if pw_res.fun < best_true_cost:
                candidate = pw_res.x
                true_c = get_cost_normalized(candidate)
                if true_c < best_true_cost:
                    best_true_cost = true_c
                    best_result = candidate.copy()
        except Exception as e:
            print(f"Powell smooth failed: {e}", file=sys.stderr)

    # Phase 3: Local Refinement with True Cost
    time_elapsed = time.time() - start_time
    if time_elapsed < TIME_LIMIT - 30:
        try:
            pw_res2 = minimize(
                lambda x: get_cost_normalized(x), 
                best_result, 
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if pw_res2.fun < best_true_cost:
                best_true_cost = pw_res2.fun
                best_result = pw_res2.x.copy()
        except Exception as e:
            print(f"Powell true failed: {e}", file=sys.stderr)

    # Phase 4: Multi-start Local Optimization
    # If we have time, try restarting from slightly perturbed best points
    time_elapsed = time.time() - start_time
    if time_elapsed < TIME_LIMIT - 10 and best_result is not None:
        rng = np.random.default_rng(123)
        for i in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Perturb
            noise = rng.normal(0, 0.05, best_result.shape)
            perturbed = best_result + noise
            
            try:
                local_res = minimize(
                    lambda x: get_cost_normalized(x), 
                    perturbed, 
                    method='Powell',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if local_res.fun < best_true_cost:
                    best_true_cost = local_res.fun
                    best_result = local_res.x.copy()
            except Exception:
                pass

    # Final normalization and return
    if best_result is None or not np.all(np.isfinite(best_result)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        
    # The evaluator expects shape (16, 2)
    final_pts = normalize(best_result).reshape(16, 2)
    
    return final_pts
