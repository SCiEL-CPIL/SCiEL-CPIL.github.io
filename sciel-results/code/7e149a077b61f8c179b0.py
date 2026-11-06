import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    n = 8
    
    def compute_score(pts):
        """Compute the score Delta/n^n where Delta is product of ordered pair distances
        after scaling to diameter 2."""
        pts = np.array(pts, dtype=float)
        if pts.shape != (n, 2):
            return -np.inf
        
        # Check distinctness
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        
        if np.any(d < 1e-9):
            return -np.inf
        
        diam = np.max(d)
        if diam < 1e-9:
            return -np.inf
        
        # Scale to diameter 2
        pts_scaled = pts * (2.0 / diam)
        
        # Recalculate distances after scaling
        diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists_s = np.sqrt(np.sum(diff_s**2, axis=2))
        d_s = dists_s[mask]
        
        # Product of all ordered pair distances (i != j)
        # log(Delta) = sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
        log_delta = np.sum(np.log(d_s))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        
        return score
    
    def angular_gaps_to_points(gaps):
        """Convert 7 angular gaps (8th is determined by sum=2*pi) to 8 points on unit circle.
        gaps: array of 7 positive values summing to < 2*pi
        Returns 8 points [x, y] on the unit circle."""
        # 8th gap
        gap8 = 2 * np.pi - np.sum(gaps)
        all_gaps = np.append(gaps, gap8)
        
        # Check all gaps are positive
        if np.any(all_gaps <= 1e-10):
            return None
        
        # Cumulative angles
        angles = np.zeros(n)
        angles[0] = 0
        for i in range(1, n):
            angles[i] = angles[i-1] + all_gaps[i-1]
        
        # Points on unit circle
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts
    
    def angular_negative_score(gaps):
        """Negative log-score for optimization. gaps is 7 values."""
        pts = angular_gaps_to_points(gaps)
        if pts is None:
            return 1e10
        
        score = compute_score(pts)
        if score <= 0:
            return 1e10
        return -np.log(score)
    
    best_score = -np.inf
    best_pts = None
    
    # Strategy A: Optimize angular gaps (7 DOF)
    # Regular octagon: all gaps = 2*pi/8 = pi/4
    regular_gaps = np.array([np.pi/4] * 7)
    
    # Grid search over angular gap variations
    if time.time() < deadline - 20:
        # Perturb the regular octagon gaps in structured ways
        base = np.pi / 4
        
        # Try various patterns of gap variation
        gap_patterns = []
        
        # Alternating large/small gaps
        for delta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for phase in [0, 1]:
                gaps = base + delta * np.array([1 if (i + phase) % 2 == 0 else -1 for i in range(7)])
                gap_patterns.append(gaps)
        
        # Two large, rest small
        for i in range(7):
            for j in range(i+1, 7):
                for delta in [0.1, 0.15, 0.2, 0.25, 0.3]:
                    gaps = base.copy()
                    gaps[i] += delta
                    gaps[j] += delta
                    # Reduce others to keep sum < 2*pi
                    remaining = 2*np.pi - gaps[i] - gaps[j]
                    others = list(range(7))
                    others.remove(i)
                    others.remove(j)
                    for k in others:
                        gaps[k] = remaining / 5.0
                    gap_patterns.append(gaps)
        
        # One very large gap
        for i in range(7):
            for delta in [0.2, 0.3, 0.4, 0.5, 0.6]:
                gaps = base.copy()
                gaps[i] += delta
                remaining = 2*np.pi - gaps[i]
                others = list(range(7))
                others.remove(i)
                for k in others:
                    gaps[k] = remaining / 6.0
                gap_patterns.append(gaps)
        
        # Random perturbations
        np.random.seed(42)
        for _ in range(50):
            gaps = base + np.random.uniform(-0.3, 0.3, size=7)
            if np.all(gaps > 0.05) and np.sum(gaps) < 2*np.pi - 0.05:
                gap_patterns.append(gaps)
        
        # Evaluate all patterns
        for gaps in gap_patterns:
            if time.time() > deadline - 10:
                break
            neg = angular_negative_score(gaps)
            if neg < 1e9:
                score = np.exp(-neg)
                if score > best_score:
                    best_score = score
                    best_pts = angular_gaps_to_points(gaps)
        
        # Local optimization from best patterns
        if best_pts is not None:
            # Find which gaps gave the best score
            for gaps in gap_patterns:
                if time.time() > deadline - 5:
                    break
                neg = angular_negative_score(gaps)
                if neg < 1e9:
                    score = np.exp(-neg)
                    if score >= best_score * 0.99:
                        try:
                            res = minimize(angular_negative_score, gaps, method='Nelder-Mead',
                                         options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
                            if res.fun < 1e9:
                                score_res = np.exp(-res.fun)
                                if score_res > best_score:
                                    best_score = score_res
                                    best_pts = angular_gaps_to_points(res.x)
                        except Exception:
                            pass
                    break  # Only refine the best one
    
    # Strategy B: Constrained optimization with explicit diameter constraint
    if time.time() < deadline - 10:
        if best_pts is None:
            # Start from regular octagon
            best_pts = angular_gaps_to_points(regular_gaps)
            best_score = compute_score(best_pts)
        
        def constrained_objective(gaps):
            pts = angular_gaps_to_points(gaps)
            if pts is None:
                return 1e10
            
            # Compute distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            mask = ~np.eye(n, dtype=bool)
            d_sq = dists_sq[mask]
            
            if np.any(d_sq < 1e-12):
                return 1e10
            
            # Diameter constraint: max distance <= 2 (we're on unit circle, so max dist = 2)
            # Actually on unit circle, max possible distance is 2 (diameter)
            # So we just need to maximize log product
            log_prod = 0.5 * np.sum(np.log(d_sq))
            return -log_prod
        
        # Constraints: all gaps positive and sum to 2*pi
        # Use equality constraint for sum and inequality for positivity
        def eq_constraint(gaps):
            return 2*np.pi - np.sum(gaps)
        
        def ineq_constraints(gaps):
            return gaps  # each gap > 0
        
        constraints = [{'type': 'eq', 'fun': eq_constraint}]
        for i in range(7):
            constraints.append({'type': 'ineq', 'fun': lambda g, i=i: g[i] - 1e-8})
        
        # Try from several starting points
        start_points = [
            regular_gaps,
            regular_gaps + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.0]),
            regular_gaps + np.array([0.2, 0.2, -0.1, -0.1, -0.1, -0.1, 0.0]),
            regular_gaps + np.array([0.15, 0.15, 0.15, -0.15, -0.15, -0.15, 0.0]),
        ]
        
        for x0 in start_points:
            if time.time() > deadline - 3:
                break
            
            try:
                res = minimize(constrained_objective, x0, method='SLSQP',
                              constraints=constraints,
                              options={'maxiter': 5000, 'ftol': 1e-12})
                
                if res.fun < 1e9:
                    pts_res = angular_gaps_to_points(res.x)
                    if pts_res is not None:
                        score_res = compute_score(pts_res)
                        if score_res > best_score:
                            best_score = score_res
                            best_pts = pts_res
            except Exception:
                pass
    
    # Strategy C: Differential evolution on 7 angular gaps
    if time.time() < deadline - 15:
        try:
            bounds = [(0.1, 1.5)] * 7  # Each gap between 0.1 and 1.5 radians
            
            def de_objective(gaps):
                # Check sum constraint
                if np.sum(gaps) > 2*np.pi - 0.01:
                    return 1e10 + (np.sum(gaps) - 2*np.pi) * 100
                
                pts = angular_gaps_to_points(gaps)
                if pts is None:
                    return 1e10
                
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                mask = ~np.eye(n, dtype=bool)
                d_sq = dists_sq[mask]
                
                if np.any(d_sq < 1e-12):
                    return 1e10
                
                log_prod = 0.5 * np.sum(np.log(d_sq))
                return -log_prod
            
            de_res = differential_evolution(de_objective, bounds, maxiter=100,
                                            popsize=15, tol=1e-10,
                                            seed=42, polish=False,
                                            updating='deferred')
            
            if de_res.fun < 1e9:
                pts_de = angular_gaps_to_points(de_res.x)
                if pts_de is not None:
                    score_de = compute_score(pts_de)
                    if score_de > best_score:
                        best_score = score_de
                        best_pts = pts_de
        except Exception:
            pass
    
    # Strategy D: Hybrid - use best angular result, then try full 2D optimization
    if time.time() < deadline - 10 and best_pts is not None:
        try:
            def full_objective(x):
                pts = x.reshape(n, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                mask = ~np.eye(n, dtype=bool)
                d_sq = dists_sq[mask]
                
                if np.any(d_sq < 1e-12):
                    return 1e10
                
                # Penalize diameter > 2
                max_d_sq = np.max(d_sq)
                if max_d_sq > 4.0:  # diameter > 2
                    penalty = (max_d_sq - 4.0) * 100
                else:
                    penalty = 0
                
                log_prod = 0.5 * np.sum(np.log(d_sq))
                return -log_prod + penalty
            
            x0 = best_pts.flatten()
            res_full = minimize(full_objective, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_full.fun < 1e9:
                pts_full = res_full.x.reshape(n, 2)
                score_full = compute_score(pts_full)
                if score_full > best_score:
                    best_score = score_full
                    best_pts = pts_full
        except Exception:
            pass
    
    # Final fallback
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure coordinates are within bounds
    best_pts = np.clip(best_pts, -1e6, 1e6)
    
    return best_pts.tolist()
