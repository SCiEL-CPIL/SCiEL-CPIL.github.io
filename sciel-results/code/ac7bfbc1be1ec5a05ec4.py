"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Multi-start Nelder-Mead with smooth soft-min objective.
    """
    
    def get_radius(x):
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        return min(np.sqrt(np.min(d2)) / 2, 1 - np.max(d_origin))
    
    def objective(x, sharpness=50.0):
        """
        Negative of a smooth approximation to the minimum radius.
        Uses soft-min via log-sum-exp for differentiability.
        """
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        d_pairs = np.sqrt(d2)
        d_boundary = 1.0 - d_origin
        
        # All distances we care about: pairwise/2 and boundary
        half_pairs = d_pairs / 2.0
        all_dists = np.concatenate([half_pairs.flatten(), d_boundary])
        all_dists = np.maximum(all_dists, 1e-12)
        
        # Soft min: -log(sum(exp(-alpha * d))) / alpha
        # This is a smooth approximation to min(dists)
        scaled = -sharpness * all_dists
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / sharpness
        
        return -soft_min
    
    def init_random_ring(seed):
        """Initialize with points on concentric rings with random jitter."""
        rng = np.random.default_rng(seed)
        pts = []
        # Ring 0: center
        pts.append([0.0, 0.0])
        # Ring 1: 6 points
        n1, r1 = 6, 0.22
        angles1 = 2*np.pi*np.arange(n1)/n1 + rng.uniform(0, 2*np.pi)
        pts.extend([[r1*np.cos(a), r1*np.sin(a)] for a in angles1])
        # Ring 2: 12 points
        n2, r2 = 12, 0.44
        angles2 = 2*np.pi*np.arange(n2)/n2 + rng.uniform(0, 2*np.pi)
        pts.extend([[r2*np.cos(a), r2*np.sin(a)] for a in angles2])
        # Ring 3: 18 points
        n3, r3 = 18, 0.66
        angles3 = 2*np.pi*np.arange(n3)/n3 + rng.uniform(0, 2*np.pi)
        pts.extend([[r3*np.cos(a), r3*np.sin(a)] for a in angles3])
        # Ring 4: 13 points
        n4, r4 = 13, 0.85
        angles4 = 2*np.pi*np.arange(n4)/n4 + rng.uniform(0, 2*np.pi)
        pts.extend([[r4*np.cos(a), r4*np.sin(a)] for a in angles4])
        
        pts = np.array(pts[:N])
        # Add small jitter
        pts += rng.normal(0, 0.02, (N, 2))
        # Ensure inside unit circle
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.98
        if np.any(too_far):
            scale = 0.98 / dists[too_far]
            pts[too_far] *= scale[:, None]
        return pts.flatten()
    
    def init_hexagonal(seed):
        """Initialize on a hexagonal grid pattern."""
        rng = np.random.default_rng(seed)
        pts = []
        # Hexagonal packing pattern
        spacing = 0.19
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        for row in range(-5, 6):
            for col in range(-5, 6):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                if x*x + y*y < 0.85:
                    pts.append([x, y])
        # Shuffle and take N
        rng.shuffle(pts)
        pts = np.array(pts[:N])
        pts += rng.normal(0, 0.01, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.95
        if np.any(too_far):
            scale = 0.95 / dists[too_far]
            pts[too_far] *= scale[:, None]
        return pts.flatten()
    
    def init_random_polar(seed):
        """Random initialization in polar coordinates with bias."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2*np.pi, N)
        # Use a distribution that favors mid-range radii
        r = rng.beta(2, 2, N) * 0.95
        pts = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
        pts += rng.normal(0, 0.015, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            scale = 0.97 / dists[too_far]
            pts[too_far] *= scale[:, None]
        return pts.flatten()
    
    def init_random_uniform(seed):
        """Uniform random initialization in the disk."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2*np.pi, N)
        r = np.sqrt(rng.uniform(0, 1, N)) * 0.95
        pts = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
        pts += rng.normal(0, 0.01, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            scale = 0.97 / dists[too_far]
            pts[too_far] *= scale[:, None]
        return pts.flatten()
    
    def push_apart_refine(x, iterations=200, step=0.005):
        """Local refinement: push overlapping circles apart."""
        pts = x.reshape(N, 2).copy()
        for it in range(iterations):
            # Compute pairwise distances
            diff = pts[:, None, :] - pts[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)
            
            # Find minimum distance
            min_d = np.min(d)
            if min_d < 2 * 0.125:  # If circles are too close
                # Find the pair with minimum distance
                idx = np.unravel_index(np.argmin(d), d.shape)
                i, j = idx
                if i == j:
                    continue
                vec = pts[i] - pts[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    # Push them apart slightly
                    push = vec / norm * step * 0.5
                    pts[i] += push
                    pts[j] -= push
            
            # Also push points toward center if too close to boundary
            d_origin = np.linalg.norm(pts, axis=1)
            for k in range(N):
                if d_origin[k] > 1 - 0.125:
                    pts[k] *= (1 - 0.125) / d_origin[k]
            
            # Ensure all points stay inside
            d_origin = np.linalg.norm(pts, axis=1)
            too_far = d_origin > 0.99
            if np.any(too_far):
                scale = 0.99 / d_origin[too_far]
                pts[too_far] *= scale[:, None]
        
        return pts.flatten()
    
    # Generate diverse initializations
    inits = [
        init_random_ring(1),
        init_random_ring(2),
        init_random_ring(3),
        init_hexagonal(10),
        init_hexagonal(20),
        init_random_polar(30),
        init_random_polar(40),
        init_random_uniform(50),
        init_random_uniform(60),
    ]
    
    best_x, best_r = None, -1
    
    # Phase 1: Coarse optimization with moderate sharpness
    for i, x0 in enumerate(inits):
        try:
            res = minimize(lambda x: objective(x, sharpness=30.0), x0, 
                          method='Nelder-Mead',
                          options={'maxiter': 2000, 'maxfev': 3000,
                                   'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
            r = get_radius(res.x)
            if r > best_r:
                best_r, best_x = r, res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        return init_random_ring(1).reshape(N, 2)
    
    # Phase 2: Medium optimization with higher sharpness
    try:
        res2 = minimize(lambda x: objective(x, sharpness=50.0), best_x,
                       method='Nelder-Mead',
                       options={'maxiter': 3000, 'maxfev': 5000,
                                'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True})
        r2 = get_radius(res2.x)
        if r2 > best_r:
            best_x, best_r = res2.x.copy(), r2
    except Exception:
        pass
    
    # Phase 3: Fine optimization with very high sharpness
    try:
        res3 = minimize(lambda x: objective(x, sharpness=100.0), best_x,
                       method='Nelder-Mead',
                       options={'maxiter': 3000, 'maxfev': 5000,
                                'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
        r3 = get_radius(res3.x)
        if r3 > best_r:
            best_x, best_r = res3.x.copy(), r3
    except Exception:
        pass
    
    # Phase 4: Very fine optimization
    try:
        res4 = minimize(lambda x: objective(x, sharpness=200.0), best_x,
                       method='Nelder-Mead',
                       options={'maxiter': 2000, 'maxfev': 3000,
                                'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
        r4 = get_radius(res4.x)
        if r4 > best_r:
            best_x, best_r = res4.x.copy(), r4
    except Exception:
        pass
    
    # Phase 5: Local push-apart refinement
    try:
        refined = push_apart_refine(best_x, iterations=300, step=0.003)
        r5 = get_radius(refined)
        if r5 > best_r:
            best_x, best_r = refined, r5
    except Exception:
        pass
    
    # Phase 6: Final fine-tune after refinement
    try:
        res6 = minimize(lambda x: objective(x, sharpness=200.0), best_x,
                       method='Nelder-Mead',
                       options={'maxiter': 1500, 'maxfev': 2000,
                                'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
        r6 = get_radius(res6.x)
        if r6 > best_r:
            best_x, best_r = res6.x.copy(), r6
    except Exception:
        pass
    
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
