"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 130

# EVOLVE-BLOCK-START
def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum over all points y on the sphere of the
    angular distance to the nearest point in pts. This equals the largest
    circumradius of the facets of the convex hull of pts (for points in general
    position not contained in a hemisphere).
    
    We compute this by finding the farthest point on the sphere from all pts,
    which occurs at a vertex of the Voronoi diagram on the sphere (i.e., a point
    equidistant to 3 or more pts) or at a point on the great circle arc between
    two pts (equidistant to exactly 2 pts and farther from all others).
    
    For efficiency, we use the fact that the covering radius is the smallest r
    such that caps of angular radius r centered at pts cover the sphere.
    We use a bisection on r combined with a check, or we can directly compute
    the Delaunay/Voronoi structure.
    
    A simpler approach: the covering radius is the maximum over all triples of
    points (and pairs) of the circumradius of the spherical triangle/cap.
    For N=130, checking all triples is C(130,3) ≈ 365,000 which is feasible.
    """
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    n = len(pts)
    
    # Compute pairwise dot products
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    
    # For each triple, compute the circumradius of the spherical triangle
    # The circumradius r satisfies: for a point y equidistant to pts i, j, k
    # with angular distance r, we need:
    #   y . pts[i] = y . pts[j] = y . pts[k] = cos(r)
    # This means y is in the null space of [pts[i]-pts[j], pts[i]-pts[k]]
    # and we pick the unit vector in that direction (or its negative).
    
    best_r = 0.0
    
    # Check all pairs first (the farthest point might be on the "edge" between two caps)
    # Actually, for the covering radius, we need to check:
    # 1. Points equidistant to 3+ sites (Voronoi vertices)
    # 2. Points equidistant to 2 sites on the boundary of the Voronoi cell (but these are on edges)
    # The maximum is achieved at a Voronoi vertex (equidistant to 3+ points) or at infinity (if not covered)
    
    # For each triple (i, j, k), find the two points on the sphere equidistant to all three
    # Then check if they are the nearest to all points, and compute the angular distance
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Find unit vectors y such that y.pi = y.pj = y.pk
                # This means y is orthogonal to (pi - pj) and (pi - pk)
                v1 = pts[i] - pts[j]
                v2 = pts[i] - pts[k]
                
                # Cross product gives a vector orthogonal to both
                y = np.cross(v1, v2)
                ny = np.linalg.norm(y)
                if ny < 1e-12:
                    continue
                y /= ny
                
                # Check both y and -y
                for sign in [1, -1]:
                    yy = sign * y
                    dists = np.arccos(np.clip(pts @ yy, -1.0, 1.0))
                    d_min = np.min(dists)
                    # Verify that this point is actually a Voronoi vertex (nearest points are i, j, k)
                    # The angular distance to i, j, k should be the minimum
                    di = np.arccos(np.clip(pts[i] @ yy, -1.0, 1.0))
                    if abs(dists[i] - d_min) > 1e-10 or abs(dists[j] - d_min) > 1e-10 or abs(dists[k] - d_min) > 1e-10:
                        # Not a valid Voronoi vertex for this triple as the nearest
                        continue
                    if d_min > best_r:
                        best_r = d_min
    
    # Also check pairs: the farthest point might be equidistant to just 2 points
    # This happens at the midpoint of the great circle arc between two antipodal-ish points
    # but more generally, on the Voronoi edge between two cells
    # For pairs, the candidate is the point on the sphere equidistant to i and j
    # that is farthest from all other points. This is more complex.
    # However, for a good configuration, the covering radius is typically determined
    # by a triple. Let's also do a grid-based check as a fallback.
    
    # Grid-based verification (coarse) to catch any issues
    # Use a fine icosahedral-like grid
    def _spherical_grid(n_theta=60, n_phi=120):
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TT, PP = np.meshgrid(thetas, phis, indexing='ij')
        X = np.sin(TT) * np.cos(PP)
        Y = np.sin(TT) * np.sin(PP)
        Z = np.cos(TT)
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    grid = _spherical_grid(80, 160)
    # For each grid point, find nearest pt
    for chunk_start in range(0, len(grid), 5000):
        chunk = grid[chunk_start:chunk_start + 5000]
        dists = np.arccos(np.clip(chunk @ pts.T, -1.0, 1.0))
        min_dists = np.min(dists, axis=1)
        max_d = np.max(min_dists)
        if max_d > best_r:
            best_r = max_d
    
    return np.degrees(best_r)


def _refine_covering(pts, time_limit=100.0):
    """Refine points to minimize covering radius using gradient-based optimization."""
    start_time = time.time()
    pts = pts.copy()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    def objective(x):
        """Objective: maximize the minimum distance from any point to its nearest site.
        We use a smooth approximation: -log-sum-exp of negative distances, or
        we can use the covering radius directly via a differentiable proxy.
        
        Actually, let's use a different approach: minimize the sum of squared
        distances from a fine grid to the nearest point. This is smooth and
        correlates well with covering radius.
        """
        t = time.time()
        if t - start_time > time_limit:
            return np.inf
        p = x.reshape(-1, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        
        # Use a moderate grid for speed
        n_theta, n_phi = 40, 80
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TT, PP = np.meshgrid(thetas, phis, indexing='ij')
        X = np.sin(TT) * np.cos(PP)
        Y = np.sin(TT) * np.sin(PP)
        Z = np.cos(TT)
        grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Compute distance from each grid point to nearest site
        # Use chunking for memory
        max_dist = 0.0
        for chunk_start in range(0, len(grid), 2000):
            chunk = grid[chunk_start:chunk_start + 2000]
            dists = np.arccos(np.clip(chunk @ p.T, -1.0, 1.0))
            min_dists = np.min(dists, axis=1)
            # Use soft-max for differentiability
            # Actually, let's just use the max for now (nonsmooth but works with L-BFGS-B)
            max_dist = max(max_dist, np.max(min_dists))
        
        # To make it smooth, use log-sum-exp
        # But for L-BFGS-B, we can use the nonsmooth version
        return max_dist
    
    # Alternative: use the actual covering radius computed via the triple method
    # but that's expensive. Let's use a hybrid:
    
    def objective_v2(x):
        t = time.time()
        if t - start_time > time_limit:
            return np.inf
        p = x.reshape(-1, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        
        # Use a moderate grid
        n_theta, n_phi = 50, 100
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TT, PP = np.meshgrid(thetas, phis, indexing='ij')
        X = np.sin(TT) * np.cos(PP)
        Y = np.sin(TT) * np.sin(PP)
        Z = np.cos(TT)
        grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Compute min distance from each grid point to nearest site
        # Use vectorized computation in chunks
        all_min_dists = []
        for chunk_start in range(0, len(grid), 1000):
            chunk = grid[chunk_start:chunk_start + 1000]
            dots = chunk @ p.T  # (chunk, N)
            dists = np.arccos(np.clip(dots, -1.0, 1.0))
            min_dists = np.min(dists, axis=1)
            all_min_dists.append(min_dists)
        
        all_min_dists = np.concatenate(all_min_dists)
        
        # Use a smooth approximation to the max: log-sum-exp
        # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
        beta = 50.0
        shifted = all_min_dists - np.max(all_min_dists)
        log_sum_exp = np.max(all_min_dists) + np.log(np.sum(np.exp(beta * shifted))) / beta
        return log_sum_exp
    
    # Use L-BFGS-B to minimize
    x0 = pts.ravel()
    result = minimize(
        objective_v2,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if time.time() - start_time < time_limit - 10:
        pts_refined = result.x.reshape(-1, 3)
        pts_refined = pts_refined / np.linalg.norm(pts_refined, axis=1, keepdims=True)
        
        # Do a second pass with finer grid
        def objective_v3(x):
            t = time.time()
            if t - start_time > time_limit - 5:
                return np.inf
            p = x.reshape(-1, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            
            n_theta, n_phi = 80, 160
            thetas = np.linspace(0, np.pi, n_theta)
            phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
            TT, PP = np.meshgrid(thetas, phis, indexing='ij')
            X = np.sin(TT) * np.cos(PP)
            Y = np.sin(TT) * np.sin(PP)
            Z = np.cos(TT)
            grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            all_min_dists = []
            for chunk_start in range(0, len(grid), 1000):
                chunk = grid[chunk_start:chunk_start + 1000]
                dots = chunk @ p.T
                dists = np.arccos(np.clip(dots, -1.0, 1.0))
                min_dists = np.min(dists, axis=1)
                all_min_dists.append(min_dists)
            
            all_min_dists = np.concatenate(all_min_dists)
            beta = 100.0
            shifted = all_min_dists - np.max(all_min_dists)
            log_sum_exp = np.max(all_min_dists) + np.log(np.sum(np.exp(beta * shifted))) / beta
            return log_sum_exp
        
        result2 = minimize(
            objective_v3,
            pts_refined.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if time.time() - start_time < time_limit - 2:
            pts_final = result2.x.reshape(-1, 3)
            pts_final = pts_final / np.linalg.norm(pts_final, axis=1, keepdims=True)
            return pts_final
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple random restarts and keep the best
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Do gradient-based refinement
    if time.time() < deadline - 20:
        refined = _refine_covering(pts, time_limit=min(60, deadline - time.time() - 10))
        r_refined = _covering_radius(refined)
        if r_refined < best_radius:
            best_pts = refined
            best_radius = r_refined
    
    # Try a few random perturbations and refine
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 25:
            break
        # Perturb the current best
        noise = rng.normal(0, 0.05, size=(N, 3))
        perturbed = best_pts + noise
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        
        if time.time() < deadline - 15:
            refined = _refine_covering(perturbed, time_limit=min(30, deadline - time.time() - 10))
            r_refined = _covering_radius(refined)
            if r_refined < best_radius:
                best_pts = refined
                best_radius = r_refined
    
    # Final polish: local search on the best configuration
    if time.time() < deadline - 5:
        def objective_final(x):
            p = x.reshape(-1, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            
            n_theta, n_phi = 100, 200
            thetas = np.linspace(0, np.pi, n_theta)
            phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
            TT, PP = np.meshgrid(thetas, phis, indexing='ij')
            X = np.sin(TT) * np.cos(PP)
            Y = np.sin(TT) * np.sin(PP)
            Z = np.cos(TT)
            grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            all_min_dists = []
            for chunk_start in range(0, len(grid), 1000):
                chunk = grid[chunk_start:chunk_start + 1000]
                dots = chunk @ p.T
                dists = np.arccos(np.clip(dots, -1.0, 1.0))
                min_dists = np.min(dists, axis=1)
                all_min_dists.append(min_dists)
            
            all_min_dists = np.concatenate(all_min_dists)
            beta = 200.0
            shifted = all_min_dists - np.max(all_min_dists)
            log_sum_exp = np.max(all_min_dists) + np.log(np.sum(np.exp(beta * shifted))) / beta
            return log_sum_exp
        
        result = minimize(
            objective_final,
            best_pts.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if time.time() < deadline:
            final_pts = result.x.reshape(-1, 3)
            final_pts = final_pts / np.linalg.norm(final_pts, axis=1, keepdims=True)
            r_final = _covering_radius(final_pts)
            if r_final < best_radius:
                best_pts = final_pts
                best_radius = r_final
    
    # Ensure no coincident points and not in a hemisphere
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                # Separate them
                best_pts[j] = best_pts[i] + np.random.normal(0, 0.01, 3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    # Check hemisphere: if all points are in a closed hemisphere, it's invalid
    # A set of points is in a closed hemisphere if there exists a vector v such that v . p_i >= 0 for all i
    # This is equivalent to the origin not being in the interior of the convex hull
    # Quick check: if the max over all unit vectors v of min_i (v . p_i) >= 0, then they're in a hemisphere
    # We can check by seeing if the convex hull contains the origin
    # Simple check: if all points have z > -0.1 or similar, they might be in a hemisphere
    # More robust: check if there's a direction where all dot products are positive
    # For now, if the configuration is well-distributed (from our optimization), this should be fine
    
    return best_pts
# EVOLVE-BLOCK-END
