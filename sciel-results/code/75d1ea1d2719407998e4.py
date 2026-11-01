"""Optimized sphere covering with 50 points using Lloyd iteration + soft-max polish."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s limit

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _facet_radius(simplex, pts):
    """Compute the circumradius (angular distance from center to farthest point) of a facet."""
    facet_pts = pts[simplex]
    if len(facet_pts) == 3:
        a, b, c = facet_pts
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return 0.0, np.zeros(3)
        n = n / n_norm
        d = np.dot(n, a)
        if d < 0:
            d = -d
            n = -n
        radius = np.arccos(np.clip(d, -1.0, 1.0))
        return radius, n
    elif len(facet_pts) == 4:
        a, b, c, d4 = facet_pts
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return 0.0, np.zeros(3)
        n = n / n_norm
        d = np.dot(n, a)
        if d < 0:
            d = -d
            n = -n
        radius = np.arccos(np.clip(d, -1.0, 1.0))
        return radius, n
    return 0.0, np.zeros(3)

def _covering_radius(pts):
    """Compute covering radius in radians using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        for simplex in hull.simplices:
            r, _ = _facet_radius(simplex, pts)
            max_radius = max(max_radius, r)
        return max_radius
    except Exception:
        # Fallback: brute force sampling
        theta = np.linspace(0, 2*np.pi, 360)
        phi = np.linspace(0, np.pi, 180)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        dots = np.sum(samples[:, np.newaxis, :] * pts[np.newaxis, :, :], axis=2)
        min_dots = np.max(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.max(max_angle)

def _all_facet_radii(pts):
    """Compute radii and normals for all hull facets."""
    try:
        hull = ConvexHull(pts)
        radii = []
        normals = []
        simplices = []
        for simplex in hull.simplices:
            r, n = _facet_radius(simplex, pts)
            radii.append(r)
            normals.append(n)
            simplices.append(simplex)
        return np.array(radii), np.array(normals), simplices
    except Exception:
        return np.array([]), np.array([]), []

def _soft_max_energy(pts, beta=100.0):
    """
    Soft-max approximation of the covering radius using log-sum-exp over facet radii.
    Returns (energy, gradient_wrt_pts).
    """
    radii, normals, simplices = _all_facet_radii(pts)
    if len(radii) == 0:
        return 0.0, np.zeros((N, 3))
    
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros((N, 3))
    
    # Log-sum-exp: soft_max = max_r + log(sum(exp(beta*(r_i - max_r)))) / beta
    exps = np.exp(beta * (radii - max_r))
    soft_max = max_r + np.log(np.sum(exps)) / beta
    
    # Gradient: d(soft_max)/dr_i = exp(beta*(r_i - max_r)) / sum(exp(...))
    weights = exps / np.sum(exps)  # (F,)
    
    return soft_max, None

def _lloyd_iteration(pts, step_size=0.1, time_budget=5.0):
    """
    Lloyd iteration: for each point, find the farthest point on the sphere within its 
    Voronoi cell (approximated by the farthest facet circumcenter), and move toward it.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    for iteration in range(50):
        if time.time() - start_time > time_budget:
            break
        
        radii, normals, simplices = _all_facet_radii(pts)
        if len(radii) == 0:
            break
        
        # For each point, find the worst facet it belongs to
        # and compute the circumcenter direction
        new_pts = pts.copy()
        
        # Build mapping: point index -> list of (facet_index, radius)
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(N):
            if i not in point_to_facets or not point_to_facets[i]:
                continue
            
            # Find the worst facet for this point
            best_fi = None
            best_r = 0.0
            for fi in point_to_facets[i]:
                if radii[fi] > best_r:
                    best_r = radii[fi]
                    best_fi = fi
            
            if best_fi is None or best_r < 1e-6:
                continue
            
            # The circumcenter direction is the normal of the worst facet
            # (pointing outward from the origin toward the farthest point in the cell)
            target_dir = normals[best_fi]
            
            # Move point i toward target_dir on the sphere
            p = pts[i]
            # Spherical interpolation
            dot_val = np.dot(p, target_dir)
            dot_val = np.clip(dot_val, -0.9999, 0.9999)
            angle_between = np.arccos(dot_val)
            
            if angle_between < 1e-10:
                continue
            
            # Rotate p toward target_dir by step_size fraction of the angle
            move_angle = min(step_size, angle_between * 0.5)
            sin_total = np.sin(angle_between)
            if sin_total < 1e-10:
                continue
            
            new_p = (np.sin(angle_between - move_angle) / sin_total) * p + \
                    (np.sin(move_angle) / sin_total) * target_dir
            new_pts[i] = new_p / np.linalg.norm(new_p)
        
        new_pts = _normalize(new_pts)
        pts = new_pts
    
    return pts

def _spectral_optimize(pts0, time_budget=20.0):
    """Optimize using spectral energy (smooth surrogate)."""
    start_time = time.time()
    pts = _normalize(pts0.copy())
    
    def _spectral_energy_and_grad_inner(pts_flat):
        pts = _normalize(pts_flat.reshape(N, 3))
        G = pts @ pts.T
        energy = -np.trace(G)
        sum_pj = pts.sum(axis=0)
        dots_with_sum = pts @ sum_pj
        grad = (pts * dots_with_sum[:, np.newaxis] - sum_pj[np.newaxis, :]).ravel()
        return energy, grad
    
    def obj(x):
        if time.time() - start_time > time_budget:
            return 0.0
        return _spectral_energy_and_grad_inner(x)[0]
    
    def jac(x):
        if time.time() - start_time > time_budget:
            return np.zeros_like(x)
        return _spectral_energy_and_grad_inner(x)[1]
    
    try:
        result = minimize(
            obj,
            pts.ravel(),
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 600}
        )
        pts = _normalize(result.x.reshape(N, 3))
    except Exception:
        pass
    
    return pts

def _direct_polish(pts0, time_budget=15.0):
    """
    Direct polish: minimize covering radius by moving all points toward their 
    worst facet circumcenters with adaptive step size.
    """
    start_time = time.time()
    pts = _normalize(pts0.copy())
    current_radius = _covering_radius(pts)
    
    step_size = 0.05
    stall_count = 0
    
    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break
        if current_radius < 1e-6:
            break
        
        radii, normals, simplices = _all_facet_radii(pts)
        if len(radii) == 0:
            break
        
        # For each point, find its worst facet and the circumcenter direction
        new_pts = pts.copy()
        
        # Build point-to-facet mapping
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(N):
            if i not in point_to_facets or not point_to_facets[i]:
                continue
            
            # Find worst facet for this point
            best_fi = None
            best_r = 0.0
            for fi in point_to_facets[i]:
                if radii[fi] > best_r:
                    best_r = radii[fi]
                    best_fi = fi
            
            if best_fi is None or best_r < 1e-6:
                continue
            
            # Target: circumcenter of worst facet (normal direction)
            target_dir = normals[best_fi]
            p = pts[i]
            
            # Spherical interpolation toward target
            dot_val = np.clip(np.dot(p, target_dir), -0.9999, 0.9999)
            angle_between = np.arccos(dot_val)
            
            if angle_between < 1e-10:
                continue
            
            move_angle = min(step_size, angle_between * 0.3)
            sin_total = np.sin(angle_between)
            if sin_total < 1e-10:
                continue
            
            new_p = (np.sin(angle_between - move_angle) / sin_total) * p + \
                    (np.sin(move_angle) / sin_total) * target_dir
            new_pts[i] = new_p / np.linalg.norm(new_p)
        
        new_pts = _normalize(new_pts)
        new_radius = _covering_radius(new_pts)
        
        if new_radius < current_radius - 1e-8:
            pts = new_pts
            current_radius = new_radius
            stall_count = 0
            step_size = min(step_size * 1.02, 0.1)
        else:
            stall_count += 1
            step_size *= 0.7
            if stall_count > 10 or step_size < 1e-5:
                break
    
    return pts, current_radius

def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci spiral points."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _two_phase_optimize(pts0, time_budget=35.0):
    """
    Multi-phase optimization:
    1. Spectral energy minimization (smooth, fast)
    2. Lloyd iteration (global redistribution)
    3. Direct polish (fine-tuning)
    4. Additional Lloyd + polish cycles
    """
    start_time = time.time()
    pts = _normalize(pts0.copy())
    
    # Phase 1: Spectral energy
    phase1_time = min(15.0, time_budget * 0.35)
    if time.time() - start_time < time_budget - 10:
        pts = _spectral_optimize(pts, time_budget=phase1_time)
    
    # Phase 2: Lloyd iteration
    remaining = time_budget - (time.time() - start_time)
    if remaining > 10:
        lloyd_time = min(10.0, remaining * 0.4)
        pts = _lloyd_iteration(pts, step_size=0.08, time_budget=lloyd_time)
    
    # Phase 3: Direct polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 8:
        polish_time = min(12.0, remaining * 0.5)
        pts, _ = _direct_polish(pts, time_budget=polish_time)
    
    # Phase 4: Additional Lloyd + polish cycles
    remaining = time_budget - (time.time() - start_time)
    if remaining > 6:
        for cycle in range(5):
            if time.time() - start_time > time_budget - 3:
                break
            pts = _lloyd_iteration(pts, step_size=0.04, time_budget=min(4.0, remaining * 0.25))
            remaining = time_budget - (time.time() - start_time)
            if remaining > 3:
                pts, _ = _direct_polish(pts, time_budget=min(5.0, remaining * 0.4))
                remaining = time_budget - (time.time() - start_time)
    
    return pts

def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    # Multiple starting points
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=0),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
    ]
    
    for i, pts0 in enumerate(starts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Evaluate starting point
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        
        # Multi-phase optimization
        time_budget = min(40.0, DEADLINE - (time.time() - start_time) - 20)
        if time_budget > 20:
            pts_opt = _two_phase_optimize(pts0, time_budget=time_budget)
            r_opt = _covering_radius(pts_opt)
            if r_opt < best_radius:
                best_radius = r_opt
                best_pts = pts_opt.copy()
    
    # Final high-precision polish from best
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 5:
        # Lloyd iteration with small steps
        pts_final = _lloyd_iteration(best_pts, step_size=0.02, time_budget=min(5.0, remaining * 0.3))
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining > 3:
            pts_final, _ = _direct_polish(pts_final, time_budget=min(8.0, remaining * 0.6))
        r_final = _covering_radius(pts_final)
        if r_final < best_radius:
            best_radius = r_final
            best_pts = pts_final.copy()
    
    # Ensure validity
    best_pts = _normalize(best_pts)
    return best_pts
