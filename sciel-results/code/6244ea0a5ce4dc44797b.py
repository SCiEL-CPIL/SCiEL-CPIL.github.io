"""Heilbronn triangles: Place 11 points in a unit-area equilateral triangle."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Precompute triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 145.0  # Safety margin

def _bary_to_cart(bary_pts):
    """Convert barycentric (u, v) to Cartesian (x, y). w = 1 - u - v."""
    u = bary_pts[:, 0]
    v = bary_pts[:, 1]
    x = u + 0.5 * v
    y = v * S3H
    return np.column_stack((x, y))

def _cart_to_bary(cart_pts):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart_pts[:, 0]
    y = cart_pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack((u, v))

def _project_bary_to_simplex(bary_pts):
    """Project barycentric points to the simplex u>=0, v>=0, u+v<=1."""
    proj = bary_pts.copy()
    for i in range(len(proj)):
        u, v = proj[i]
        if u < 0: u = 0
        if v < 0: v = 0
        if u + v > 1:
            s = u + v
            if s > 1e-12:
                u = u / s
                v = v / s
            else:
                u = 0
                v = 0
        proj[i] = [u, v]
    return proj

def _compute_areas(bary_pts):
    """Compute areas of all triangles formed by the points."""
    pi = bary_pts[_T_I]
    pj = bary_pts[_T_J]
    pk = bary_pts[_T_K]
    
    wi = 1 - pi[:, 0] - pi[:, 1]
    wj = 1 - pj[:, 0] - pj[:, 1]
    wk = 1 - pk[:, 0] - pk[:, 1]
    
    # Determinant of 3x3 matrix with rows (u, v, w)
    det = (pi[:, 0] * (pj[:, 1] * wk - wj * pk[:, 1]) -
           pi[:, 1] * (pj[:, 0] * wk - wj * pk[:, 0]) +
           wi     * (pj[:, 0] * pk[:, 1] - pj[:, 1] * pk[:, 0]))
    
    areas = np.abs(det) * TRI_AREA
    return areas

def _soft_min_area(bary_pts, beta=100.0):
    """Compute soft minimum area for smooth optimization."""
    areas = _compute_areas(bary_pts)
    max_a = np.max(areas)
    # To avoid overflow in exp, shift by max
    shifted = -beta * (areas - max_a)
    # log-sum-exp trick
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _objective_bary(flat_params, beta=100.0):
    """Objective function for minimization: -soft_min_area."""
    bary_pts = flat_params.reshape(-1, 2)
    # Soft penalty for going out of bounds
    sums = bary_pts[:, 0] + bary_pts[:, 1]
    viol = np.maximum(0, sums - 1)
    penalty = 10.0 * np.sum(viol**2)
    
    area = _soft_min_area(bary_pts, beta)
    return -area + penalty

def _generate_lattice_init(rng):
    """Generate a structured initial configuration based on a triangular lattice."""
    # A good heuristic for n=11 is to distribute points roughly evenly.
    # One common configuration is a 3x3 grid in barycentric coords, plus 2 extra,
    # or a hexagonal packing.
    # Let's try a jittered grid in Cartesian space, then convert to Barycentric.
    
    # Grid in Cartesian
    # Triangle vertices: (0,0), (1,0), (0.5, S3H)
    # We want to fill the triangle.
    # Simple approach: sample from a triangular grid in barycentric coords.
    # Barycentric grid points: u, v such that u>=0, v>=0, u+v<=1.
    
    # For N=11, a 3x3 grid has 6 points (if we take u,v in {0, 0.5, 1} constrained).
    # Let's try a denser set.
    # Points in barycentric space:
    # Level 0: (1,0), (0,1), (0,0) -> 3 points
    # Level 1: (0.5, 0.5), (0.5, 0), (0, 0.5) -> 3 points
    # Level 2: (1/3, 1/3), (2/3, 1/3), (1/3, 2/3) -> 3 points? 
    # This is getting complicated to ensure uniformity.
    
    # Alternative: Just use a random jittered version of a known good pattern.
    # The optimal pattern for n=11 is often close to a triangular lattice.
    # Let's create a triangular lattice in Cartesian, clip to triangle, and jitter.
    
    # Generate lattice points
    s = 0.25  # Spacing
    pts = []
    y = 0
    row = 0
    while y <= S3H + 1e-6:
        x_start = 0 if row % 2 == 0 else s/2
        x = x_start
        while x <= 1 + 1e-6:
            # Check if inside triangle
            # y <= S3H * (1 - x) for x > 0.5? 
            # Line from (0.5, S3H) to (1,0): y = -S3H*(x-1) = S3H*(1-x)
            # Line from (0,0) to (0.5, S3H): y = S3H * x
            # So y <= S3H * min(x, 1-x)
            if y <= S3H * min(x, 1-x) + 1e-6:
                pts.append([x, y])
            x += s
        y += s * S3H / 2  # Height of equilateral triangle with side s is s*sqrt(3)/2
        row += 1
    
    # We might have more or fewer than 11 points.
    # If we have more, we can randomly subsample. If fewer, add random points.
    # For N=11, s=0.25 gives roughly:
    # y=0: x=0, 0.25, 0.5, 0.75, 1 (5 pts)
    # y~0.216: x=0.125, 0.375, 0.625, 0.875 (4 pts) -> 9 pts
    # y~0.433: x=0.25, 0.5, 0.75 (3 pts) -> 12 pts
    # So we get ~12 points. We can pick 11.
    
    if len(pts) > N:
        idx = rng.choice(len(pts), N, replace=False)
        pts = [pts[i] for i in idx]
    elif len(pts) < N:
        # Add random points
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                x = u + 0.5 * v
                y = v * S3H
                pts.append([x, y])
    
    cart_pts = np.array(pts)
    # Jitter
    cart_pts += rng.normal(0, 0.05, size=cart_pts.shape)
    bary_pts = _cart_to_bary(cart_pts)
    bary_pts = _project_bary_to_simplex(bary_pts)
    return bary_pts

def _simulated_annealing(bary_pts, rng, start_temp=0.05, end_temp=0.001, steps=5000):
    """Perform Simulated Annealing on barycentric coordinates."""
    current_pts = bary_pts.copy()
    current_areas = _compute_areas(current_pts)
    current_min_area = np.min(current_areas)
    current_score = -current_min_area  # We want to maximize min area, so minimize -min area
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Precompute neighbor indices? No, random perturbations are better for global search.
    
    for i in range(steps):
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / steps)
        
        # Generate a neighbor: perturb one or a few points
        n_perturb = rng.integers(1, 4)  # Perturb 1 to 3 points
        idx = rng.choice(N, n_perturb, replace=False)
        
        new_pts = current_pts.copy()
        # Gaussian perturbation
        scale = 0.1 * temp + 0.01
        new_pts[idx] += rng.normal(0, scale, size=(n_perturb, 2))
        new_pts = _project_bary_to_simplex(new_pts)
        
        new_areas = _compute_areas(new_pts)
        new_min_area = np.min(new_areas)
        new_score = -new_min_area
        
        delta = new_score - current_score
        
        # Acceptance probability
        if delta < 0:
            if rng.random() < np.exp(-delta / temp):
                current_pts = new_pts
                current_min_area = new_min_area
                current_score = new_score
        else:
            current_pts = new_pts
            current_min_area = new_min_area
            current_score = new_score
        
        if current_score < best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
    return best_pts, -best_score

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_area = 0.0
    
    # Phase 1: Structured Initialization
    bary_init = _generate_lattice_init(rng)
    area, _ = _compute_areas(bary_init)
    min_area = np.min(area)
    best_bary = bary_init.copy()
    best_area = min_area
    
    # Phase 2: Simulated Annealing
    elapsed = time.time() - start_time
    if elapsed < 60:
        # Run SA for a fixed number of steps, checking time
        target_steps = 10000
        # Estimate time per step
        # Let's just run a loop with a time check
        sa_start = time.time()
        current_best_bary = best_bary
        current_best_area = best_area
        
        # We can run multiple SA restarts or one long one.
        # Let's do one long one from the lattice init.
        res_bary, res_area = _simulated_annealing(
            current_best_bary, rng, 
            start_temp=0.05, end_temp=0.0005, 
            steps=target_steps
        )
        if res_area > current_best_area:
            current_best_bary = res_bary
            current_best_area = res_area
            
        best_bary = current_best_bary
        best_area = current_best_area
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed < 90:
             # Run SA again from a slightly perturbed version of the best
             perturbed = best_bary + rng.normal(0, 0.05, size=(N, 2))
             perturbed = _project_bary_to_simplex(perturbed)
             res_bary2, res_area2 = _simulated_annealing(
                perturbed, rng, 
                start_temp=0.03, end_temp=0.0002, 
                steps=5000
             )
             if res_area2 > best_area:
                 best_bary = res_bary2
                 best_area = res_area2

    # Phase 3: Local Polish with Nelder-Mead
    elapsed = time.time() - start_time
    if best_bary is not None and elapsed < DEADLINE - 10:
        # Use a high beta for the soft-min to approximate the true min closely
        # This makes the objective smoother near the optimum
        try:
            result = minimize(
                lambda x: _objective_bary(x, beta=200.0),
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_bary = result.x.reshape(-1, 2)
            opt_bary = _project_bary_to_simplex(opt_bary)
            
            areas = _compute_areas(opt_bary)
            min_area = np.min(areas)
            if min_area > best_area:
                best_area = min_area
                best_bary = opt_bary.copy()
                
        except Exception:
            pass

    # Phase 4: Final Polish with even higher precision
    elapsed = time.time() - start_time
    if best_bary is not None and elapsed < DEADLINE - 2:
        try:
            result = minimize(
                lambda x: _objective_bary(x, beta=500.0),
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            opt_bary = result.x.reshape(-1, 2)
            opt_bary = _project_bary_to_simplex(opt_bary)
            
            areas = _compute_areas(opt_bary)
            min_area = np.min(areas)
            if min_area > best_area:
                best_area = min_area
                best_bary = opt_bary.copy()
                
        except Exception:
            pass

    # Convert best barycentric to Cartesian
    if best_bary is None:
        best_bary = _generate_lattice_init(rng)
        
    best_cart = _bary_to_cart(best_bary)
    
    return best_cart

