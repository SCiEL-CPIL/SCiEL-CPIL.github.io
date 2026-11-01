import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to minimum pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_ratio_obj(pts, sharpness=20.0):
    """
    A smooth surrogate objective function for optimization.
    Instead of hard max/min, we use soft-min and soft-max (log-sum-exp variants)
    to create a differentiable landscape that approximates the ratio.
    We minimize (soft_max / soft_min)^2.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Soft min: -1/k * log(sum(exp(-k*d)))
    # Soft max: 1/k * log(sum(exp(k*d)))
    # To avoid overflow, we shift by min/max
    d_min = d.min()
    d_max = d.max()
    
    # If points are too close, penalize heavily
    if d_min < 1e-6:
        return 1e6
    
    # Use a moderate sharpness to keep gradients stable
    k = sharpness
    
    # Soft min calculation
    exp_neg_k_d = np.exp(-k * (d - d_min))
    soft_min = d_min - (1.0 / k) * np.log(np.sum(exp_neg_k_d))
    
    # Soft max calculation
    exp_k_d = np.exp(k * (d - d_max))
    soft_max = d_max + (1.0 / k) * np.log(np.sum(exp_k_d))
    
    if soft_min < 1e-9:
        return 1e6
        
    ratio = soft_max / soft_min
    return ratio ** 2

def _get_snub_disphenoid_coords():
    """
    Generate the 14 vertices of a snub disphenoid.
    The snub disphenoid (or tetragonal disphenoid with snubbing) has 14 vertices.
    Actually, the standard "snub disphenoid" has 14 vertices and is one of the Johnson solids (J84 is not right, it's J... wait).
    Let's check: The snub disphenoid is a convex polyhedron with 14 vertices.
    It can be constructed by snubbing a disphenoid (a tetrahedron with 4 congruent isosceles triangles).
    
    Alternatively, we can use a known set of coordinates for a 14-point configuration with high symmetry.
    A simpler robust initialization is to use the vertices of a regular icosahedron (12) plus 2 extra points, 
    but that breaks symmetry.
    
    Better: Use the 14 vertices of the "snub cube" projected? No, snub cube has 24.
    
    Let's use a deterministic high-quality initialization:
    1. Start with a regular icosahedron (12 points).
    2. Add 2 points near the "poles" or in gaps?
    
    Actually, for N=14, the optimal packing on a sphere is known to be a perturbation of a specific structure.
    However, we are in Euclidean space, not constrained to a sphere.
    
    Let's use a "sphere-filling" initialization:
    Generate points using a Fibonacci sphere, then optimize.
    But to be "materially different", let's use a **simulated annealing** approach on the coordinates,
    which is a global optimization meta-heuristic, distinct from the gradient descent in the parent.
    
    Wait, the prompt asks for a *different design family*.
    Parent: Random restarts + Gradient Refinement on Sphere.
    New: **Deterministic Multi-Start Local Optimization on Euclidean Space using Smooth Surrogate**.
    
    I will implement a robust multi-start BFGS optimizer.
    Initializations:
    1. Fibonacci Sphere (normalized)
    2. Icosahedron + 2 random perturbations
    3. Random Gaussian blobs (centered)
    
    The key is using `scipy.optimize.minimize` with a smooth objective, which is more robust
    than hand-coded gradient steps, and allowing points to move in full 3D space (not just on a sphere),
    though they will likely converge to a sphere-like shape due to the ratio objective.
    """
    pass # Placeholder, logic moved to construct

def _optimize_from_start(start_pts, time_limit):
    """
    Run scipy.optimize.minimize on the smooth surrogate objective.
    """
    start_time = time.time()
    
    # Flatten points for scipy
    x0 = start_pts.flatten()
    
    def obj_flat(x):
        pts = x.reshape(N, D)
        return _soft_ratio_obj(pts)
    
    # Use BFGS for local optimization
    # We don't provide gradients to let scipy estimate them (finite difference),
    # or we could provide them, but finite difference is safer for implementation simplicity
    # and robustness against coding errors in gradients.
    # However, finite difference is slow. Let's try providing gradients if possible?
    # For now, let's rely on BFGS with finite differences but limit iterations.
    
    # Actually, let's use L-BFGS-B which is memory efficient.
    # We will run this with a tight iteration limit to respect time.
    
    try:
        res = minimize(
            obj_flat, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > time_limit:
            return start_pts # Timeout, return original
            
        return res.x.reshape(N, D)
        
    except Exception:
        return start_pts

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = np.inf
    
    # Generate a set of diverse initializations
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N)
    phi = np.arccos(1 - 2*(i+0.5)/N)
    theta = np.pi*(1+5**0.5)*i
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    fib_pts = np.column_stack([x, y, z])
    inits.append(fib_pts)
    
    # 2. Icosahedron based (12 verts + 2 extra)
    phi_ratio = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_ratio, 1], [0, phi_ratio, -1], [0, -phi_ratio, 1], [0, -phi_ratio, -1],
        [1, 0, phi_ratio], [1, 0, -phi_ratio], [-1, 0, phi_ratio], [-1, 0, -phi_ratio],
        [phi_ratio, 1, 0], [phi_ratio, -1, 0], [-phi_ratio, 1, 0], [-phi_ratio, -1, 0]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Take first 12, add 2 random points on sphere
    icosa_pts = verts[:12].copy()
    extra = rng.normal(0, 1, size=(2, 3))
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    icosa_pts = np.vstack([icosa_pts, extra])
    inits.append(icosa_pts)
    
    # 3. Random Gaussian (centered)
    for _ in range(5):
        rand_pts = rng.normal(0, 1, size=(N, 3))
        rand_pts -= rand_pts.mean(axis=0)
        inits.append(rand_pts)
    
    # 4. Random Sphere
    for _ in range(5):
        rand_sph = rng.normal(0, 1, size=(N, 3))
        rand_sph = rand_sph / np.linalg.norm(rand_sph, axis=1, keepdims=True)
        inits.append(rand_sph)

    # Optimization Loop
    # We have ~145 seconds.
    # Each optimization takes maybe 0.5 - 2 seconds depending on convergence.
    # We can afford ~50-100 optimizations.
    
    for init in inits:
        if time.time() - _t0 > DEADLINE - 10:
            break
            
        # Run optimization
        # Give each optimization up to 5 seconds
        opt_pts = _optimize_from_start(init, time_limit=5.0)
        
        # Evaluate true score
        score = _true_score(opt_pts)
        
        if score < best_score:
            best_score = score
            best_pts = opt_pts.copy()
            
        # If we found a very good score, we can try to refine it further with a higher sharpness
        # or just accept it.
        if best_score < 1.1:
            # We are close to optimal (theoretical lower bound for N=14 is unknown but likely ~1.0-1.1)
            # Do a final polish with higher precision
            pass

    # If we haven't found anything (shouldn't happen), fallback
    if best_pts is None:
        best_pts = inits[0]
        
    # Final check and return
    # Ensure distinct points
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        # Add tiny noise to separate points
        best_pts += rng.normal(0, 1e-4, size=(N, D))
        
    return best_pts
