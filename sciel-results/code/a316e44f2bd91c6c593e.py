import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2
_DEADLINE = 145.0

def _pairwise_distances(pts):
    """Compute all pairwise Euclidean distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _structured_config(params):
    """
    Generate 16 points based on a structured hexagonal-like lattice.
    params: [aspect_ratio, shear_angle, outer_scale]
    - aspect_ratio: y-scale relative to x-scale (1.0 is regular hex lattice)
    - shear_angle: rotation/shear of the lattice
    - outer_scale: scaling factor for the outermost ring points to break symmetry
    """
    aspect = params[0]
    shear = params[1]
    outer_scale = params[2]
    
    # Base hexagonal lattice vectors
    # v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
    # We want to generate a cluster of 16 points.
    # A common compact cluster is 1 center + 6 ring 1 + 9 ring 2 (partial).
    # Let's use coordinates from a hex lattice and select 16.
    
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    # Apply aspect ratio to v2's y-component? Or scale entire y-axis?
    # Let's scale the y-axis of the lattice.
    v2[1] *= aspect
    
    # Generate lattice points
    lattice = []
    # We need enough points to pick 16 distinct ones that form a nice cluster.
    # Range -2 to 2 in both i, j gives 25 points.
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            lattice.append(p)
    lattice = np.array(lattice)
    
    # Center the lattice
    lattice -= lattice.mean(axis=0)
    
    # Apply shear/rotation
    # Simple rotation matrix
    rot = np.array([[np.cos(shear), -np.sin(shear)],
                    [np.sin(shear), np.cos(shear)]])
    lattice = lattice @ rot.T
    
    # Identify "outer" points vs "inner" points to apply outer_scale
    # Distance from center
    dists = np.linalg.norm(lattice, axis=1)
    # Sort by distance
    sorted_indices = np.argsort(dists)
    
    # We want 16 points. Let's take the 16 closest to the center.
    # But we want to scale the outermost ones relative to the inner ones.
    # Let's define "outer" as the points in the 16-point set that have the largest distances.
    
    # First, select the 16 closest points from the generated lattice
    top_16_indices = sorted_indices[:16]
    selected_pts = lattice[top_16_indices].copy()
    
    # Calculate distances of selected points
    sel_dists = np.linalg.norm(selected_pts, axis=1)
    
    # Determine a threshold for "outer" points. 
    # Maybe the top 50% of distances in the selected set?
    median_dist = np.median(sel_dists)
    
    # Scale points that are farther than median by outer_scale
    # This allows breaking the perfect symmetry of the lattice
    for i in range(len(selected_pts)):
        if sel_dists[i] > median_dist:
            selected_pts[i] *= outer_scale
        # Optionally, we could scale inner points inversely, but let's keep it simple.
    
    return selected_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # 1. Initial guess based on regular hexagonal packing
    # Regular hex lattice: aspect=1, shear=0, outer_scale=1
    init_params = np.array([1.0, 0.0, 1.0])
    init_pts = _structured_config(init_params)
    update_best(init_pts, _score(init_pts))

    # 2. Global Optimization on the 3 parameters
    # Bounds:
    # aspect: 0.5 to 1.5
    # shear: -pi/6 to pi/6 (30 degrees)
    # outer_scale: 0.8 to 1.2
    bounds = [
        (0.5, 1.5),
        (-np.pi/6, np.pi/6),
        (0.8, 1.2)
    ]

    def param_objective(params):
        pts = _structured_config(params)
        return _score(pts)

    try:
        if time.time() < deadline - 60:
            res_de = differential_evolution(
                param_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                polish=False
            )
            if res_de.fun < best_score:
                new_pts = _structured_config(res_de.x)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
    except Exception as e:
        pass

    # 3. Local Refinement on the 3 parameters
    try:
        if time.time() < deadline - 30:
            # Use the best params found or initial
            # We need to recover params from best_pts? No, we just minimize from current best params.
            # Let's keep track of best_params
            # Actually, DE returns x. If it succeeded, we can use that.
            # If not, use init_params.
            # Let's just run minimize from multiple starting points including init and DE result if available.
            
            start_points = [init_params]
            # If DE ran, add its result
            if 'res_de' in locals() and hasattr(res_de, 'x'):
                start_points.append(res_de.x)
            
            for sp in start_points:
                if time.time() > deadline - 30:
                    break
                res_min = minimize(
                    param_objective,
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_min.fun < best_score:
                    new_pts = _structured_config(res_min.x)
                    new_score = _score(new_pts)
                    update_best(new_pts, new_score)
    except Exception as e:
        pass

    # 4. Final Polish on the 32 coordinates
    # The structured model is rigid. We can now fine-tune the actual coordinates
    # to see if we can improve further by breaking the rigid constraints slightly.
    if best_pts is not None and time.time() < deadline - 10:
        try:
            def raw_obj(x):
                pts = x.reshape(N, 2)
                return _score(pts)
            
            res_polish = minimize(
                raw_obj,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res_polish.fun < best_score:
                new_pts = res_polish.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception as e:
            pass

    if best_pts is None:
        return _structured_config(init_params)

    # Final sanity check
    d = _pairwise_distances(best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-10:
        rng = np.random.default_rng(12345)
        best_pts = best_pts + rng.normal(0, 1e-6, size=best_pts.shape)
    
    return best_pts
