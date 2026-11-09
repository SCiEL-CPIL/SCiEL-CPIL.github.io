"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _hexagonal_lattice():
    """Generate a high-quality hexagonal lattice packing."""
    # Optimal hexagonal packing density is pi / (2 * 3**0.5) ~ 0.9069
    # For N=300 circles in unit disk:
    # Area of N circles = N * pi * r^2
    # Area of container = pi * 1^2
    # Density = N * r^2 / 1^2
    # r ~ sqrt(Density / N) = sqrt(0.9069 / 300) ~ 0.055
    # Let's try to fit as many as possible with a slightly smaller r to allow optimization
    best_r = 0.0
    best_points = None
    
    # We want to generate a lattice that is dense enough.
    # Let's iterate over potential radii or spacings.
    # Spacing s = 2*r.
    # For a hex lattice, number of points in radius R is approx pi * R^2 / (s^2 * sqrt(3)/2)
    # We want N points.
    # s ~ sqrt(2 * pi * 1^2 / (N * sqrt(3)))
    est_s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Try a range of spacings around the estimate
    for spacing in np.linspace(est_s * 0.95, est_s * 1.05, 20):
        a = spacing
        h = a * np.sqrt(3) / 2
        
        points = []
        # Generate points in a large enough bounding box
        max_range = int(np.ceil(1.0 / h)) + 2
        for j in range(-max_range, max_range + 1):
            for i in range(-max_range, max_range + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        # Select N points closest to center to maximize inner density
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
    
    return best_points, best_r

def _objective(x, N):
    """
    Objective function to minimize (negative radius).
    x is a flattened array of 2*N + 1 values: [x_1, y_1, ..., x_N, y_N, r]
    """
    points = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    # Check validity
    # 1. Boundary constraint: ||c_i|| + r <= 1  =>  ||c_i|| <= 1 - r
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0 - r + 1e-9):
        # Penalty for violating boundary
        penalty = np.sum(np.maximum(norms - (1.0 - r), 0)) ** 2
        return -r + penalty
    
    # 2. Non-overlap constraint: ||c_i - c_j|| >= 2r
    d = pdist(points)
    min_d = np.min(d)
    if min_d < 2*r - 1e-9:
        # Penalty for overlapping
        penalty = np.sum(np.maximum(2*r - d, 0)) ** 2
        return -r + penalty
        
    return -r

def _constraints(x, N):
    """
    Constraints for SLSQP.
    Returns a list of constraint dicts.
    """
    points = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    # 1. Boundary: 1 - r - ||c_i|| >= 0
    # We can define this as a single vector constraint or individual.
    # SLSQP handles vector inequalities.
    c_boundary = 1.0 - r - np.linalg.norm(points, axis=1)
    
    # 2. Non-overlap: ||c_i - c_j|| - 2r >= 0
    # This is O(N^2) constraints. For N=300, that's ~45k constraints.
    # SLSQP might struggle with this many.
    # Alternative: Use a penalty method or reduce constraints.
    # However, the directive asks for "explicit nonlinear inequality constraints".
    # Let's try to implement it, but be aware of performance.
    
    # To make it faster, we can only enforce constraints for pairs that are "close".
    # But defining "close" dynamically is tricky for the constraint function signature.
    # Let's stick to all pairs for correctness in the model, but optimize the evaluation.
    
    d = pdist(points)
    c_overlap = d - 2*r
    
    # SLSQP requires constraints in the form: fun(x) >= 0 (type='ineq')
    # We can return a single function that returns a vector of all constraint values.
    # But SLSQP's 'ineq' constraint expects a function that returns a scalar or vector.
    # If vector, all elements must be >= 0.
    
    # Combining them into one vector
    all_constraints = np.concatenate([c_boundary, c_overlap])
    
    return {'type': 'ineq', 'fun': lambda x: all_constraints, 'jac': None}

def _optimize_constrained(points, r_init, max_time=60):
    """
    Optimize the packing using SLSQP with explicit constraints.
    """
    if points is None:
        return points, r_init
    
    # Flatten points and append r
    x0 = np.concatenate([points.flatten(), [r_init]])
    
    # Bounds for x, y: [-1, 1] (loose, constraints handle it)
    # Bounds for r: [0, 1]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 1.0)]
    
    # Define constraint function properly for SLSQP
    def constraint_fun(x):
        pts = x[:2*N].reshape(N, 2)
        r_val = x[2*N]
        
        # Boundary
        c_b = 1.0 - r_val - np.linalg.norm(pts, axis=1)
        
        # Overlap
        d = pdist(pts)
        c_o = d - 2*r_val
        
        return np.concatenate([c_b, c_o])
    
    constraints = ({'type': 'ineq', 'fun': constraint_fun},)
    
    start_time = time.time()
    best_x = x0.copy()
    best_r = r_init
    best_points = points.copy()
    
    try:
        # SLSQP can be slow with many constraints.
        # We limit iterations and time.
        result = minimize(
            _objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        
        if time.time() - start_time < max_time:
            if result.success or result.fun < -best_r:
                # Check if the result is actually valid (constraints satisfied)
                c_vals = constraint_fun(result.x)
                if np.all(c_vals >= -1e-6):
                    new_points = result.x[:2*N].reshape(N, 2)
                    new_r = result.x[2*N]
                    # Verify computed radius
                    actual_r = _compute_radius(new_points)
                    if actual_r > best_r:
                        best_r = actual_r
                        best_points = new_points
                        best_x = result.x
    except Exception as e:
        pass

    # If SLSQP failed or didn't improve, try a few more restarts or different methods?
    # For now, return the best found.
    
    return best_points, best_r

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # 1. Get initial hexagonal lattice
    hex_points, hex_r = _hexagonal_lattice()
    
    if hex_points is None:
        # Fallback
        spacing = 0.06
        points = []
        for i in range(-17, 18):
            for j in range(-17, 18):
                x = i * spacing
                y = j * spacing
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            points = points[idx]
        else:
            # Should not happen
            points = np.zeros((N, 2))
        hex_r = _compute_radius(points)
        hex_points = points

    best_points = hex_points
    best_r = hex_r
    
    # 2. Optimize with SLSQP
    if time.time() < deadline - 10:
        opt_points, opt_r = _optimize_constrained(hex_points, hex_r, max_time=60)
        if opt_r > best_r:
            best_points = opt_points
            best_r = opt_r
            
    # 3. Local refinement (simple perturbation) if time remains
    # This helps fix any numerical drift or slightly improve if SLSQP got stuck
    if time.time() < deadline - 5:
        # Simple local search
        for _ in range(10): # Max 10 passes
            if time.time() >= deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() >= deadline - 5:
                    break
                # Try small perturbations
                for dx in np.linspace(-0.002, 0.002, 5):
                    for dy in np.linspace(-0.002, 0.002, 5):
                        new_points = best_points.copy()
                        new_points[i] = [best_points[i][0] + dx, best_points[i][1] + dy]
                        
                        # Quick check: inside boundary?
                        if np.linalg.norm(new_points[i]) > 1.0:
                            continue
                        
                        # Full check
                        r_new = _compute_radius(new_points)
                        if r_new > best_r + 1e-9:
                            best_r = r_new
                            best_points = new_points
                            improved = True
            if not improved:
                break

    return best_points
# EVOLVE-BLOCK-END
