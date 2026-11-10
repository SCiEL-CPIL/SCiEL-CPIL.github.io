import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """
    Generate an initial packing using a hexagonal lattice clipped to the circle.
    This is often a better local optimum start than concentric rings for dense packings.
    """
    # Estimate spacing for N=50 in unit circle.
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # r ~ sqrt(1/N) ~ 0.14. Spacing ~ 2r ~ 0.28.
    # Let's use a grid with spacing 0.25 to be safe, then filter.
    spacing = 0.22
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    
    xs = np.arange(x_min, x_max, spacing)
    ys = np.arange(y_min, y_max, spacing)
    
    points = []
    for i, x in enumerate(xs):
        # Offset every other row
        offset = spacing * 0.866 if i % 2 == 1 else 0.0
        for y in ys:
            points.append([x + offset, y])
            
    points = np.array(points)
    
    # Keep only points inside a slightly smaller circle to ensure boundary constraints are feasible
    # We want them strictly inside, so radius < 1. Let's use 0.95 for init.
    dists = np.hypot(points[:, 0], points[:, 1])
    mask = dists < 0.95
    points = points[mask]
    
    # If we have more than N points, we need to select N.
    # We want them to be well-distributed. 
    # Simple strategy: pick the N points closest to the center? No, that leaves gaps.
    # Better: pick N points that maximize the minimum distance? That's expensive.
    # Heuristic: Pick points in a spiral order from center outwards.
    if len(points) > N:
        # Sort by distance from center
        dists_to_center = np.hypot(points[:, 0], points[:, 1])
        sorted_indices = np.argsort(dists_to_center)
        # Take first N? This creates a cluster in the center.
        # Instead, let's just take the first N from the generated list, 
        # but ensure the list is generated in a way that covers the area.
        # The arange generation is row-major.
        # Let's try a different init: random points inside circle, then relax?
        # No, let's stick to structured.
        
        # Alternative: Use the concentric ring init as primary, hex as secondary.
        # The prompt asks for multi-start. 
        # Let's just return the first N points from a dense hex grid centered at origin.
        # To avoid clustering, we can sort by angle then radius?
        
        # Let's try: generate a grid, filter by radius < 1, 
        # then use a greedy selection to pick N points maximizing min distance?
        # Too slow for init? N=50, points ~ 100-200. 
        # Greedy: 
        # 1. Start with center point.
        # 2. Add point with max min-distance to existing set.
        
        candidates = points
        if len(candidates) == 0:
             return _concentric_packing()
             
        # Find center-most point
        dists_c = np.hypot(candidates[:, 0], candidates[:, 1])
        idx = np.argmin(dists_c)
        selected = [candidates[idx]]
        candidates = np.delete(candidates, idx, axis=0)
        
        while len(selected) < N and len(candidates) > 0:
            sel = np.array(selected)
            # Compute min distance from each candidate to any selected point
            # candidates: (M, 2), selected: (K, 2)
            # dists: (M, K)
            dx = candidates[:, 0:1] - sel[:, 0:1].T
            dy = candidates[:, 1:2] - sel[:, 1:2].T
            d = np.sqrt(dx**2 + dy**2)
            min_d = np.min(d, axis=1)
            # Pick candidate with max min_d
            idx = np.argmax(min_d)
            selected.append(candidates[idx])
            candidates = np.delete(candidates, idx, axis=0)
            
        return np.array(selected)
    else:
        # If we have fewer than N, pad with concentric ring points or just return what we have?
        # We must return exactly N.
        # If hex grid is too sparse, fall back to concentric.
        return _concentric_packing()

def _concentric_packing():
    """
    Generate an initial packing using concentric rings.
    """
    ring_counts = [1, 6, 12, 18, 13]
    radii = [0.0, 0.25, 0.50, 0.75, 0.90]
    
    centers = []
    for k, count in enumerate(ring_counts):
        r_k = radii[k]
        if count == 1:
            centers.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            if k % 2 == 1:
                angles += np.pi / count
            for a in angles:
                x = r_k * np.cos(a)
                y = r_k * np.sin(a)
                centers.append([x, y])
                
    return np.array(centers)

def _sqp_objective_and_grad(x, i_idx, j_idx):
    """
    Objective: -r (we maximize r, so minimize -r)
    x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    Constraints handled by SLSQP via separate function or integrated?
    SLSQP accepts a 'constraints' dict.
    We will define the objective here, and constraints separately.
    """
    r = x[-1]
    # Minimize -r
    obj = -r
    # Gradient: 0 for centers, -1 for r
    grad = np.zeros_like(x)
    grad[-1] = -1.0
    return obj, grad

def _make_constraints(i_idx, j_idx):
    """
    Return a list of constraint dicts for SLSQP.
    Constraints:
    1. |c_i - c_j| >= 2r  => |c_i - c_j| - 2r >= 0
    2. |c_i| + r <= 1     => 1 - |c_i| - r >= 0
    """
    constraints = []
    
    # Pairwise constraints
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        
        def con_pair(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            d = np.linalg.norm(ci - cj)
            return d - 2*r
            
        def con_pair_grad(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            d = np.linalg.norm(ci - cj)
            grad = np.zeros_like(x)
            if d > 1e-8:
                vec = (ci - cj) / d
                grad[2*i:2*i+2] = vec
                grad[2*j:2*j+2] = -vec
            # Derivative of -2r
            grad[-1] = -2.0
            return grad
            
        constraints.append({
            'type': 'ineq',
            'fun': con_pair,
            'jac': con_pair_grad
        })
        
    # Boundary constraints
    for i in range(N):
        def con_bound(x, i=i):
            ci = x[2*i:2*i+2]
            r = x[-1]
            d = np.linalg.norm(ci)
            return 1.0 - d - r
            
        def con_bound_grad(x, i=i):
            ci = x[2*i:2*i+2]
            r = x[-1]
            d = np.linalg.norm(ci)
            grad = np.zeros_like(x)
            if d > 1e-8:
                grad[2*i:2*i+2] = -ci / d
            # Derivative of -r
            grad[-1] = -1.0
            return grad
            
        constraints.append({
            'type': 'ineq',
            'fun': con_bound,
            'jac': con_bound_grad
        })
        
    return constraints

def construct():
    deadline = time.time() + 145.0
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    constraints = _make_constraints(i_idx, j_idx)
    
    best_x = None
    best_val = 0.0
    
    # Multi-start configurations
    inits = []
    
    # 1. Hexagonal
    hex_centers = _hexagonal_packing()
    # Estimate initial r
    r_est = _compute_radius(hex_centers) * 0.9
    x0_hex = np.concatenate([hex_centers.flatten(), [r_est]])
    inits.append(x0_hex)
    
    # 2. Concentric
    conc_centers = _concentric_packing()
    r_est = _compute_radius(conc_centers) * 0.9
    x0_conc = np.concatenate([conc_centers.flatten(), [r_est]])
    inits.append(x0_conc)
    
    # 3. Random (seeded for reproducibility)
    np.random.seed(42)
    for _ in range(2):
        # Generate random points inside circle
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
        x_rand = radii * np.cos(angles)
        y_rand = radii * np.sin(angles)
        rand_centers = np.column_stack((x_rand, y_rand))
        r_est = 0.1
        x0_rand = np.concatenate([rand_centers.flatten(), [r_est]])
        inits.append(x0_rand)
        
    for x0 in inits:
        if time.time() > deadline - 20:
            break
            
        # SLSQP
        # maxiter needs to be high enough. N=50, 100 vars, ~1275 constraints.
        # This is heavy. 
        # We might need to reduce constraints or use a penalty method if SLSQP is too slow.
        # However, SLSQP is robust for small N.
        
        try:
            res = minimize(
                _sqp_objective_and_grad,
                x0,
                args=(i_idx, j_idx),
                method='SLSQP',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2*N) + [(0.0, 1.0)],
                constraints=constraints,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-9,
                    'iprint': 0
                }
            )
            
            if res.success or res.fun < -best_val:
                x_res = res.x
                centers_res = x_res[:2*N].reshape(N, 2)
                r_res = x_res[-1]
                
                # Verify validity
                r_valid = _compute_radius(centers_res)
                
                # The optimizer might return an r that violates constraints slightly due to tolerance.
                # We should use the valid radius computed from geometry, not the variable r,
                # but we can use r as a hint.
                # Actually, the score is based on the geometric min distance.
                # So we evaluate the geometric radius.
                
                if r_valid > best_val:
                    best_val = r_valid
                    best_x = centers_res.copy()
                    
        except Exception as e:
            # If SLSQP fails or times out, try a fallback or just skip
            pass

    # Fallback if no valid solution found (should not happen with inits)
    if best_x is None:
        best_x = _concentric_packing()
        best_val = _compute_radius(best_x)
        
    return best_x

# Dummy functions to maintain interface compatibility
def _soft_min_value_and_grad(x, beta):
    return 0.0, np.zeros_like(x)

def _build_grid(centers, cell_size):
    return {}

def _get_neighbors(grid, x, y, cell_size, search_radius):
    return []

def _calc_forces(centers, r_target):
    return np.zeros_like(centers)

def _anneal(centers, deadline):
    return centers, _compute_radius(centers)

def _hexagonal_fallback():
    return _concentric_packing()
