"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2.0
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Vectorized cross product area calculation
    # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)


def _clamp_to_triangle(points):
    """
    Project points to the interior or boundary of the equilateral triangle.
    points: (N, 2)
    Returns: (N, 2) projected points
    """
    # We check each point against the three half-planes defining the triangle.
    # If a point violates a half-plane, we project it onto the boundary line.
    # Triangle: A(0,0), B(1,0), C(0.5, H)
    # 1. y >= 0
    # 2. y <= -sqrt(3)*(x - 1)  => sqrt(3)*x + y - sqrt(3) <= 0
    # 3. y <= sqrt(3)*x         => -sqrt(3)*x + y <= 0
    
    p = points.copy()
    x = p[:, 0]
    y = p[:, 1]
    
    # Check 1: y < 0
    mask1 = y < 0
    if np.any(mask1):
        p[mask1, 1] = 0.0
        # Also clamp x to [0,1] for this edge
        p[mask1, 0] = np.clip(p[mask1, 0], 0.0, 1.0)
        
    # Check 2: sqrt(3)*x + y - sqrt(3) > 0
    # Line: y = -sqrt(3)x + sqrt(3). Normal: (sqrt(3), 1).
    # Projection of point (x,y) onto line ax+by+c=0:
    # d = (ax+by+c)/(a^2+b^2)
    # x_proj = x - a*d, y_proj = y - b*d
    mask2 = np.sqrt(3) * x + y - np.sqrt(3) > 1e-9
    if np.any(mask2):
        a = np.sqrt(3)
        b = 1.0
        c = -np.sqrt(3)
        d = (a * p[mask2, 0] + b * p[mask2, 1] + c) / (a*a + b*b)
        p[mask2, 0] = p[mask2, 0] - a * d
        p[mask2, 1] = p[mask2, 1] - b * d
        # Clamp to segment BC: x in [0.5, 1]
        p[mask2, 0] = np.clip(p[mask2, 0], 0.5, 1.0)
        # Recalculate y from x to ensure it's on the line
        p[mask2, 1] = -np.sqrt(3) * (p[mask2, 0] - 1.0)
        
    # Check 3: -sqrt(3)*x + y > 0
    # Line: y = sqrt(3)x. Normal: (-sqrt(3), 1).
    mask3 = -np.sqrt(3) * x + y > 1e-9
    if np.any(mask3):
        a = -np.sqrt(3)
        b = 1.0
        c = 0.0
        d = (a * p[mask3, 0] + b * p[mask3, 1] + c) / (a*a + b*b)
        p[mask3, 0] = p[mask3, 0] - a * d
        p[mask3, 1] = p[mask3, 1] - b * d
        # Clamp to segment CA: x in [0, 0.5]
        p[mask3, 0] = np.clip(p[mask3, 0], 0.0, 0.5)
        # Recalculate y from x
        p[mask3, 1] = np.sqrt(3) * p[mask3, 0]
        
    return p


def _objective(params):
    """
    Objective function to minimize: -min_area.
    params: (2N,)
    """
    points = params.reshape(N, 2)
    points = _clamp_to_triangle(points)
    return -_min_area(points)


def _init_lattice():
    """Initialize with a lattice of points inside the triangle."""
    points = []
    # Try a few grid densities
    for n_grid in [3, 4, 5]:
        # Generate grid in barycentric coordinates or just Cartesian and filter
        # Cartesian grid
        xs = np.linspace(0, 1, n_grid + 1)
        ys = np.linspace(0, H, n_grid + 1)
        candidates = []
        for x in xs:
            for y in ys:
                # Check if inside
                if y >= 0 and np.sqrt(3)*x + y - np.sqrt(3) <= 1e-9 and -np.sqrt(3)*x + y <= 1e-9:
                    candidates.append([x, y])
        if len(candidates) >= N:
            # Select N points from candidates, trying to spread them out
            # Simple selection: pick every k-th point or use random sampling
            # For now, just take the first N if we have enough, or sample
            if len(candidates) == N:
                return np.array(candidates)
            else:
                # Sample N points from candidates
                idx = np.random.choice(len(candidates), N, replace=False)
                return np.array([candidates[i] for i in idx])
    
    # Fallback random
    return np.random.rand(N, 2) * np.array([1.0, H])


def _init_random():
    """Initialize with random points, clamped."""
    points = np.random.rand(N, 2) * np.array([1.0, H])
    return _clamp_to_triangle(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_params = None
    best_value = -np.inf

    # Phase 1: Differential Evolution with multiple restarts
    # Bounds for each point: x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    # Try a few different initialization strategies for DE
    init_strategies = [
        'latinhypercube',
        'sobol' # if available, else latinhypercube
    ]
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        try:
            # Use a smaller population for speed, rely on restarts
            result = differential_evolution(
                _objective,
                bounds,
                seed=42 + r,
                maxiter=100,
                popsize=10,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.0),
                recombination=0.8,
                update_interval=5,
            )
            if result.fun < 0:  # Valid
                val = -result.fun
                if val > best_value:
                    best_value = val
                    best_params = result.x.copy()
        except Exception:
            continue

    # Also try starting from lattice and random initializations with DE
    # But DE with random init is usually good. Let's add a specific local search from lattice.
    
    # Phase 2: Local Refinement from best DE result and other good starts
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
    
    # Add lattice start
    lattice_points = _init_lattice()
    candidates.append(lattice_points.flatten())
    
    # Add random starts
    for _ in range(3):
        candidates.append(_init_random().flatten())

    for start_params in candidates:
        if time.time() > deadline - 10.0:
            break
        try:
            # Nelder-Mead is robust for non-smooth functions
            result2 = minimize(
                _objective,
                start_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
            )
            if result2.fun < 0:
                val2 = -result2.fun
                if val2 > best_value:
                    best_value = val2
                    best_params = result2.x.copy()
        except Exception:
            pass

    # Phase 3: Perturbation and Re-refine
    if best_params is not None and time.time() < deadline - 5.0:
        best_points = _clamp_to_triangle(best_params.reshape(N, 2))
        # Try perturbing each point slightly and re-optimizing
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            for _ in range(3): # 3 perturbations per point
                perturbed = best_points.copy()
                # Add small noise
                noise = np.random.normal(0, 0.05, size=2)
                perturbed[i] += noise
                perturbed = _clamp_to_triangle(perturbed)
                
                try:
                    result3 = minimize(
                        _objective,
                        perturbed.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True},
                    )
                    if result3.fun < 0:
                        val3 = -result3.fun
                        if val3 > best_value:
                            best_value = val3
                            best_params = result3.x.copy()
                except Exception:
                    pass

    if best_params is None:
        # Fallback
        points = _init_random()
        return points

    points = _clamp_to_triangle(best_params.reshape(N, 2))
    return points
