"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = 3 ** 0.5
TRI_AREA = SQRT3 / 4.0  # area of unit equilateral triangle

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

def _point_in_triangle(p):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    # Barycentric coordinates
    # p = a*V0 + b*V1 + c*V2 where a+b+c=1
    # Using the standard formula
    den = SQRT3 / 2.0
    # Area of triangle (V1, V2, p) / total area = weight of V0
    # Actually let's use a more direct method
    x, y = p
    # Check: y >= 0, y <= sqrt(3)/2 * x, y <= sqrt(3)/2 * (1-x)
    if y < -1e-10:
        return False
    if y > SQRT3 / 2.0 * x + 1e-10:
        return False
    if y > SQRT3 / 2.0 * (1.0 - x) + 1e-10:
        return False
    return True

def _project_to_triangle(p):
    """Project point p onto the interior of the unit equilateral triangle."""
    x, y = p
    # Clip to be inside the triangle
    # Constraint 1: y >= 0
    if y < 0:
        y = 0
    # Constraint 2: y <= sqrt(3)/2 * x
    if y > SQRT3 / 2.0 * x:
        # Project onto line y = sqrt(3)/2 * x
        # The line is: sqrt(3)/2 * x - y = 0
        # Normal direction: (sqrt(3)/2, -1), normalize
        # Project p onto the line
        # Actually, let's just clamp
        y = SQRT3 / 2.0 * x
    # Constraint 3: y <= sqrt(3)/2 * (1-x)
    if y > SQRT3 / 2.0 * (1.0 - x):
        y = SQRT3 / 2.0 * (1.0 - x)
    # Also need x >= 0 and x <= 1
    if x < 0:
        x = 0
        y = 0
    if x > 1:
        x = 1
        y = 0
    # Re-check after clipping
    if y < 0:
        y = 0
    if y > SQRT3 / 2.0 * x:
        y = SQRT3 / 2.0 * x
    if y > SQRT3 / 2.0 * (1.0 - x):
        y = SQRT3 / 2.0 * (1.0 - x)
    return np.array([x, y])

def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)

def _objective_flat(params):
    """Objective function for flat parameter vector. Returns negative min area."""
    points = params.reshape(N, 2)
    # Project points to ensure they're in the triangle
    points = np.array([_project_to_triangle(p) for p in points])
    areas = _all_triangle_areas(points)
    return -np.min(areas)

def _objective_sym(params):
    """
    Symmetry-reduced objective.
    params: [r0, theta0, r1, theta1, r2, theta2] for 3 orbits under 120-degree rotation.
    Plus we have 11 points. With 3-fold symmetry, we can have orbits of size 3, or fixed points.
    11 = 3*3 + 2, so we could have 3 orbits of 3 and 2 fixed points, but fixed points under 120-deg
    rotation must be at the center.
    
    Actually, let's think differently. The optimal config for Heilbronn triangles often has
    some symmetry. Let's try: center point + 3 orbits of 3 points each = 1 + 9 = 10. 
    That's 10, need 11. So maybe center + 3 orbits of 3 + 1 more = 11? No, that breaks symmetry.
    
    Let me try a different approach: use 3-fold rotational symmetry.
    Points at center: 1 point (invariant)
    Orbits of 3: we can have k orbits, giving 3k points.
    1 + 3k = 11 => 3k = 10, not integer.
    
    So exact 3-fold symmetry doesn't work for 11 points. Let me use a different strategy.
    
    Instead, let's use a multi-restart Nelder-Mead with a good initialization based on
    a known good configuration pattern.
    """
    points = params.reshape(N, 2)
    points = np.array([_project_to_triangle(p) for p in points])
    areas = _all_triangle_areas(points)
    return -np.min(areas)

def _make_lattice_init(seed=0):
    """Create an initial configuration based on a triangular lattice pattern."""
    rng = np.random.RandomState(seed)
    points = []
    
    # Start with a roughly uniform distribution
    # Use a jittered grid approach
    # Place points in a triangular lattice pattern
    
    # For 11 points, let's try a 3-4-4 arrangement or similar
    # Actually, let's use a more systematic approach
    
    # Place points on a triangular lattice
    # The equilateral triangle has side 1, height sqrt(3)/2
    
    # Let's try placing points at specific positions
    # Center: (0.5, sqrt(3)/6)
    center = np.array([0.5, SQRT3 / 6.0])
    points.append(center)
    
    # Add points in a ring pattern
    # Ring 1: 6 points at some radius
    r1 = 0.25
    for i in range(6):
        angle = i * np.pi / 3.0 + rng.uniform(-0.1, 0.1)
        p = center + r1 * np.array([np.cos(angle), np.sin(angle)])
        p = _project_to_triangle(p)
        points.append(p)
    
    # Add more points
    r2 = 0.45
    for i in range(4):
        angle = i * np.pi / 2.0 + np.pi / 4.0 + rng.uniform(-0.1, 0.1)
        p = center + r2 * np.array([np.cos(angle), np.sin(angle)])
        p = _project_to_triangle(p)
        points.append(p)
    
    # We have 1 + 6 + 4 = 11 points
    return np.array(points[:N])

def _make_random_init(seed=0):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(N):
        while True:
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, SQRT3 / 2.0 * 0.95)
            p = np.array([x, y])
            if _point_in_triangle(p):
                points.append(p)
                break
    return np.array(points)

def _make_vertex_init(seed=0):
    """Create initial config with points near vertices and center."""
    rng = np.random.RandomState(seed)
    points = []
    
    # Near vertices
    for v in [V0, V1, V2]:
        p = v + rng.uniform(-0.05, 0.05, size=2)
        p = _project_to_triangle(p)
        points.append(p)
    
    # Center
    points.append(np.array([0.5, SQRT3 / 6.0]))
    
    # Midpoints of edges
    mid01 = (V0 + V1) / 2.0
    mid12 = (V1 + V2) / 2.0
    mid20 = (V2 + V0) / 2.0
    for m in [mid01, mid12, mid20]:
        p = m + rng.uniform(-0.03, 0.03, size=2)
        p = _project_to_triangle(p)
        points.append(p)
    
    # A few more
    for _ in range(2):
        while True:
            x = rng.uniform(0.2, 0.8)
            y = rng.uniform(0.1, SQRT3 / 2.0 * 0.8)
            p = np.array([x, y])
            if _point_in_triangle(p):
                points.append(p)
                break
    
    return np.array(points[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initializations and optimize
    inits = []
    
    # Lattice-based inits
    for s in range(5):
        inits.append(_make_lattice_init(s))
    
    # Vertex-based inits
    for s in range(5):
        inits.append(_make_vertex_init(s))
    
    # Random inits
    for s in range(5):
        inits.append(_make_random_init(s))
    
    # Also try the seed program's approach as a baseline
    import random
    rng = random.Random(7)
    seed_points = []
    while len(seed_points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.98 and u > 0.01 and v > 0.01:
            seed_points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
    inits.append(np.array(seed_points))
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        # Ensure init is valid
        init = np.array([_project_to_triangle(p) for p in init])
        
        # Evaluate initial
        areas = _all_triangle_areas(init)
        init_score = np.min(areas)
        if init_score > best_score:
            best_score = init_score
            best_points = init.copy()
        
        # Nelder-Mead optimization
        try:
            result = minimize(
                _objective_flat,
                init.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            opt_points = result.x.reshape(N, 2)
            opt_points = np.array([_project_to_triangle(p) for p in opt_points])
            
            areas = _all_triangle_areas(opt_points)
            opt_score = np.min(areas)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            pass
    
    # If we have time, do a final polish with more iterations on the best found
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective_flat,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            opt_points = result.x.reshape(N, 2)
            opt_points = np.array([_project_to_triangle(p) for p in opt_points])
            
            areas = _all_triangle_areas(opt_points)
            opt_score = np.min(areas)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return seed program result
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
    
    return best_points
