"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

# Pre-compute all index triples for 11 points
_TRIPLES = np.array(list(combinations(range(N), 3)))  # shape (165, 3)

def _min_triangle_area(pts):
    """Vectorized: compute min area of all 165 triangles from 11 points."""
    # pts shape (11, 2)
    i, j, k = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]
    pi = pts[i]  # (165, 2)
    pj = pts[j]
    pk = pts[k]
    # Cross product: (pj - pi) x (pk - pi), take abs/2
    d1 = pj - pi
    d2 = pk - pi
    areas = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    return np.min(areas)

def _clamp_to_triangle(p):
    """Clamp a single point (x, y) to be inside the equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p
    eps = 1e-12
    # Inside triangle: y >= 0, y <= h*(1-x) for x in [0,1]... 
    # Actually vertices are (0,0), (1,0), (0.5, h)
    # Constraints: y >= 0, y <= h - 2*sqrt(3)/2 * (x - 0.5) ... let me think
    # The three edges:
    # Bottom: y = 0
    # Left edge from (0,0) to (0.5, h): y = sqrt(3)*x => x >= y/sqrt(3) => x - y/sqrt(3) >= 0
    # Right edge from (1,0) to (0.5, h): y = -sqrt(3)*(x-1) => y + sqrt(3)*x - sqrt(3) <= 0
    # So: y >= 0, x >= y/sqrt(3), y <= sqrt(3)*(1-x)
    # i.e., y/sqrt(3) + x <= 1... wait let me redo
    # Left edge: from (0,0) to (0.5, h). Slope = h/0.5 = sqrt(3). Line: y = sqrt(3)*x
    #   Inside is to the right: x >= y/sqrt(3), i.e., x - y/sqrt(3) >= 0
    # Right edge: from (1,0) to (0.5, h). Slope = h/(0.5-1) = -sqrt(3). Line: y = -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    #   Inside is below: y <= sqrt(3)*(1-x), i.e., y + sqrt(3)*x - sqrt(3) <= 0
    # Bottom: y >= 0
    
    # Project onto triangle
    # Simple approach: check each constraint and project
    s3 = np.sqrt(3)
    
    # Constraint 1: y >= 0
    if y < 0:
        y = 0
    
    # Constraint 2: x - y/s3 >= 0 => x >= y/s3
    if x < y / s3:
        # Project onto line x = y/s3, i.e., y = s3*x
        # Closest point on line y = s3*x to (x,y)
        # The line passes through origin with direction (1, s3)
        t = (x + s3*y) / (1 + 3)  # projection onto direction (1, s3), norm^2 = 1+3=4
        x = t
        y = s3 * t
        # But also need y >= 0 and y <= s3*(1-x)
        if y < 0:
            y = 0
            x = 0
    
    # Constraint 3: y + s3*x - s3 <= 0 => y <= s3*(1-x)
    if y > s3 * (1 - x):
        # Project onto line y = s3*(1-x), i.e., y + s3*x = s3
        # Line: s3*x + y = s3, normal (s3, 1), norm^2 = 3+1=4
        # Distance from (x,y) to line: (s3*x + y - s3) / 2
        # Projection: (x,y) - (s3*x+y-s3)/4 * (s3, 1)
        d = (s3 * x + y - s3) / 4
        x = x - s3 * d
        y = y - d
        if y < 0:
            y = 0
            x = 1  # corner (1,0)
    
    # Final clamp to [0,1] x [0, h]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, h)
    return np.array([x, y])

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _objective_flat(flat):
    """Objective for DE: flat array of 22 values (11 points x 2 coords)."""
    pts = flat.reshape(N, 2)
    pts = _clamp_all(pts)
    return -_min_triangle_area(pts)  # minimize negative = maximize min area

def _hexagonal_init():
    """Generate a hexagonal packing of 11 points in the triangle."""
    # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # Area = sqrt(3)/4
    # For 11 points, hexagonal packing...
    # Let's try a structured approach: rows of points
    # Row 0 (bottom): points along y=0
    # Row 1: slightly above, offset
    # etc.
    
    h = np.sqrt(3) / 2
    s3 = np.sqrt(3)
    
    # Try a 4-3-3-1 or similar arrangement
    # Actually, let's use a more systematic approach
    # Place points in a hexagonal lattice scaled to fit
    
    # Hexagonal lattice spacing: for n points, roughly sqrt(area/n) * factor
    # Area of triangle = sqrt(3)/4 ≈ 0.433
    # Per point area ≈ 0.0394
    # Hexagonal cell area for spacing d: d^2 * sqrt(3)/2
    # So d^2 * 0.866 ≈ 0.0394 => d^2 ≈ 0.0455 => d ≈ 0.213
    
    d = 0.22  # spacing
    
    points = []
    # Generate hexagonal lattice points
    row = 0
    while len(points) < N:
        y = row * d * s3 / 2
        if y > h:
            break
        # x offset for this row
        x_start = 0 if row % 2 == 0 else d / 2
        x = x_start
        while len(points) < N:
            # Check if (x, y) is inside triangle
            if y >= -1e-10 and x >= y / s3 - 1e-10 and y <= s3 * (1 - x) + 1e-10:
                if 0 <= x <= 1 and 0 <= y <= h:
                    points.append([x, y])
            x += d
        row += 1
    
    # If we don't have enough, add more
    while len(points) < N:
        points.append([0.5, h * 0.5])  # center as fallback
    
    return np.array(points[:N])

def construct():
    t0 = time.time()
    deadline = t0 + 120  # safety margin before 150s kill
    
    # Start with hexagonal initialization
    pts0 = _hexagonal_init()
    best_pts = pts0.copy()
    best_val = _min_triangle_area(best_pts)
    
    # Differential Evolution
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    def _de_objective(flat):
        pts = flat.reshape(N, 2)
        pts = _clamp_all(pts)
        return -_min_triangle_area(pts)
    
    try:
        result = differential_evolution(
            _de_objective,
            bounds,
            maxiter=80,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.7,
            tol=1e-8,
            seed=42,
            polish=False,  # We'll do our own polishing
            update_interval=5,
            callback=lambda x, convergence: (time.time() > deadline)
        )
        
        if result.success or result.x is not None:
            de_pts = _clamp_all(result.x.reshape(N, 2))
            de_val = _min_triangle_area(de_pts)
            if de_val > best_val:
                best_pts = de_pts
                best_val = de_val
    except Exception:
        pass
    
    # Nelder-Mead polishing on the best point
    if time.time() < deadline - 5:
        try:
            def _nm_objective(flat):
                pts = flat.reshape(N, 2)
                pts = _clamp_all(pts)
                return -_min_triangle_area(pts)
            
            nm_result = minimize(
                _nm_objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            nm_pts = _clamp_all(nm_result.x.reshape(N, 2))
            nm_val = _min_triangle_area(nm_pts)
            if nm_val > best_val:
                best_pts = nm_pts
                best_val = nm_val
        except Exception:
            pass
    
    # Try a few more local searches from perturbed best points
    if time.time() < deadline - 3:
        rng = np.random.RandomState(123)
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            perturbed = best_pts + rng.normal(0, 0.05, (N, 2))
            perturbed = _clamp_all(perturbed)
            
            try:
                def _nm_obj2(flat):
                    pts = flat.reshape(N, 2)
                    pts = _clamp_all(pts)
                    return -_min_triangle_area(pts)
                
                nm2 = minimize(
                    _nm_obj2,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                nm2_pts = _clamp_all(nm2.x.reshape(N, 2))
                nm2_val = _min_triangle_area(nm2_pts)
                if nm2_val > best_val:
                    best_pts = nm2_pts
                    best_val = nm2_val
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
