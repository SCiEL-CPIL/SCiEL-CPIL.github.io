"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
DEADLINE = 140  # seconds; leave 10s safety margin for evaluator overhead

def _bary_to_cart(b):
    """Convert barycentric coords (u, v) with u,v >= 0, u+v <= 1 to Cartesian in unit-height triangle."""
    # Vertices: A=(0,0), B=(1,0), C=(0.5, H)
    # Point = u*A + v*B + w*C where w = 1-u-v
    x = v * 1.0 + (1.0 - u - v) * 0.5
    y = (1.0 - u - v) * H
    return x, y

def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3 (2D points)."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _objective(bary_coords):
    """Negative of the minimum triangle area over all triples of points.
    bary_coords: shape (N, 2) with each row (u, v), u,v >= 0, u+v <= 1."""
    # Clip to valid region (small numerical violations)
    bary_coords = np.clip(bary_coords, 0, 1)
    # Ensure u + v <= 1
    s = bary_coords[:, 0] + bary_coords[:, 1]
    mask = s > 1
    if np.any(mask):
        bary_coords = bary_coords.copy()
        bary_coords[mask] *= (1.0 / s[mask])[:, None]
    
    # Convert to Cartesian
    x = bary_coords[:, 1] + (1.0 - bary_coords[:, 0] - bary_coords[:, 1]) * 0.5
    y = (1.0 - bary_coords[:, 0] - bary_coords[:, 1]) * H
    pts = np.column_stack((x, y))
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) 
                               - (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if area < min_area:
                    min_area = area
    return -min_area

def _initial_guesses():
    """Generate diverse initial guesses for optimization."""
    guesses = []
    # 1. Vertices + centroid + edge midpoints
    verts = np.array([[0, 0], [1, 0], [0.5, 1.0]])  # barycentric: A, B, C
    centroid = np.array([[1/3, 1/3]])
    mid_ab = np.array([[0.5, 0.5]])
    mid_bc = np.array([[0, 0.5]])  # midpoint of B=(1,0) and C=(0.5,1) in barycentric? Let me reconsider
    # In barycentric: A=(1,0,0)->(u=1,v=0), B=(0,1,0)->(u=0,v=1), C=(0,0,1)->(u=0,v=0)
    # Midpoint of AB: barycentric (0.5, 0.5, 0) -> (u=0.5, v=0.5)
    # Midpoint of BC: barycentric (0, 0.5, 0.5) -> (u=0, v=0.5)
    # Midpoint of CA: barycentric (0.5, 0, 0.5) -> (u=0.5, v=0)
    mid_ca = np.array([[0.5, 0.0]])
    
    base = np.vstack([verts, centroid, mid_ab, mid_bc, mid_ca])
    # That's 7 points. Add 4 more near the centroid with small perturbations
    extra = centroid[0].copy()
    for i in range(4):
        extra = base[3].copy()  # centroid
        extra[0] += 0.05 * (i - 1.5)
        extra[1] += 0.05 * (i - 1.5)
        base = np.vstack([base, extra])
    guesses.append(base[:N])
    
    # 2. Random points
    rng = np.random.RandomState(42)
    for _ in range(5):
        b = rng.random((N, 2))
        s = b.sum(axis=1)
        b = b / s[:, None]
        # Scale to be inside
        b *= 0.8
        guesses.append(b)
    
    return guesses

def construct():
    start_time = time.time()
    best_result = None
    best_score = -float('inf')
    
    # Try multiple optimization approaches
    # Approach 1: Local optimization from multiple starting points
    for guess in _initial_guesses():
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            res = minimize(_objective, guess, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        # Convert barycentric to Cartesian
        bary = np.clip(best_result, 0, 1)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 1
        if np.any(mask):
            bary = bary.copy()
            bary[mask] *= (1.0 / s[mask])[:, None]
        
        x = bary[:, 1] + (1.0 - bary[:, 0] - bary[:, 1]) * 0.5
        y = (1.0 - bary[:, 0] - bary[:, 1]) * H
        pts = np.column_stack((x, y))
        return pts
    
    # Fallback: simple grid
    pts = []
    for i in range(N):
        u = (i % 3) * 0.33
        v = (i // 3) * 0.33
        if u + v > 0.95:
            u = 0.3
            v = 0.3
        pts.append([u + 0.5 * (1 - u - v), H * (1 - u - v)])
    return np.array(pts[:N])

# EVOLVE-BLOCK-END
