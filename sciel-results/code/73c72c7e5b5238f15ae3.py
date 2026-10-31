"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
def _score(points):
    """Compute (max_dist / min_dist)^2 for a set of points."""
    n = len(points)
    # Compute all pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    # Exclude self-distances
    np.fill_diagonal(dists_sq, np.inf)
    min_d_sq = np.min(dists_sq)
    max_d_sq = np.max(dists_sq)
    if min_d_sq <= 0:
        return np.inf
    return max_d_sq / min_d_sq

def _make_hexagonal_cluster():
    """
    Create a 16-point hexagonal cluster with 3 degrees of freedom:
    - aspect_ratio (a): scales x vs y
    - shear (s): shears the lattice
    - radial_scale (r): scales outer ring radially
    
    The base cluster uses hexagonal lattice coordinates for 16 points.
    A common 16-point hexagonal arrangement:
    Center (1) + Ring 1 (6) + Ring 2 (9) = 16
    Or: 4-3-4-3-2 pattern, or use hexagonal numbers.
    
    Let's use: center + 6 in first ring + 9 in second ring (partial)
    Actually, hexagonal numbers: 1, 7, 19. So 16 = 1 + 6 + 9 (partial second ring)
    Or we can use a 4x4-ish hexagonal cluster.
    
    Let me define explicit hexagonal lattice positions for 16 points:
    Using axial coordinates in a hex grid, select 16 positions forming a compact cluster.
    """
    # Hexagonal lattice basis vectors
    # Points in a hexagonal arrangement centered at origin
    # Ring 0: (0,0) - 1 point
    # Ring 1: 6 points at distance 1
    # Ring 2: we need 9 more to reach 16
    
    # Axial coordinates for hex grid
    # Ring 0
    pts = [(0, 0)]
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append((round(np.cos(angle)), round(np.sin(angle))))
    # Ring 2: 12 points, take 9
    for i in range(12):
        angle = i * np.pi / 6 + np.pi / 6  # offset
        # Actually, let me use proper hex ring generation
        pass
    
    # Simpler: use explicit coordinates
    # Center
    pts = np.array([[0.0, 0.0]])
    # First ring (6 points)
    for i in range(6):
        angle = i * 2 * np.pi / 6
        pts = np.vstack([pts, [np.cos(angle), np.sin(angle)]])
    # Second ring: need 9 more. Use 9 of the 12 positions on ring 2
    for i in range(12):
        angle = i * 2 * np.pi / 12
        # Only add if we still need points
        if len(pts) < 16:
            pts = np.vstack([pts, [2*np.cos(angle), 2*np.sin(angle)]])
        if len(pts) >= 16:
            break
    
    return pts[:16]

def _transform_hex(pts, aspect, shear, radial):
    """Apply the 3-parameter transformation to the hexagonal cluster."""
    # Apply shear
    shear_mat = np.array([[1, shear], [0, 1]])
    # Apply aspect ratio (scale x)
    aspect_mat = np.array([[aspect, 0], [0, 1]])
    # Apply radial scaling to outer points only (points with norm > 1.5)
    norms = np.linalg.norm(pts, axis=1)
    # Outer points: those with norm > 1.5 (ring 2)
    outer_mask = norms > 1.5
    
    transformed = pts.copy()
    # Apply linear transformation to all
    transformed = transformed @ (aspect_mat @ shear_mat).T
    
    # Apply radial scaling to outer points
    for i in range(len(transformed)):
        if outer_mask[i]:
            transformed[i] *= radial
    
    return transformed

def _objective(params):
    """Objective function for optimization."""
    aspect, shear, radial = params
    pts = _make_hexagonal_cluster()
    transformed = _transform_hex(pts, aspect, shear, radial)
    return _score(transformed)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial hexagonal cluster
    base_pts = _make_hexagonal_cluster()
    
    # Initial parameters
    best_params = np.array([1.0, 0.0, 1.0])
    best_score = _score(_transform_hex(base_pts, 1.0, 0.0, 1.0))
    
    # Try a few initial guesses
    initial_guesses = [
        np.array([1.0, 0.0, 1.0]),
        np.array([1.1, 0.1, 1.0]),
        np.array([0.9, -0.1, 1.05]),
        np.array([1.0, 0.2, 0.95]),
        np.array([1.2, 0.0, 1.1]),
    ]
    
    best_result = _transform_hex(base_pts, 1.0, 0.0, 1.0)
    best_score = _score(best_result)
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective,
                guess,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_result = _transform_hex(base_pts, *res.x)
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Try a square grid as fallback/comparison
    side = int(np.ceil(N ** (1 / D)))
    grid = np.array([[i % side, i // side] for i in range(N)], dtype=float)
    grid_score = _score(grid)
    if grid_score < best_score:
        best_result = grid
        best_score = grid_score
    
    return best_result
# EVOLVE-BLOCK-END
