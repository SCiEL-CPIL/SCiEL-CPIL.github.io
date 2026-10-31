"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def _pairwise_distances(pts):
    """Compute all pairwise distances."""
    return cdist(pts, pts)

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 0:
        return np.inf
    return (dmax / dmin) ** 2

def _objective(pts):
    """Objective function for minimization: (max/min)^2."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return 1e10
    return (dmax / dmin) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Start with a regular hexagonal lattice subset
    # For 16 points, try a hexagonal arrangement
    # A 4x4 grid is the seed; let's start with something better
    
    # Initial configuration: regular hexagonal packing
    # Points in a hexagonal lattice
    sqrt3 = np.sqrt(3)
    # Create a larger hexagonal grid and pick 16 points
    # Use a pattern that's close to optimal
    
    # Start with a 4x4 grid perturbed into hexagonal pattern
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    pts = np.column_stack([x.ravel(), y.ravel()])
    
    # Apply hexagonal offset to make it more like hexagonal packing
    for i in range(4):
        for j in range(4):
            if j % 2 == 1:
                pts[i*4 + j, 0] += 0.5
    
    pts = pts.astype(float)
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Try multiple initial configurations
    # Configuration 1: Hexagonal
    # Configuration 2: Square grid
    sq_pts = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    sq_score = _score(sq_pts)
    if sq_score < best_score:
        best_pts = sq_pts.copy()
        best_score = sq_score
    
    # Configuration 3: Circular arrangement with two rings
    # Inner ring: 6 points, outer ring: 10 points? Or 4 inner, 12 outer
    # Let's try: 1 center, 5 inner ring, 10 outer ring = 16
    # Or: 4 inner (square), 12 outer (regular 12-gon)
    
    # Try: regular polygon arrangements
    # 16 points on a circle
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circ_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    circ_score = _score(circ_pts)
    if circ_score < best_score:
        best_pts = circ_pts.copy()
        best_score = circ_score
    
    # Two concentric circles: 6 inner, 10 outer
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer_angles = np.linspace(0, 2*np.pi, 10, endpoint=False) + np.pi/10
    r_inner = 0.5
    r_outer = 1.0
    two_ring = np.vstack([
        r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)]),
        r_outer * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    ])
    two_ring_score = _score(two_ring)
    if two_ring_score < best_score:
        best_pts = two_ring.copy()
        best_score = two_ring_score
    
    # 4 inner (square), 12 outer
    inner_sq = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]], dtype=float)
    outer_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer_12 = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    four_twelve = np.vstack([inner_sq, outer_12])
    four_twelve_score = _score(four_twelve)
    if four_twelve_score < best_score:
        best_pts = four_twelve.copy()
        best_score = four_twelve_score
    
    # Local search using scipy's minimize
    def objective_flat(flat_pts):
        pts_2d = flat_pts.reshape(N, 2)
        return _objective(pts_2d)
    
    # Multiple restarts with different methods
    methods = ['BFGS', 'Nelder-Mead', 'Powell']
    random_starts = []
    
    # Add some random perturbations of best_pts
    np.random.seed(42)
    for _ in range(5):
        perturbed = best_pts + np.random.randn(N, 2) * 0.1
        random_starts.append(perturbed)
    
    # Also add the initial candidates
    random_starts.append(pts.copy())
    random_starts.append(sq_pts.copy())
    random_starts.append(two_ring.copy())
    random_starts.append(four_twelve.copy())
    
    for start_idx, init_pts in enumerate(random_starts):
        if time.time() > deadline - 5:
            break
        
        for method in methods:
            if time.time() > deadline - 5:
                break
            
            try:
                result = minimize(
                    objective_flat,
                    init_pts.ravel(),
                    method=method,
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if result.success or result.fun < best_score:
                    new_pts = result.x.reshape(N, 2)
                    new_score = _score(new_pts)
                    if new_score < best_score:
                        best_pts = new_pts.copy()
                        best_score = new_score
            except Exception:
                pass
    
    # Final refinement with smaller steps
    if time.time() < deadline - 3:
        try:
            result = minimize(
                objective_flat,
                best_pts.ravel(),
                method='BFGS',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < best_score:
                    best_pts = new_pts.copy()
                    best_score = new_score
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
