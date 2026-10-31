import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 16
D = 2

def _pairwise_distances(pts):
    """Compute all pairwise distances efficiently."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    return dists[iu]

def _objective(pts):
    """Objective: (max_dist / min_dist)^2"""
    d = _pairwise_distances(pts)
    d_min = d.min()
    d_max = d.max()
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min) ** 2

def _make_hexagonal(n=16):
    """Create a hexagonal lattice arrangement."""
    pts = []
    # Hexagonal lattice: rows offset
    # Try different configurations
    # For 16 points, try 4 rows with 4,4,4,4 but shifted
    # Or 3,4,4,5 or similar
    configs = [
        [4,4,4,4],
        [3,4,4,5],
        [5,4,4,3],
        [4,5,4,3],
        [3,5,4,4],
        [2,4,4,6],
        [6,4,4,2],
        [4,6,4,2],
        [2,6,4,4],
        [4,4,5,3],
        [3,5,4,4],
        [5,3,4,4],
        [4,4,3,5],
        [5,4,3,4],
        [4,3,5,4],
        [3,4,5,4],
    ]
    best_pts = None
    best_score = float('inf')
    
    for config in configs:
        if sum(config) != n:
            continue
        pts = []
        y_spacing = 1.0
        x_spacing = 1.0
        for i, count in enumerate(config):
            y = i * y_spacing * np.sqrt(3)/2
            offset = (i % 2) * x_spacing / 2
            for j in range(count):
                x = j * x_spacing + offset
                pts.append([x, y])
        pts = np.array(pts)
        score = _objective(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: square grid
        side = int(np.ceil(n ** 0.5))
        grid = list(itertools.product(range(side), repeat=2))[:n]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts

def _make_circular(n=16):
    """Create a circular arrangement with rings."""
    # Try different ring configurations
    best_pts = None
    best_score = float('inf')
    
    # Ring configs: number of points per ring (inner to outer)
    configs = [
        [1, 5, 10],
        [1, 6, 9],
        [2, 6, 8],
        [1, 7, 8],
        [1, 8, 7],
        [2, 5, 9],
        [1, 4, 11],
        [1, 5, 10],
        [1, 9, 6],
        [2, 4, 10],
        [1, 3, 12],
        [1, 10, 5],
        [2, 7, 7],
        [3, 5, 8],
        [1, 6, 9],
        [4, 4, 8],
        [2, 6, 8],
        [1, 8, 7],
    ]
    
    for config in configs:
        if sum(config) != n:
            continue
        pts = []
        for ring_idx, count in enumerate(config):
            if count == 0:
                continue
            if ring_idx == 0:
                # Inner ring: place at center or small radius
                if count == 1:
                    pts.append([0, 0])
                else:
                    r = 0.5
                    angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                    for a in angles:
                        pts.append([r*np.cos(a), r*np.sin(a)])
            else:
                # Outer rings
                r = ring_idx * 1.0
                angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                for a in angles:
                    pts.append([r*np.cos(a), r*np.sin(a)])
        
        pts = np.array(pts)
        score = _objective(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

def _make_square_grid(n=16):
    """Create a square grid arrangement."""
    side = int(np.ceil(n ** 0.5))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) < n:
                pts.append([i, j])
    return np.array(pts, dtype=float)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate initial guesses
    candidates = [
        _make_hexagonal(N),
        _make_circular(N),
        _make_square_grid(N),
    ]
    
    # Also add some perturbed versions
    np.random.seed(42)
    for _ in range(5):
        base = np.random.randn(N, 2) * 2
        candidates.append(base)
    
    # Find the best initial candidate
    best_pts = None
    best_score = float('inf')
    for cand in candidates:
        score = _objective(cand)
        if score < best_score:
            best_score = score
            best_pts = cand.copy()
    
    # Refine with optimization
    if time.time() < deadline - 10:
        # Use Nelder-Mead for refinement
        result = minimize(
            _objective,
            best_pts,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_pts = result.x.copy()
    
    # Try more starts if time permits
    if time.time() < deadline - 30:
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            # Random perturbation of best solution
            perturbed = best_pts + np.random.randn(N, 2) * 0.1
            result = minimize(
                _objective,
                perturbed,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_pts = result.x.copy()
    
    # Final refinement
    if time.time() < deadline - 5:
        result = minimize(
            _objective,
            best_pts,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_pts = result.x.copy()
    
    return best_pts
