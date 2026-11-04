import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _score_ring(r_in, angle_offset):
    """
    Score for a specific ring configuration:
    7 points on unit circle, 7 points on circle of radius r_in,
    with angle_offset for the inner ring.
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = angle_offset + np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r_in * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return _score(pts), pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    best_score = 0
    best_points = None
    current_points = None
    current_score = 0
    
    # 1. Explore the 7+7 ring structure systematically
    # We know 7+7 is likely optimal. Let's search over r_in and angle_offset.
    # Start with a coarse grid.
    r_in_values = np.linspace(0.2, 0.9, 8)
    angle_offsets = np.linspace(0, np.pi/7, 8) # The offset is periodic with pi/7? No, 2*pi/7.
    # Actually, the symmetry suggests the offset is related to pi/7.
    # Let's try a grid of offsets in [0, pi/7] and radii.
    
    for r_in in r_in_values:
        for offset in angle_offsets:
            s, pts = _score_ring(r_in, offset)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                current_points = pts.copy()
                current_score = s
    
    # 2. Refine the best ring configuration with Nelder-Mead on (r_in, offset)
    if best_points is not None:
        # Extract approximate r_in and offset from best_points
        # This is tricky if points are mixed. Let's just use the last tried parameters or re-optimize.
        # Better: run Nelder-Mead on the full 28D point space, but constrained to the ring structure?
        # No, let's just run Nelder-Mead on the full 28D space starting from the best ring candidate.
        
        def objective(x):
            if time.time() > deadline - 2:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return _neg_score(p)
            
        try:
            # Use a tight time limit for the first optimization
            time_limit = deadline - 20
            res = minimize(objective, current_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 5000})
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(N, 2)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts
                    current_points = new_pts
                    current_score = s
        except TimeoutError:
            pass
        except Exception:
            pass

    # 3. If time remains, try more random restarts with quick local search
    # Or refine the ring parameters further.
    # Let's try a second round of Nelder-Mead if time permits, or a different initial guess.
    
    # Alternative initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    s_reg = _score(init_reg)
    if s_reg > best_score:
        best_score = s_reg
        best_points = init_reg.copy()
        current_points = init_reg.copy()
        current_score = s_reg

    if time.time() < deadline - 10:
        def objective2(x):
            if time.time() > deadline - 2:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return _neg_score(p)
            
        try:
            time_limit = deadline - 2
            res = minimize(objective2, current_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 10000})
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(N, 2)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts
        except TimeoutError:
            pass
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial configurations
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(init1)
    
    # 2. Two concentric heptagons (interleaved)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False)
    
    for r_in in [0.3, 0.5, 0.7, 0.8]:
        inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
        outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        candidates.append(np.vstack([inner_pts, outer_pts]))
        
    # 3. One center, 13 on circle
    angles_13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    outer_13 = np.column_stack([np.cos(angles_13), np.sin(angles_13)])
    center = np.array([[0.0, 0.0]])
    candidates.append(np.vstack([center, outer_13]))
    
    # 4. Random points in unit disk
    np.random.seed(42)
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        candidates.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 5. Perturbed regular polygon
    for _ in range(3):
        pert = np.random.normal(0, 0.1, (N, 2))
        candidates.append(init1 + pert)

    best_score = 0
    best_points = None
    current_points = None
    current_score = 0
    
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            current_points = c.copy()
            current_score = s
            
    if best_points is None:
        return init1

    # Nelder-Mead Optimization
    def run_nm(pts, time_limit):
        start = time.time()
        def objective(x):
            if time.time() > time_limit:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return -_score(p)
            
        try:
            res = minimize(objective, pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(N, 2)
                return new_pts
        except TimeoutError:
            pass
        except Exception:
            pass
        return None

    if time.time() < deadline - 5:
        time_limit = deadline - 2
        res_pts = run_nm(current_points, time_limit)
        if res_pts is not None:
            s = _score(res_pts)
            if s > best_score:
                best_score = s
                best_points = res_pts
                current_points = res_pts
                current_score = s

    while time.time() < deadline - 10:
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        cand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        time_limit = deadline - 5
        res_pts = run_nm(cand, time_limit)
        
        if res_pts is not None:
            s = _score(res_pts)
            if s > best_score:
                best_score = s
                best_points = res_pts
                current_points = res_pts
                current_score = s

    return best_points
# EVOLVE-BLOCK-END
