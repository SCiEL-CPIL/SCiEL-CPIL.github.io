"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles."""
    n = len(pts)
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            parts.append(0.5 * cross)
    return np.concatenate(parts)

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _convex_hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = np.min(_all_triangle_areas(pts))
    return min_area / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _fermat_spiral_init(n):
    """Generate initial points using Fermat spiral (sunflower pattern)."""
    angles = np.arange(n) * np.pi * (3 - np.sqrt(5))
    radii = np.sqrt(np.arange(n) / n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _regular_polygon_init(n, with_center=False):
    """Generate regular polygon initialization."""
    if with_center:
        m = n - 1
        angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
        ring = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([ring, center])
    else:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

def _simulated_annealing(pts, iterations=5000, t_start=0.05, t_end=0.001, seed=42):
    """
    Simulated annealing targeting the worst triangle.
    At each step, identify the triangle with minimum area, perturb one of its vertices,
    and accept if it improves the minimum area.
    """
    rng = np.random.RandomState(seed)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    for it in range(iterations):
        t = t_start * (t_end / t_start) ** (it / iterations)
        
        # Get current triangle areas to find the worst
        areas = _all_triangle_areas(current_pts)
        min_idx = np.argmin(areas)
        
        # Map flat index to triangle (i, j, k)
        n = len(current_pts)
        # Count triangles: for i in range(n), for j in range(i+1, n), for k in range(j+1, n)
        # We need to find which (i,j,k) corresponds to min_idx
        count = 0
        tri_i, tri_j, tri_k = 0, 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if count == min_idx:
                        tri_i, tri_j, tri_k = i, j, k
                        break
                    count += 1
                if count == min_idx:
                    break
            if count == min_idx:
                break
        
        # Perturb one of the three vertices of the worst triangle
        vertex_choice = rng.randint(0, 3)
        if vertex_choice == 0:
            idx = tri_i
        elif vertex_choice == 1:
            idx = tri_j
        else:
            idx = tri_k
        
        # Generate a perturbation
        step_size = t * 0.1
        new_pts = current_pts.copy()
        new_pts[idx] += rng.randn(2) * step_size
        
        new_score = _score(new_pts)
        
        # Accept if better or with probability exp((new_score - current_score) / t)
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            delta = new_score - current_score
            if delta < -10 * t:
                continue  # Too bad, reject
            prob = np.exp(delta / t)
            if rng.rand() < prob:
                current_pts = new_pts
                current_score = new_score
    
    return best_pts, best_score

def _local_refinement(pts, max_iter=2000):
    """Local refinement using Nelder-Mead on the exact score."""
    try:
        res = minimize(
            _neg_score,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        return res.x.reshape(-1, 2), _score(res.x.reshape(-1, 2))
    except Exception:
        return pts, _score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fermat spiral
    initializations.append(_fermat_spiral_init(N))
    
    # 2. Regular 13-gon
    initializations.append(_regular_polygon_init(N))
    
    # 3. 12-gon + center
    initializations.append(_regular_polygon_init(N, with_center=True))
    
    # 4. Two concentric rings (6 + 7)
    n1, n2 = 6, 7
    ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    r1, r2 = 0.5, 1.0
    ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    initializations.append(np.vstack([ring1, ring2]))
    
    # 5. Three concentric rings (4 + 5 + 4)
    n1, n2, n3 = 4, 5, 4
    ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n3
    r1, r2, r3 = 0.3, 0.6, 1.0
    ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
    initializations.append(np.vstack([ring1, ring2, ring3]))
    
    # 6. Random initialization
    rng = np.random.RandomState(123)
    initializations.append(rng.randn(N, 2) * 0.5)
    
    # 7. Another random
    rng2 = np.random.RandomState(456)
    initializations.append(rng2.randn(N, 2) * 0.5)
    
    # Run simulated annealing from each initialization
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Simulated annealing
        sa_pts, sa_score = _simulated_annealing(
            init_pts,
            iterations=3000,
            t_start=0.05,
            t_end=0.0005,
            seed=42 + init_idx
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
        
        # Local refinement from the SA result
        if time.time() > deadline - 15.0:
            continue
        
        refined_pts, refined_score = _local_refinement(sa_pts, max_iter=1000)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Final global refinement from the best result found
    if time.time() < deadline - 5.0 and best_score > 0:
        final_pts, final_score = _local_refinement(best_pts, max_iter=2000)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
