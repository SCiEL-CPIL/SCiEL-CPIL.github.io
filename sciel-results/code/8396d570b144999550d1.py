"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # There are C(13,3) = 286 triangles. We can compute all at once using broadcasting.
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_tri = min_triangle_area(points)
    return min_tri / hull_area


def construct():
    entry_time = time.time()
    deadline = entry_time + (DEADLINE - SAFETY_MARGIN)
    
    rng = np.random.default_rng(42)
    best_score = 0.0
    best_points = None
    
    # Initial candidates
    # Regular heptagon + 6 interior points on symmetry axes? 
    # Let's try a regular polygon first as baseline.
    theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Try placing remaining 6 points in a regular hexagon scaled down and rotated
    # Or just start with a 13-gon.
    theta13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    poly13 = np.column_stack([np.cos(theta13), np.sin(theta13)])
    s = score(poly13)
    if s > best_score:
        best_score = s
        best_points = poly13.copy()
        
    # Random initial points
    for _ in range(50):
        pts = rng.random((N, 2))
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    # Normalize
    def normalize(pts):
        pts = pts - pts.mean(axis=0)
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
        return pts
        
    best_points = normalize(best_points)
    best_score = score(best_points)
    
    # Structural Change: Symmetric Parameterization + Grid Search + Local Refine
    # We hypothesize that optimal configurations for small N have high symmetry.
    # Let's try a configuration of:
    # - 1 center point
    # - 6 points on a regular hexagon (radius r1, angle offset a1)
    # - 6 points on another regular hexagon (radius r2, angle offset a2)
    # This gives 1 + 6 + 6 = 13 points.
    # We can also vary the symmetry group:
    # Option A: C6 symmetry (center + 2 hexagons)
    # Option B: C7 symmetry (not possible with 13 = 1 + 6 + 6, but 13 is prime. 
    #   Maybe 7 outer + 6 inner? 7+6=13. 
    #   Let's try 7 outer regular heptagon + 6 inner regular hexagon + 1 center? No, 7+6+1=14.
    #   7 outer + 6 inner = 13. No center.
    #   Let's try: 7 points on heptagon, 6 points on hexagon.
    
    def make_config_A(r1, r2, a1, a2):
        # Center
        center = np.array([[0.0, 0.0]])
        # Hexagon 1
        theta1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a1
        hex1 = r1 * np.column_stack([np.cos(theta1), np.sin(theta1)])
        # Hexagon 2
        theta2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a2
        hex2 = r2 * np.column_stack([np.cos(theta2), np.sin(theta2)])
        return np.vstack([center, hex1, hex2])
        
    def make_config_B(r1, r2, a1, a2):
        # 7 points heptagon
        theta1 = np.linspace(0, 2*np.pi, 7, endpoint=False) + a1
        hept1 = r1 * np.column_stack([np.cos(theta1), np.sin(theta1)])
        # 6 points hexagon
        theta2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a2
        hex2 = r2 * np.column_stack([np.cos(theta2), np.sin(theta2)])
        return np.vstack([hept1, hex2])
        
    def make_config_C(r1, r2, r3, a1, a2, a3):
        # 1 center, 3 points triangle, 4 points square, 5 points pentagon? 1+3+4+5=13.
        center = np.array([[0.0, 0.0]])
        th1 = np.linspace(0, 2*np.pi, 3, endpoint=False) + a1
        p1 = r1 * np.column_stack([np.cos(th1), np.sin(th1)])
        th2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + a2
        p2 = r2 * np.column_stack([np.cos(th2), np.sin(th2)])
        th3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + a3
        p3 = r3 * np.column_stack([np.cos(th3), np.sin(th3)])
        return np.vstack([center, p1, p2, p3])
        
    # Grid search over these symmetric families
    # Config A: 4 params
    # Config B: 4 params
    # Config C: 6 params
    
    # We will do a coarse grid search, then a local optimization (Nelder-Mead) on the best few.
    
    candidates = []
    
    # Coarse grid for Config A (1 + 6 + 6)
    r1_vals = np.linspace(0.1, 0.8, 8)
    r2_vals = np.linspace(0.1, 0.8, 8)
    a1_vals = np.linspace(0, np.pi/6, 6)
    a2_vals = np.linspace(0, np.pi/6, 6)
    
    # To save time, we only grid search a subset and then refine.
    # Actually, 8*8*6*6 = 2304 evaluations. Each eval is fast (<1ms). 
    # 2304 * 1ms = 2.3s. This is very fast.
    
    for r1 in r1_vals:
        if time.time() > deadline: break
        for r2 in r2_vals:
            if time.time() > deadline: break
            for a1 in a1_vals:
                if time.time() > deadline: break
                for a2 in a2_vals:
                    if time.time() > deadline: break
                    pts = make_config_A(r1, r2, a1, a2)
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
                    # Keep top few for refinement? Or just track best.
                    # Let's collect all scores and points for a global best.
                    
    # Coarse grid for Config B (7 + 6)
    for r1 in r1_vals:
        if time.time() > deadline: break
        for r2 in r2_vals:
            if time.time() > deadline: break
            for a1 in a1_vals:
                if time.time() > deadline: break
                for a2 in a2_vals:
                    if time.time() > deadline: break
                    pts = make_config_B(r1, r2, a1, a2)
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
                        
    # Coarse grid for Config C (1 + 3 + 4 + 5)
    # 6 params, 4^6 = 4096 if 4 steps. 4096 * 1ms = 4s.
    r_vals_c = np.linspace(0.1, 0.9, 4)
    a_vals_c = np.linspace(0, np.pi/3, 4)
    for r1 in r_vals_c:
        if time.time() > deadline: break
        for r2 in r_vals_c:
            if time.time() > deadline: break
            for r3 in r_vals_c:
                if time.time() > deadline: break
                for a1 in a_vals_c:
                    if time.time() > deadline: break
                    for a2 in a_vals_c:
                        if time.time() > deadline: break
                        for a3 in a_vals_c:
                            if time.time() > deadline: break
                            pts = make_config_C(r1, r2, r3, a1, a2, a3)
                            s = score(pts)
                            if s > best_score:
                                best_score = s
                                best_points = pts.copy()
                                
    # Now, refine the best point found using Nelder-Mead on the full 26-dimensional space?
    # Or refine the symmetric parameters?
    # The symmetric parameterization reduces dimensionality.
    # Let's try to refine the best symmetric configuration found.
    # We need to know which config type gave the best score.
    # This is a bit messy. Let's just do a generic local search from the best_points
    # but with a smaller step size, or use Nelder-Mead on the 13 points directly?
    # Nelder-Mead on 26 variables is possible.
    
    from scipy.optimize import minimize
    
    def neg_score(pts_flat):
        pts = pts_flat.reshape(13, 2)
        return -score(pts)
        
    # Start from the best symmetric point
    x0 = best_points.flatten()
    
    # We have time left?
    if time.time() < deadline - 20:
        try:
            # Nelder-Mead on the full point set
            # This might break symmetry, which is good if it helps.
            # But it might get stuck in local minima.
            # Let's try a few restarts of NM from slightly perturbed best_points.
            
            for _ in range(10):
                if time.time() > deadline - 10: break
                x0_pert = x0 + rng.normal(0, 0.01, (26,))
                res = minimize(neg_score, x0_pert, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success:
                    pts_opt = res.x.reshape(13, 2)
                    s_opt = -res.fun
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt.copy()
        except Exception:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
