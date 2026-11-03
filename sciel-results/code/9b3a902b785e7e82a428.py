# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 11

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def project_to_triangle(p):
    if point_in_triangle(p):
        return p
    best = p
    best_dist = np.inf
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for a, b in edges:
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best = proj
    centroid = (V0 + V1 + V2) / 3.0
    best = 0.99 * best + 0.01 * centroid
    return best

def barycentric_coords(p, a=V0, b=V1, c=V2):
    def area2(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    A = area2(a, b, c)
    if np.abs(A) < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    u = area2(p, b, c) / A
    v = area2(a, p, c) / A
    w = area2(a, b, p) / A
    return np.array([u, v, w])

def min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
        
    # Generate indices for all triples (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    if not i_idx:
        return 0.0
        
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    points = x.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        u, v, w = barycentric_coords(p)
        if u < 0: penalty += u * u
        if v < 0: penalty += v * v
        if w < 0: penalty += w * w
    if penalty > 1e-8:
        return penalty * 1000.0
    return -min_triangle_area(points)

def generate_starts():
    starts = []
    
    # Create a dense grid of candidate points
    # Use a triangular grid to match the domain
    n_grid = 20
    candidates = []
    
    # Iterate over barycentric coordinates grid
    for u in np.linspace(0, 1, n_grid):
        for v in np.linspace(0, 1 - u, n_grid):
            w = 1 - u - v
            p = u * V0 + v * V1 + w * V2
            candidates.append(p)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Pre-compute a set of diverse index selections
    # We want to select 11 points from candidates for each start
    # Strategy: Use a "sliding window" or "shifted sampling" on the grid
    
    num_starts = 50
    # Shuffle candidates deterministically? No, keep order for reproducibility but select different subsets.
    # Actually, randomizing the order of candidates helps decorrelate starts if we pick first 11, next 11, etc.
    # But we want structured diversity.
    
    # Let's use a random permutation of indices for selection, seeded for reproducibility
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_cand)
    
    for s in range(num_starts):
        # Select a subset of 11 points
        # We can use a stride to ensure spread
        stride = max(1, n_cand // (N * 2))
        
        start_offset = (s * 7) % n_cand # Prime step for decorrelation
        
        indices = []
        idx = start_offset
        for _ in range(N):
            indices.append(idx % n_cand)
            idx += stride
            
        # Ensure uniqueness
        seen = set()
        unique_indices = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                unique_indices.append(i)
                
        if len(unique_indices) < N:
            # Fill with random other indices
            for i in perm:
                if i not in seen:
                    unique_indices.append(i)
                    seen.add(i)
                if len(unique_indices) == N:
                    break
                    
        points = candidates[unique_indices].copy()
        
        # Apply structured jitter
        # Jitter magnitude increases with start index to explore larger deviations
        base_mag = 0.02 + 0.01 * s
        for i in range(N):
            # Deterministic jitter based on start and point index
            angle = 2 * np.pi * (s * 13 + i * 7) / 100.0
            r = base_mag * (0.5 + 0.5 * np.sin(s * 0.7 + i * 0.3))
            points[i] += r * np.array([np.cos(angle), np.sin(angle)])
            points[i] = project_to_triangle(points[i])
            
        starts.append(points.flatten())
        
    return starts

def construct():
    starts = generate_starts()
    best_x = None
    best_val = -np.inf
    
    for x0 in starts:
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            x_opt = res.x
            points = x_opt.reshape(-1, 2)
            
            # Final cleanup
            for i in range(len(points)):
                points[i] = project_to_triangle(points[i])
                
            val = min_triangle_area(points)
            if val > best_val:
                best_val = val
                best_x = points.copy()
        except Exception:
            continue
            
    return best_x
# EVOLVE-BLOCK-END
