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
    # Pull slightly towards center to ensure strict interior for optimization stability
    centroid = (V0 + V1 + V2) / 3.0
    best = 0.995 * best + 0.005 * centroid
    return best

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    points = x.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        # Fast barycentric check
        w = p[1] / H
        v = p[0] - 0.5 * w
        u = 1.0 - v - w
        if u < 0: penalty += u * u
        if v < 0: penalty += v * v
        if w < 0: penalty += w * w
        
    if penalty > 1e-8:
        return penalty * 10000.0
    return -min_triangle_area(points)

def generate_starts():
    starts = []
    
    # 1. Symmetric Starts
    # Configuration A: 1 center, 3 on altitudes, 3 on midlines, etc.
    # Let's just use a few hardcoded symmetric-like patterns
    
    # Pattern 1: Vertices + Centroid + 7 points on a hexagonal grid clipped
    # Pattern 2: 11 points on a coarse grid
    
    # Generate candidates from a fine barycentric grid
    num_grid = 15
    candidates = []
    for i in range(num_grid):
        for j in range(num_grid - i):
            k = num_grid - 1 - i - j
            u = i / (num_grid - 1)
            v = j / (num_grid - 1)
            w = k / (num_grid - 1)
            p = u * V0 + v * V1 + w * V2
            candidates.append(p)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Greedy selection function
    def greedy_select(cands, n_sel, seed):
        np.random.seed(seed)
        start_idx = np.random.randint(n_cand)
        selected = [cands[start_idx].copy()]
        remaining_indices = [i for i in range(n_cand) if i != start_idx]
        
        for _ in range(n_sel - 1):
            if not remaining_indices:
                break
            # Vectorized min distance to selected
            # dists shape (n_rem, n_sel)
            rem_cands = cands[remaining_indices]
            dists = np.linalg.norm(rem_cands[:, np.newaxis, :] - np.array(selected)[np.newaxis, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            max_idx_local = np.argmax(min_dists)
            max_idx_global = remaining_indices[max_idx_local]
            selected.append(cands[max_idx_global].copy())
            remaining_indices.remove(max_idx_global)
            
        return np.array(selected)

    # Generate 50 diverse starts
    num_starts = 50
    for s in range(num_starts):
        # Mix of greedy grid selection and random jittering
        
        if s < 10:
            # Pure greedy grid selection with different seeds
            points = greedy_select(candidates, N, 100 + s)
        elif s < 20:
            # Random points in triangle, then greedy select from them? 
            # Or just random points projected
            np.random.seed(200 + s)
            # Sample barycentric coords
            r1 = np.random.rand(N)
            r2 = np.random.rand(N)
            u = 1 - np.sqrt(r1)
            v = r2 * (1 - u)
            w = 1 - u - v
            points = np.array([u[i]*V0 + v[i]*V1 + w[i]*V2 for i in range(N)])
            # Refine by greedy selection from a local grid? 
            # Just use these as start, they are random
        else:
            # Structured Jitter: Take a good symmetric base and jitter
            # Base: 1 center, 3 on medians, 3 on midlines, 3 on vertices? 
            # Let's use the result of a previous greedy run as base and jitter it
            base = greedy_select(candidates, N, 300 + (s % 10))
            points = base.copy()
            
            # Add deterministic jitter
            for i in range(N):
                angle = 2 * np.pi * (s * 13 + i * 7) / 11.0
                r = 0.01 * (0.5 + 0.5 * np.sin(s * 0.5 + i * 0.2))
                points[i] += r * np.array([np.cos(angle), np.sin(angle)])
                points[i] = project_to_triangle(points[i])

        # Ensure all points are strictly inside
        points = np.array([project_to_triangle(p) for p in points])
        
        # Flatten
        starts.append(points.flatten())
        
    return starts

def construct():
    starts = generate_starts()
    best_x = None
    best_val = -np.inf
    
    for x0 in starts:
        try:
            # Stage 1: Coarse optimization
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            x_opt = res.x
            points = x_opt.reshape(-1, 2)
            
            # Stage 2: Fine polishing
            res2 = minimize(
                objective, 
                x_opt, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            x_opt2 = res2.x
            points = x_opt2.reshape(-1, 2)
            
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
