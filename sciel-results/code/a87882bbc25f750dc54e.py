"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def _inside_triangle(p):
    """Check if point p is inside the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    # Barycentric coordinates
    # Vertex A=(0,0), B=(1,0), C=(0.5, h)
    # Area of ABC = 0.5 * 1 * h = h/2
    # Area of PBC = 0.5 * |det(B-P, C-P)| ... but simpler:
    # y >= 0
    # y <= sqrt(3) * x  (line from A to C: y = sqrt(3) * x)
    # y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)  (line from B to C)
    if y < -1e-9:
        return False
    if y > h + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |det(p2-p1, p3-p1)|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_pairwise_distance(points):
    """Compute the minimum pairwise distance."""
    n = len(points)
    min_d = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_d:
                min_d = d
    return min_d


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    
    # Clamp to triangle boundaries
    # y >= 0
    if y < 0:
        y = 0
    # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    if y > np.sqrt(3) * x:
        # Project onto line y = sqrt(3) * x
        # The closest point on the line y = sqrt(3)*x to (x,y)
        # Direction of line: (1, sqrt(3))
        # Normal: (sqrt(3), -1) / 2
        # Actually, let's just solve for the closest point on the segment from (0,0) to (0.5, h)
        # Parametric: t * (0.5, h) for t in [0,1]
        # Minimize |p - t*v|^2 where v = (0.5, h)
        v = np.array([0.5, h])
        t = np.dot(p, v) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        proj = t * v
        x, y = proj
    # y <= sqrt(3) * (1 - x)  =>  x >= 1 - y / sqrt(3)
    if y > np.sqrt(3) * (1 - x):
        # Project onto line from (1,0) to (0.5, h)
        # Direction: (-0.5, h)
        # Point on line: (1,0) + s * (-0.5, h) for s in [0,1]
        v = np.array([-0.5, h])
        p_rel = np.array([x - 1, y])
        s = np.dot(p_rel, v) / np.dot(v, v)
        s = np.clip(s, 0, 1)
        proj = np.array([1, 0]) + s * v
        x, y = proj
    
    return np.array([x, y])


def _objective(x_flat, points):
    """Objective function: negative of minimum triangle area, with penalty for points outside triangle."""
    n = len(points)
    pts = x_flat.reshape(n, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(n):
        if not _inside_triangle(pts[i]):
            # Distance to triangle
            proj = _project_to_triangle(pts[i])
            penalty += 10.0 * np.linalg.norm(pts[i] - proj) ** 2
    
    min_area = _min_triangle_area(pts)
    return penalty - min_area


def construct():
    """Construct 11 points in the unit equilateral triangle to maximize minimum triangle area."""
    h = np.sqrt(3) / 2
    
    # Step 1: Lattice search over barycentric coordinates
    # Use a grid of barycentric coordinates (a, b, c) with a+b+c=1, a,b,c >= 0
    # Map to Cartesian: P = a*(0,0) + b*(1,0) + c*(0.5, h) = (b + 0.5*c, h*c)
    
    # Try different grid resolutions
    best_points = None
    best_min_area = 0.0
    
    # Coarse to fine lattice search
    for res in [3, 4, 5, 6]:
        # Generate all valid barycentric triples with a+b+c=1
        # a, b, c are multiples of 1/res
        candidates = []
        for a_num in range(res + 1):
            for b_num in range(res + 1 - a_num):
                c_num = res - a_num - b_num
                a = a_num / res
                b = b_num / res
                c = c_num / res
                x = b + 0.5 * c
                y = h * c
                candidates.append((x, y))
        
        # We need to choose N points from these candidates
        # Use a greedy approach: pick points that maximize the minimum distance to already-picked points
        # Then evaluate min triangle area
        
        # For small candidate sets, we can try all combinations if feasible
        n_cand = len(candidates)
        
        if n_cand <= 20:
            # Try all combinations
            for combo in combinations(range(n_cand), N):
                pts = np.array([candidates[i] for i in combo])
                min_area = _min_triangle_area(pts)
                if min_area > best_min_area:
                    best_min_area = min_area
                    best_points = pts.copy()
        else:
            # Greedy packing: pick points to maximize minimum pairwise distance
            # Then refine with local optimization
            for _ in range(10):  # Multiple random restarts
                picked = []
                remaining = list(range(n_cand))
                
                # Start with a random point
                start_idx = np.random.choice(remaining)
                picked.append(start_idx)
                remaining.remove(start_idx)
                
                while len(picked) < N:
                    # Find the point in remaining that maximizes the minimum distance to picked
                    best_idx = None
                    best_min_dist = -1
                    for idx in remaining:
                        min_dist = min(np.linalg.norm(np.array(candidates[idx]) - np.array(candidates[p])) for p in picked)
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = idx
                    picked.append(best_idx)
                    remaining.remove(best_idx)
                
                pts = np.array([candidates[i] for i in picked])
                min_area = _min_triangle_area(pts)
                if min_area > best_min_area:
                    best_min_area = min_area
                    best_points = pts.copy()
    
    if best_points is None:
        # Fallback: random points
        points = []
        for _ in range(N):
            while True:
                u, v = np.random.random(), np.random.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = h * v
                    points.append([x, y])
                    break
        return np.array(points)
    
    # Step 2: Local polish using scipy.optimize
    # Maximize minimum triangle area by minimizing negative min area
    x0 = best_points.flatten()
    
    def neg_min_area(x_flat):
        pts = x_flat.reshape(N, 2)
        # Project each point to triangle to ensure feasibility
        pts_clipped = np.array([_project_to_triangle(p) for p in pts])
        return -_min_triangle_area(pts_clipped)
    
    # Use Nelder-Mead for local optimization
    result = minimize(neg_min_area, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.success or -result.fun > best_min_area:
        best_points = result.x.reshape(N, 2)
        # Final clip to triangle
        best_points = np.array([_project_to_triangle(p) for p in best_points])
    
    return best_points

# EVOLVE-BLOCK-END
