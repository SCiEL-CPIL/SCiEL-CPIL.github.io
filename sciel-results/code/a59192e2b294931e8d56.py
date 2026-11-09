"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
TRI_AREA = H / 2.0  # area of unit equilateral triangle

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    return np.min(areas)


def _is_inside(points):
    """Check if all points are inside the equilateral triangle."""
    # For point (x,y) in triangle with vertices (0,0), (1,0), (0.5, H):
    # y >= 0, y <= H, and for the left edge: x >= y/H (line from (0,0) to (0.5,H): x = y/(2H)*... wait)
    # Actually: left edge from (0,0) to (0.5, H): parametric x = 0.5*t, y = H*t => x = y/(2H)
    # Right edge from (1,0) to (0.5, H): x = 1 - 0.5*t, y = H*t => x = 1 - y/(2H)
    # So: 0 <= y <= H, y/(2H) <= x <= 1 - y/(2H)
    x = points[:, 0]
    y = points[:, 1]
    eps = 1e-12
    return np.all(y >= -eps) and np.all(y <= H + eps) and \
           np.all(x >= y / (2 * H) - eps) and np.all(x <= 1 - y / (2 * H) + eps)


def _soft_min(areas, beta=50.0):
    """Soft minimum using log-sum-exp trick: -log(sum(exp(-beta*a)))/beta approximates min(a)."""
    # Use the identity: min(a) ≈ -(1/beta) * log(sum(exp(-beta * a)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return -log_sum / beta


def _objective(params, beta=50.0):
    """Negative soft-min area (to minimize). params: (N*2,) flattened."""
    points = params.reshape(N, 2)
    if not _is_inside(points):
        # Heavy penalty for being outside
        x = points[:, 0]
        y = points[:, 1]
        penalty = 0.0
        penalty += np.sum(np.maximum(0, -y)) * 10
        penalty += np.sum(np.maximum(0, y - H)) * 10
        penalty += np.sum(np.maximum(0, y / (2 * H) - x)) * 10
        penalty += np.sum(np.maximum(0, x - (1 - y / (2 * H)))) * 10
        return penalty + 100.0
    
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    return -_soft_min(areas, beta)


def _refine(points, deadline, max_iter=500):
    """Local refinement: move the vertices of the worst triangle to improve min area."""
    best_points = points.copy()
    best_area = _min_area(best_points)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find worst triangle
        p1 = best_points[_idx_i]
        p2 = best_points[_idx_j]
        p3 = best_points[_idx_k]
        cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                       (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
        areas = 0.5 * cross
        worst_idx = np.argmin(areas)
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        current_worst_area = areas[worst_idx]
        
        # Try moving each of the three vertices in small steps
        improved = False
        for vi in [i, j, k]:
            for dx in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                for dy in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                    new_points = best_points.copy()
                    new_points[vi, 0] += dx
                    new_points[vi, 1] += dy
                    if not _is_inside(new_points):
                        continue
                    new_area = _min_area(new_points)
                    if new_area > best_area + 1e-12:
                        best_points = new_points
                        best_area = new_area
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try larger steps
            for vi in [i, j, k]:
                for dx in [-0.05, -0.03, 0.03, 0.05]:
                    for dy in [-0.05, -0.03, 0.03, 0.05]:
                        new_points = best_points.copy()
                        new_points[vi, 0] += dx
                        new_points[vi, 1] += dy
                        if not _is_inside(new_points):
                            continue
                        new_area = _min_area(new_points)
                        if new_area > best_area + 1e-12:
                            best_points = new_points
                            best_area = new_area
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_points, best_area


def _lattice_initial():
    """Generate a good initial configuration using lattice-based placement."""
    # Use a structured grid in barycentric coordinates
    # Place points at strategic positions: corners, edge midpoints, center, etc.
    # For 11 points, try a mix of known good positions
    
    # Start with a symmetric configuration
    points = np.array([
        [0.0, 0.0],      # corner A
        [1.0, 0.0],      # corner B
        [0.5, H],        # corner C
        [0.5, 0.0],      # midpoint of AB
        [0.25, H/2],     # midpoint of AC
        [0.75, H/2],     # midpoint of BC
        [0.5, H/3],      # centroid
        [1/3, 0.0],      # trisection of AB
        [2/3, 0.0],      # trisection of AB
        [0.25, H/4],     # interior
        [0.75, H/4],     # interior
    ])
    
    # Verify all inside
    if not _is_inside(points):
        # Adjust if needed
        points[0] = [0.05, 0.05]
        points[1] = [0.95, 0.05]
        points[2] = [0.5, H - 0.05]
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with lattice-based initial configuration
    points = _lattice_initial()
    best_points, best_area = _refine(points, deadline, max_iter=200)
    
    # Try multiple random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Generate random initial points inside triangle
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            # Barycentric to Cartesian
            x = v * 0.5 + (1 - u - v) * 0.0 + u * 1.0
            y = v * H
            # Actually: A=(0,0), B=(1,0), C=(0.5,H)
            # Point = (1-u-v)*A + u*B + v*C = u*(1,0) + v*(0.5,H)
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
        pts = np.array(pts)
        
        if not _is_inside(pts):
            continue
        
        # Refine this random configuration
        refined_pts, refined_area = _refine(pts, deadline, max_iter=100)
        
        if refined_area > best_area:
            best_points = refined_pts
            best_area = refined_area
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        best_points, best_area = _refine(best_points, deadline, max_iter=1000)
    
    return best_points
