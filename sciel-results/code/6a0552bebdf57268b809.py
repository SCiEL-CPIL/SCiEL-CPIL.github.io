"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the n points. Vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    # Compute all cross products for triples
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    # Barycentric coordinates
    v0 = V2 - V0
    v1 = V1 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return u >= -eps and v >= -eps and (u + v) <= (1.0 + eps)


def _project_to_triangle(p):
    """Project point p onto the interior of the equilateral triangle."""
    if _is_inside_triangle(p):
        return p
    # Find closest point on triangle boundary
    best_dist = float('inf')
    best_pt = p
    # Check edges
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for a, b in edges:
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        d = np.linalg.norm(p - proj)
        if d < best_dist:
            best_dist = d
            best_pt = proj
    # Move slightly inward
    centroid = (V0 + V1 + V2) / 3.0
    best_pt = best_pt * 0.999 + centroid * 0.001
    return best_pt


def _structured_init(seed=0):
    """Create a structured initial configuration based on symmetric positions."""
    rng = random.Random(seed)
    # Start with vertices and midpoints and centroid
    mid01 = (V0 + V1) / 2.0
    mid12 = (V1 + V2) / 2.0
    mid20 = (V2 + V0) / 2.0
    centroid = (V0 + V1 + V2) / 3.0
    
    # 11 points: 3 vertices, 3 midpoints, centroid, plus 4 more
    # Add points at 1/3 and 2/3 along edges, and some interior points
    p13_01 = V0 + (1.0/3.0) * (V1 - V0)
    p23_01 = V0 + (2.0/3.0) * (V1 - V0)
    p13_12 = V1 + (1.0/3.0) * (V2 - V1)
    p23_12 = V1 + (2.0/3.0) * (V2 - V1)
    p13_20 = V2 + (1.0/3.0) * (V0 - V2)
    p23_20 = V2 + (2.0/3.0) * (V0 - V2)
    
    # Interior points: trisection of medians
    # Median from V0 to mid12
    med0_1 = V0 + (1.0/3.0) * (mid12 - V0)
    med0_2 = V0 + (2.0/3.0) * (mid12 - V0)
    # Median from V1 to mid20
    med1_1 = V1 + (1.0/3.0) * (mid20 - V1)
    med1_2 = V1 + (2.0/3.0) * (mid20 - V1)
    # Median from V2 to mid01
    med2_1 = V2 + (1.0/3.0) * (mid01 - V2)
    med2_2 = V2 + (2.0/3.0) * (mid01 - V2)
    
    # Pick 11 from these candidates
    candidates = [V0, V1, V2, mid01, mid12, mid20, centroid,
                  p13_01, p23_01, p13_12, p23_12, p13_20, p23_20,
                  med0_1, med0_2, med1_1, med1_2, med2_1, med2_2]
    
    # Select 11 with some randomness
    # Always include vertices and centroid
    fixed = [V0, V1, V2, centroid]
    remaining = [c for c in candidates if c not in fixed]
    rng.shuffle(remaining)
    selected = fixed + remaining[:7]
    
    # Add small random perturbation to break symmetry
    points = []
    for p in selected:
        perturbed = p + np.array([rng.gauss(0, 0.01), rng.gauss(0, 0.01)])
        points.append(_project_to_triangle(perturbed))
    
    return np.array(points)


def _random_init(seed=0):
    """Create a random initial configuration inside the triangle."""
    rng = random.Random(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            # Add small margin from edges
            points.append(p)
    return np.array(points)


def _local_optimize(points, max_iters=2000, step_size=0.05, time_limit=None, start_time=None):
    """Coordinate descent with adaptive step sizes to maximize min triangle area."""
    points = points.copy()
    best_score = _min_triangle_area(points)
    current_score = best_score
    
    if start_time is None:
        start_time = time.time()
    
    it = 0
    while it < max_iters:
        if time_limit is not None and (time.time() - start_time) > time_limit - 5:
            break
        
        improved = False
        for i in range(N):
            # Try moving point i in small steps
            best_pos = points[i].copy()
            best_local = current_score
            
            # Try several step sizes
            for ss in [step_size, step_size * 0.5, step_size * 0.25, step_size * 2.0]:
                for dx in [-ss, 0.0, ss]:
                    for dy in [-ss, 0.0, ss]:
                        if dx == 0 and dy == 0:
                            continue
                        new_p = points[i].copy()
                        new_p[0] += dx
                        new_p[1] += dy
                        new_p = _project_to_triangle(new_p)
                        
                        old_p = points[i]
                        points[i] = new_p
                        score = _min_triangle_area(points)
                        if score > best_local + 1e-12:
                            best_local = score
                            best_pos = new_p.copy()
                        points[i] = old_p
            
            if np.any(best_pos != points[i]):
                points[i] = best_pos
                current_score = _min_triangle_area(points)
                if current_score > best_score:
                    best_score = current_score
                improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
        
        it += 1
    
    return points, best_score


def _simulated_annealing(points, max_iters=5000, time_limit=None, start_time=None):
    """Simulated annealing to escape local optima."""
    points = points.copy()
    best_points = points.copy()
    best_score = _min_triangle_area(points)
    current_score = best_score
    
    T = 0.1  # initial temperature
    T_min = 1e-8
    cooling = 0.995
    
    if start_time is None:
        start_time = time.time()
    
    rng = random.Random(42)
    
    it = 0
    while it < max_iters:
        if time_limit is not None and (time.time() - start_time) > time_limit - 5:
            break
        
        # Pick a random point to move
        i = rng.randint(0, N - 1)
        old_p = points[i].copy()
        
        # Random perturbation
        scale = 0.05 * (T / 0.1) + 0.001
        new_p = old_p + np.array([rng.gauss(0, scale), rng.gauss(0, scale)])
        new_p = _project_to_triangle(new_p)
        
        points[i] = new_p
        new_score = _min_triangle_area(points)
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[i] = old_p
        
        T *= cooling
        it += 1
    
    return best_points, best_score


def construct():
    start_time = time.time()
    time_budget = 130.0  # Leave 20s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Try multiple structured initializations with local optimization
    n_structured = 5
    for s in range(n_structured):
        if time.time() - start_time > time_budget - 10:
            break
        pts = _structured_init(seed=s * 100)
        pts, score = _local_optimize(pts, max_iters=1500, step_size=0.05,
                                     time_limit=time_budget, start_time=start_time)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 2: Simulated annealing from best found
    if best_points is not None:
        if time.time() - start_time < time_budget - 20:
            pts, score = _simulated_annealing(best_points, max_iters=5000,
                                              time_limit=time_budget, start_time=start_time)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # Phase 3: Random restarts if time remains
    n_random = 10
    for s in range(n_random):
        if time.time() - start_time > time_budget - 5:
            break
        pts = _random_init(seed=1000 + s * 7)
        pts, score = _local_optimize(pts, max_iters=1000, step_size=0.04,
                                     time_limit=time_budget, start_time=start_time)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Final refinement from best
    if best_points is not None and time.time() - start_time < time_budget - 3:
        pts, score = _local_optimize(best_points, max_iters=3000, step_size=0.02,
                                     time_limit=time_budget, start_time=start_time)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: structured init without optimization
        best_points = _structured_init(seed=0)
    
    return best_points
