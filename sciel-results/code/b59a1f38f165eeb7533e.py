"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1
H = np.sqrt(3) / 2.0     # height of the triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _barycentric_to_cartesian(b):
    """Convert barycentric coordinates (u,v,w) to Cartesian."""
    u, v, w = b
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.array([x, y])


def _cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric coordinates."""
    x, y = p
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.array([u, v, w])


def _clamp_to_triangle(p, eps=1e-9):
    """Clamp point to be inside the triangle using barycentric coordinates."""
    b = _cartesian_to_barycentric(p)
    if np.any(b < -eps):
        b = np.maximum(b, 0.0)
        s = b.sum()
        if s < 1e-12:
            return (V0 + V1 + V2) / 3.0
        b = b / s
    return _barycentric_to_cartesian(b)


def _clamp_points(points):
    """Clamp an array of points to the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])


def _min_triangle_area_fast(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    """
    n = len(points)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i = i_idx[mask]
    j = j_idx[mask]
    k = k_idx[mask]
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _min_triangle_area_normalized(points):
    """Return min triangle area divided by triangle area."""
    return _min_triangle_area_fast(points) / AREA


def _generate_initial_config(rng, strategy='structured'):
    """Generate an initial configuration of 11 points."""
    if strategy == 'structured':
        inset = 0.05
        p0 = V0 + inset * (V1 - V0) + inset * (V2 - V0)
        p1 = V1 + inset * (V0 - V1) + inset * (V2 - V1)
        p2 = V2 + inset * (V0 - V2) + inset * (V1 - V2)
        m01 = (V0 + V1) / 2.0
        m12 = (V1 + V2) / 2.0
        m20 = (V2 + V0) / 2.0
        c = (V0 + V1 + V2) / 3.0
        points = [p0, p1, p2, m01, m12, m20, c]
        while len(points) < N:
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            w = 1.0 - u - v
            if w > 0.05:
                p = _barycentric_to_cartesian(np.array([u, v, w]))
                min_dist = min(np.linalg.norm(p - q) for q in points)
                if min_dist > 0.05:
                    points.append(p)
        return np.array(points)
    elif strategy == 'random':
        points = []
        for _ in range(N):
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                p = [u + 0.5 * v, H * v]
                points.append(p)
            else:
                points.append([rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.4)])
        return np.array(points)
    else:
        return _generate_initial_config(rng, 'random')


def _simulated_annealing(points_init, time_limit, rng):
    """Run Simulated Annealing to maximize the min triangle area."""
    start_time = time.time()
    current_points = points_init.copy()
    current_area = _min_triangle_area_fast(current_points)
    best_points = current_points.copy()
    best_area = current_area
    T = 0.01
    alpha = 0.999
    step_size = 0.05
    while time.time() < time_limit:
        num_points_to_move = rng.randint(1, 3)
        idxs = rng.sample(range(N), num_points_to_move)
        new_points = current_points.copy()
        for idx in idxs:
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] += noise
        new_points = _clamp_points(new_points)
        new_area = _min_triangle_area_fast(new_points)
        delta = new_area - current_area
        if delta > 0:
            current_points = new_points
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
        else:
            if T > 1e-6:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_area = new_area
        T *= alpha
        step_size = max(0.001, 0.05 * (T / 0.01))
    return best_points, best_area


def _differential_evolution_search(time_limit, rng):
    """
    Differential Evolution to maximize min triangle area.
    This is a population-based global optimizer that works well for this kind of problem.
    """
    start_time = time.time()
    pop_size = 30
    F = 0.8  # differential weight
    CR = 0.9  # crossover rate
    
    # Initialize population with diverse starting points
    population = []
    for i in range(pop_size):
        strategy = 'structured' if i % 2 == 0 else 'random'
        pts = _generate_initial_config(rng, strategy)
        pts = _clamp_points(pts)
        population.append(pts)
    
    # Evaluate initial population
    scores = [_min_triangle_area_fast(pts) for pts in population]
    best_idx = np.argmax(scores)
    best_points = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # DE loop
    while time.time() < time_limit:
        # For each individual, create a mutant
        for i in range(pop_size):
            # Select two distinct random individuals different from i
            r1, r2 = rng.sample([j for j in range(pop_size) if j != i], 2)
            
            # Mutation: v = population[r1] + F * (population[r2] - population[i])
            mutant = population[r1] + F * (population[r2] - population[i])
            
            # Crossover: create trial vector
            trial = np.zeros_like(population[i])
            for d in range(N * 2):
                if rng.random() < CR or d == rng.randint(0, N * 2 - 1):
                    trial[d] = mutant[d]
                else:
                    trial[d] = population[i][d]
            
            # Clamp to triangle
            trial_pts = trial.reshape(-1, 2)
            trial_pts = _clamp_points(trial_pts)
            
            # Evaluate
            trial_score = _min_triangle_area_fast(trial_pts)
            
            # Selection
            if trial_score >= scores[i]:
                population[i] = trial_pts
                scores[i] = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = trial_pts.copy()
            
            # Check time
            if time.time() > time_limit:
                break
        if time.time() > time_limit:
            break
    
    return best_points, best_score


def _local_polish(points, time_limit, rng):
    """
    Local optimization using coordinate descent on barycentric coordinates.
    For each point, try to move it in small steps to improve the min triangle area.
    """
    start_time = time.time()
    pts = points.copy()
    current_score = _min_triangle_area_fast(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Use barycentric coordinates for constraint handling
    bary_pts = np.array([_cartesian_to_barycentric(p) for p in pts])
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step in step_sizes:
        if time.time() > time_limit:
            break
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N):
                # Try moving point i in 8 directions (and stay)
                directions = [
                    [0, 0],
                    [step, 0], [-step, 0],
                    [0, step], [0, -step],
                    [step, step], [step, -step], [-step, step], [-step, -step]
                ]
                best_local_score = current_score
                best_local_bary = bary_pts[i].copy()
                
                for dx, dy in directions:
                    new_bary = bary_pts[i].copy()
                    new_bary[0] += dx
                    new_bary[1] += dy
                    # Normalize to stay in triangle
                    new_bary = np.maximum(new_bary, 0)
                    s = new_bary.sum()
                    if s < 1e-12:
                        continue
                    new_bary = new_bary / s
                    # Check if valid (should be after normalization)
                    if np.any(new_bary < -1e-9):
                        continue
                    
                    # Convert to cartesian and evaluate
                    new_pts = pts.copy()
                    new_pts[i] = _barycentric_to_cartesian(new_bary)
                    new_score = _min_triangle_area_fast(new_pts)
                    
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_bary = new_bary.copy()
                
                if best_local_score > current_score + 1e-12:
                    bary_pts[i] = best_local_bary
                    pts[i] = _barycentric_to_cartesian(bary_pts[i])
                    current_score = best_local_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    best_points_global = None
    best_score_global = -float('inf')
    
    # Phase 1: Differential Evolution (global search)
    de_time = 60.0
    de_deadline = start_time + de_time
    if time.time() < de_deadline:
        points_de, score_de = _differential_evolution_search(de_deadline, rng)
        if score_de > best_score_global:
            best_score_global = score_de
            best_points_global = points_de.copy()
    
    # Phase 2: Local polishing on the best result
    if best_points_global is not None and time.time() < deadline - 10.0:
        polish_deadline = deadline - 5.0
        points_polished, score_polished = _local_polish(best_points_global, polish_deadline, rng)
        if score_polished > best_score_global:
            best_score_global = score_polished
            best_points_global = points_polished.copy()
    
    # Phase 3: Run a few SA runs on the best result for fine-tuning
    if best_points_global is not None and time.time() < deadline - 10.0:
        sa_deadline = deadline - 5.0
        points_sa, score_sa = _simulated_annealing(best_points_global, sa_deadline, rng)
        if score_sa > best_score_global:
            best_score_global = score_sa
            best_points_global = points_sa.copy()
    
    # Final local polish
    if best_points_global is not None and time.time() < deadline - 3.0:
        final_deadline = deadline - 1.0
        points_final, score_final = _local_polish(best_points_global, final_deadline, rng)
        if score_final > best_score_global:
            best_score_global = score_final
            best_points_global = points_final.copy()
    
    # Ensure final points are valid
    if best_points_global is not None:
        best_points_global = _clamp_points(best_points_global)
    
    return best_points_global


# EVOLVE-BLOCK-START
# (The construct function is defined above, but we keep this block for compatibility)
# EVOLVE-BLOCK-END
