"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import math
from itertools import combinations
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all C(n,3) triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                area = 0.5 * abs(dx_ij * (yk - yi) - dy_ij * (xk - xi))
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.area
    except Exception:
        # Fallback: if all points are collinear or nearly so
        return 1e-10


def _score(pts):
    """Score = min triangle area / hull area."""
    h_area = _hull_area(pts)
    if h_area < 1e-12:
        return 0.0
    m_area = _min_triangle_area(pts)
    return m_area / h_area


def _halton_sequence(index, base):
    """Generate the Halton sequence value for a given index and base."""
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def _initial_halton():
    """Generate initial points using Halton sequence in [0,1]^2."""
    pts = []
    for i in range(1, N + 1):
        x = _halton_sequence(i, 2)
        y = _halton_sequence(i, 3)
        pts.append([x, y])
    return np.array(pts, dtype=np.float64)


def _regular_polygon_with_center(n):
    """Generate n points: (n-1) on a regular polygon + 1 at center."""
    k = n - 1  # points on polygon
    angles = np.linspace(0, 2 * math.pi, k, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.5, 0.5]])
    return np.vstack([pts * 0.45 + 0.5, center])


def _perturb_and_optimize(pts0, time_limit, rng):
    """Local optimization: iterative perturbation of worst triangles."""
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    
    # Precompute all triples for efficiency
    triples = list(combinations(range(N), 3))
    n_triples = len(triples)
    
    deadline = time.time() + time_limit
    step_size = 0.05
    iters = 0
    
    while time.time() < deadline - 2.0:
        iters += 1
        # Find the worst triple
        min_area = float('inf')
        worst_idx = 0
        for idx, (i, j, k) in enumerate(triples):
            xi, yi = best_pts[i]
            xj, yj = best_pts[j]
            xk, yk = best_pts[k]
            area = 0.5 * abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
            if area < min_area:
                min_area = area
                worst_idx = idx
        
        i, j, k = triples[worst_idx]
        
        # Try moving each of the three points slightly to increase the area
        improved = False
        for pt_idx in [i, j, k]:
            for attempt in range(20):
                # Random small perturbation
                dx = rng.gauss(0, step_size)
                dy = rng.gauss(0, step_size)
                new_pts = best_pts.copy()
                new_pts[pt_idx, 0] += dx
                new_pts[pt_idx, 1] += dy
                
                new_score = _score(new_pts)
                if new_score > best_score + 1e-12:
                    best_pts = new_pts
                    best_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.95
            if step_size < 1e-6:
                break
        
        # Occasionally try a larger random move to escape local optima
        if iters % 50 == 0:
            idx = rng.randint(0, N - 1)
            best_pts[idx, 0] += rng.gauss(0, step_size * 3)
            best_pts[idx, 1] += rng.gauss(0, step_size * 3)
            s = _score(best_pts)
            if s < best_score:
                best_score = s  # accept for now, we'll re-optimize
    
    return best_pts, best_score


def _differential_evolution_step(pts0, time_limit, rng):
    """Simple population-based search."""
    pop_size = 10
    population = [pts0.copy()]
    for _ in range(pop_size - 1):
        p = pts0.copy()
        for idx in range(N):
            p[idx, 0] += rng.gauss(0, 0.1)
            p[idx, 1] += rng.gauss(0, 0.1)
        population.append(p)
    
    scores = [_score(p) for p in population]
    best_idx = int(np.argmax(scores))
    best_pts = population[best_idx].copy()
    best_score = scores[best_idx]
    
    deadline = time.time() + time_limit
    gen = 0
    
    while time.time() < deadline - 2.0:
        gen += 1
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i].copy() for i in order]
        scores = [scores[i] for i in order]
        
        # Generate offspring
        for i in range(pop_size):
            if time.time() > deadline - 2.0:
                break
            # Select three distinct parents
            a, b, c = rng.sample(range(pop_size), 3)
            trial = population[a].copy()
            for idx in range(N):
                for dim in range(2):
                    if rng.random() < 0.5:
                        trial[idx, dim] = population[b][idx, dim] + rng.uniform(0, 0.1) * (population[c][idx, dim] - population[a][idx, dim])
                    else:
                        trial[idx, dim] = population[a][idx, dim] + rng.gauss(0, 0.05)
            
            trial_score = _score(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score
        
        # Update best
        max_score = max(scores)
        max_idx = int(np.argmax(scores))
        if max_score > best_score:
            best_score = max_score
            best_pts = population[max_idx].copy()
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    total_budget = 110.0  # Leave margin for the 120s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_sets = [
        _initial_halton(),
        _regular_polygon_with_center(N),
        np.array([[rng.random(), rng.random()] for _ in range(N)], dtype=np.float64),
        np.array([[rng.random(), rng.random()] for _ in range(N)], dtype=np.float64),
    ]
    
    best_pts = None
    best_score = 0.0
    
    for pts0 in initial_sets:
        s = _score(pts0)
        if s > best_score:
            best_score = s
            best_pts = pts0.copy()
    
    if best_pts is None:
        best_pts = _initial_halton()
        best_score = _score(best_pts)
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Phase 1: Local optimization on best initial
    if remaining > 5:
        t1 = min(remaining * 0.5, 40.0)
        best_pts, best_score = _perturb_and_optimize(best_pts, t1, rng)
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Phase 2: Population-based search
    if remaining > 5:
        t2 = min(remaining * 0.6, 50.0)
        pts2, score2 = _differential_evolution_step(best_pts, t2, rng)
        if score2 > best_score:
            best_score = score2
            best_pts = pts2
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Phase 3: Final local refinement
    if remaining > 3:
        t3 = min(remaining * 0.7, 30.0)
        best_pts, best_score = _perturb_and_optimize(best_pts, t3, rng)
    
    return best_pts

# EVOLVE-BLOCK-END
