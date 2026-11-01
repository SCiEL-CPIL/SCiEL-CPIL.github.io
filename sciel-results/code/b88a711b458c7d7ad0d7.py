import time
import numpy as np
from scipy.spatial.distance import cdist
import sys

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _init_tetrahedron_symmetric():
    """
    Generate initial points based on a regular tetrahedron structure.
    4 vertices + 6 edge midpoints + 4 face centers = 14 points.
    This structure has high symmetry (T_d) which is a good starting point
    for minimizing distance ratio.
    """
    # Regular tetrahedron vertices centered at origin
    # Vertices of a regular tetrahedron inscribed in a sphere
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    # Normalize to unit sphere
    v = v / np.linalg.norm(v, axis=1)[:, None]
    
    # 4 Vertices
    pts = v.copy()
    
    # 6 Edge Midpoints
    # Edges are all pairs of vertices
    mids = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (v[i] + v[j]) / 2.0
            # Scale to lie on the same "radius" as vertices? 
            # No, midpoints are naturally closer to center. 
            # But for the ratio, we want them distinct.
            # Let's keep them at their natural geometric position first.
            mids.append(mid)
    mids = np.array(mids)
    pts = np.vstack([pts, mids])
    
    # 4 Face Centers
    # Faces are all triplets of vertices
    centers = []
    # Indices: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    for f in faces:
        center = (v[f[0]] + v[f[1]] + v[f[2]]) / 3.0
        centers.append(center)
    centers = np.array(centers)
    pts = np.vstack([pts, centers])
    
    # Center the whole configuration
    pts = pts - pts.mean(axis=0)
    
    # Normalize scale (optional, but helps with GA step sizes)
    # Let's normalize so that the mean squared distance is 1 or similar
    # Actually, let's just normalize so that the max distance is 1.
    d = cdist(pts, pts)
    np.fill_diagonal(d, 0)
    max_d = d.max()
    if max_d > 0:
        pts = pts / max_d
        
    return pts

def _init_fibonacci_sphere():
    """
    Fallback initialization: Fibonacci sphere.
    """
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts = np.column_stack([x, y, z])
    pts = pts - pts.mean(axis=0)
    d = cdist(pts, pts)
    np.fill_diagonal(d, 0)
    max_d = d.max()
    if max_d > 0:
        pts = pts / max_d
    return pts

def _ga_objective(pts):
    """
    Objective function for the Genetic Algorithm.
    Returns the squared ratio.
    """
    return _true_score(pts)

def _ga_step(population, scores, rng, step_size):
    """
    Perform one step of the Genetic Algorithm.
    1. Selection (Tournament)
    2. Crossover (Blend)
    3. Mutation (Gaussian)
    """
    n_pop = len(population)
    new_population = []
    
    # Elitism: Keep the best 10%
    num_elite = max(1, n_pop // 10)
    elite_indices = np.argsort(scores)[:num_elite]
    for idx in elite_indices:
        new_population.append(population[idx].copy())
    
    # Fill the rest with offspring
    while len(new_population) < n_pop:
        # Tournament Selection
        t_size = 3
        parents = []
        for _ in range(2):
            candidates = rng.integers(0, n_pop, t_size)
            best_candidate = candidates[np.argmin(scores[candidates])]
            parents.append(population[best_candidate])
        
        # Blend Crossover (Linear)
        # p_child = alpha * p1 + (1 - alpha) * p2
        alpha = rng.uniform(0.2, 0.8)
        child = alpha * parents[0] + (1 - alpha) * parents[1]
        
        # Mutation
        # Adaptive step size based on current best score?
        # Simple: fixed step size with some random variation
        noise = rng.normal(0, step_size, child.shape)
        child = child + noise
        
        # Normalize scale to prevent drift
        d = cdist(child, child)
        np.fill_diagonal(d, 0)
        max_d = d.max()
        if max_d > 1e-9:
            child = child / max_d
        else:
            # Degenerate, re-initialize this individual
            child = _init_fibonacci_sphere()
            
        new_population.append(child)
        
    return np.array(new_population)

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    # Initialize Population
    # Start with the symmetric tetrahedron structure for many individuals
    # and some random/fibonacci for diversity
    pop_size = 60
    population = []
    
    # 50% Tetrahedron Symmetric
    for _ in range(pop_size // 2):
        pts = _init_tetrahedron_symmetric()
        # Add small random perturbation to break symmetry slightly
        pts += rng.normal(0, 0.01, pts.shape)
        d = cdist(pts, pts)
        np.fill_diagonal(d, 0)
        max_d = d.max()
        if max_d > 1e-9:
            pts = pts / max_d
        population.append(pts)
        
    # 30% Fibonacci
    for _ in range(pop_size // 3):
        pts = _init_fibonacci_sphere()
        pts += rng.normal(0, 0.01, pts.shape)
        d = cdist(pts, pts)
        np.fill_diagonal(d, 0)
        max_d = d.max()
        if max_d > 1e-9:
            pts = pts / max_d
        population.append(pts)
        
    # 20% Random
    for _ in range(pop_size - len(population)):
        pts = rng.normal(size=(N, D))
        pts = pts - pts.mean(axis=0)
        d = cdist(pts, pts)
        np.fill_diagonal(d, 0)
        max_d = d.max()
        if max_d > 1e-9:
            pts = pts / max_d
        population.append(pts)
        
    population = np.array(population)
    
    # Evaluate initial population
    scores = np.array([_ga_objective(p) for p in population])
    
    best_score = scores.min()
    best_idx = np.argmin(scores)
    best_pts = population[best_idx].copy()
    
    # GA Loop
    step_size = 0.05
    patience = 0
    max_patience = 20
    
    while time.time() - _t0 < DEADLINE - 5:
        # Evaluate new population
        new_population = _ga_step(population, scores, rng, step_size)
        population = new_population
        scores = np.array([_ga_objective(p) for p in population])
        
        current_best = scores.min()
        current_best_idx = np.argmin(scores)
        
        if current_best < best_score:
            best_score = current_best
            best_pts = population[current_best_idx].copy()
            patience = 0
            # Reduce step size slightly on improvement
            step_size *= 0.95
        else:
            patience += 1
            # If stuck, increase step size to escape local minima
            if patience % 5 == 0:
                step_size *= 1.1
                # Occasionally inject a completely new random individual
                if patience % 10 == 0:
                    idx_to_replace = rng.integers(0, pop_size)
                    population[idx_to_replace] = _init_fibonacci_sphere() + rng.normal(0, 0.05, (N, D))
                    d = cdist(population[idx_to_replace], population[idx_to_replace])
                    np.fill_diagonal(d, 0)
                    max_d = d.max()
                    if max_d > 1e-9:
                        population[idx_to_replace] = population[idx_to_replace] / max_d
                    scores[idx_to_replace] = _ga_objective(population[idx_to_replace])
        
        if patience > max_patience * 2:
            # Reset step size and try a different strategy
            step_size = 0.1
            patience = 0

    # Final check
    if _true_score(best_pts) < best_score:
        best_score = _true_score(best_pts)
        
    return best_pts
