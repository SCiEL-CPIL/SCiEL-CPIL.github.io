# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _initial_bcc():
    """Create BCC-like lattice initial placement."""
    a = 0.28
    points = []
    # BCC basis: corners + center
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    center = np.array([0.5, 0.5, 0.5])
    
    for ix in range(-1, 3):
        for iy in range(-1, 3):
            for iz in range(-1, 3):
                origin = np.array([ix*a, iy*a, iz*a])
                for corner in corners:
                    p = origin + corner * a
                    if np.all(np.abs(p) <= CUBE_HALF):
                        points.append(p)
                p_center = origin + center * a
                if np.all(np.abs(p_center) <= CUBE_HALF):
                    points.append(p_center)
    
    points = np.array(points)
    np.random.shuffle(points)
    if len(points) >= N:
        return points[:N].copy()
    else:
        return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _initial_fcc():
    """Create FCC-like lattice initial placement."""
    a = 0.28
    points = []
    basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
    
    for ix in range(-1, 3):
        for iy in range(-1, 3):
            for iz in range(-1, 3):
                origin = np.array([ix*a, iy*a, iz*a])
                for b in basis:
                    p = origin + b * a
                    if np.all(np.abs(p) <= CUBE_HALF):
                        points.append(p)
    
    points = np.array(points)
    np.random.shuffle(points)
    if len(points) >= N:
        return points[:N].copy()
    else:
        return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _initial_random():
    """Create a random initial placement."""
    return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _initial_jittered_lattice():
    """Create a jittered simple cubic lattice."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.shuffle(grid)
    c = grid[:N].copy()
    # Add small random jitter
    c += np.random.randn(N, 3) * 0.01
    c = np.clip(c, -CUBE_HALF, CUBE_HALF)
    return c

def _stochastic_local_search(c, time_limit=50, T_start=0.01, T_end=1e-8, alpha=0.997):
    """
    Stochastic Local Search with simulated annealing.
    Includes swap moves to help escape local minima.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T_start
    step_size = 0.03
    
    # Move type probabilities: single, double, swap, jump
    p_single = 0.45
    p_double = 0.15
    p_swap = 0.25
    p_jump = 0.15
    
    while time.time() - start_time < time_limit:
        r = random.random()
        c_new = c.copy()
        
        if r < p_single:
            # Move one point
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_size
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
        elif r < p_single + p_double:
            # Move two points
            i1, i2 = random.sample(range(N), 2)
            c_new[i1] = c_new[i1] + np.random.randn(3) * step_size
            c_new[i2] = c_new[i2] + np.random.randn(3) * step_size
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
        elif r < p_single + p_double + p_swap:
            # Swap positions of two points (with small perturbation)
            i1, i2 = random.sample(range(N), 2)
            # Swap with some mixing to avoid exact swap
            mix = random.uniform(0.3, 0.7)
            orig1 = c_new[i1].copy()
            orig2 = c_new[i2].copy()
            c_new[i1] = mix * orig2 + (1 - mix) * orig1
            c_new[i2] = mix * orig1 + (1 - mix) * orig2
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
        else:
            # Large jump for one point
            idx = random.randint(0, N - 1)
            c_new[idx] = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=3)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta = new_score - current_score
            if delta > -100 * T:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
        
        # Adaptive step size
        step_size = max(0.003, 0.03 * (T / T_start) ** 0.4)
    
    return best_c, best_score

def _polish(c, time_limit=10):
    """
    Final polishing: hill climbing with small steps.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step_size in step_sizes:
        while time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # Try multiple directions
                directions = [
                    (step_size, 0, 0), (-step_size, 0, 0),
                    (0, step_size, 0), (0, -step_size, 0),
                    (0, 0, step_size), (0, 0, -step_size),
                    (step_size/np.sqrt(3), step_size/np.sqrt(3), step_size/np.sqrt(3)),
                    (-step_size/np.sqrt(3), -step_size/np.sqrt(3), -step_size/np.sqrt(3)),
                    (step_size/np.sqrt(2), 0, step_size/np.sqrt(2)),
                    (-step_size/np.sqrt(2), 0, -step_size/np.sqrt(2)),
                    (0, step_size/np.sqrt(2), step_size/np.sqrt(2)),
                    (0, -step_size/np.sqrt(2), -step_size/np.sqrt(2)),
                ]
                for dx, dy, dz in directions:
                    c_test = c.copy()
                    c_test[i] = c[i] + np.array([dx, dy, dz])
                    c_test[i] = np.clip(c_test[i], -CUBE_HALF, CUBE_HALF)
                    s_test = _score(c_test)
                    if s_test > current_score + 1e-12:
                        c = c_test
                        current_score = s_test
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_c = c.copy()
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_c, best_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s total. Leave 5s for overhead.
    total_budget = 115
    
    # Multiple diverse starting points
    starts = [
        ('bcc', _initial_bcc()),
        ('fcc', _initial_fcc()),
        ('jittered', _initial_jittered_lattice()),
        ('random1', _initial_random()),
        ('random2', _initial_random()),
        ('random3', _initial_random()),
    ]
    
    # Run SLS on each start with decreasing time allocation
    time_allocs = [22, 22, 18, 15, 15, 13]
    
    for (name, c_start), t_alloc in zip(starts, time_allocs):
        c_best, s_best = _stochastic_local_search(c_start, time_limit=t_alloc)
        if s_best > best_score:
            best_score = s_best
            best_c = c_best
    
    # Final refinement on the best solution
    if best_c is not None:
        c_polish, s_polish = _polish(best_c, time_limit=10)
        if s_polish > best_score:
            best_score = s_polish
            best_c = c_polish
    
    return best_c
# EVOLVE-BLOCK-END
