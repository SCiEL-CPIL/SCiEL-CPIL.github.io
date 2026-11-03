"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute all index triples for vectorized area computation
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])


def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])


def _clamp_bary(bary):
    """Clamp barycentric coords to valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary = bary.copy()
        bary[mask] = bary[mask] / s[mask, None]
    return bary


def _objective_bary(x):
    """Objective: negative min triangle area, with points in barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)


def _evaluate_bary(bary):
    """Evaluate barycentric config, return (score, clamped_bary)."""
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    score = _min_triangle_area_vectorized(points)
    return score, bary


def _make_initial_bary(seed, strategy='grid'):
    """Generate initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    if strategy == 'grid':
        bary_list = [
            [0, 0],
            [1, 0],
            [0, 1],
            [0.5, 0],
            [0.5, 0.5],
            [0, 0.5],
            [1/3, 1/3],
            [1/3, 2/3],
            [2/3, 1/3],
            [0.25, 0.25],
            [0.75, 0.25],
        ]
        while len(bary_list) < N:
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            if u + v < 0.9:
                bary_list.append([u, v])
        return np.array(bary_list[:N])
    elif strategy == 'hex':
        # Hexagonal-like grid in barycentric space
        bary_list = []
        for i in range(4):
            for j in range(4):
                u = i * 0.3 + (j % 2) * 0.15
                v = j * 0.3
                if u >= 0 and v >= 0 and u + v <= 1:
                    bary_list.append([u, v])
        # Deduplicate and fill
        seen = set()
        unique = []
        for b in bary_list:
            key = (round(b[0], 2), round(b[1], 2))
            if key not in seen:
                seen.add(key)
                unique.append(b)
        for b in unique:
            if len(bary_list) >= N:
                break
            bary_list.append(b)
        while len(bary_list) < N:
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            if u + v < 0.9:
                bary_list.append([u, v])
        return np.array(bary_list[:N])
    else:
        # Random spread
        bary_list = []
        for _ in range(500):
            u = rng.uniform(0.02, 0.98)
            v = rng.uniform(0.02, 0.98)
            if u + v < 0.95:
                bary_list.append([u, v])
                if len(bary_list) == N:
                    break
        while len(bary_list) < N:
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            if u + v < 0.85:
                bary_list.append([u, v])
        return np.array(bary_list[:N])


def _local_optimize(bary0, maxiter=500, gtol=1e-12, deadline=None):
    """Run L-BFGS-B local optimization in barycentric space."""
    x0 = bary0.flatten()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(
            _objective_bary, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        bary_opt = res.x.reshape(-1, 2)
        bary_opt = _clamp_bary(bary_opt)
        return bary_opt
    except Exception:
        return bary0


def _de_step(pop, fitness, F=0.8, CR=0.9, rng=None):
    """One differential evolution step in barycentric space."""
    n_pop = len(pop)
    dim = 2 * N
    new_pop = [p.copy() for p in pop]
    new_fitness = list(fitness)
    
    for i in range(n_pop):
        # Select three distinct individuals
        candidates = [j for j in range(n_pop) if j != i]
        a, b, c = rng.choice(candidates, size=3, replace=False)
        
        # DE mutation
        v = pop[a] + F * (pop[b] - pop[c])
        
        # Crossover
        r = rng.random()
        j_rand = rng.randint(dim)
        trial = np.zeros(dim)
        for j in range(dim):
            if j == j_rand or rng.random() < CR:
                trial[j] = v[j]
            else:
                trial[j] = pop[i][j]
        
        # Clamp to valid barycentric region
        trial_bary = _clamp_bary(trial.reshape(-1, 2))
        trial_score = -_min_triangle_area_vectorized(_bary_to_cart(trial_bary))
        
        # Selection
        if trial_score <= new_fitness[i]:
            new_pop[i] = trial_bary.flatten()
            new_fitness[i] = trial_score
    
    return new_pop, new_fitness


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_points = None
    best_score = -np.inf
    best_bary = None

    def time_left():
        return deadline - time.time()

    def update_best(bary):
        nonlocal best_points, best_score, best_bary
        score, bary_clamped = _evaluate_bary(bary)
        if score > best_score:
            best_score = score
            best_bary = bary_clamped.copy()
            best_points = _bary_to_cart(bary_clamped).copy()

    # Phase 1: Multiple restarts with local optimization
    strategies = ['grid', 'hex', 'random']
    n_restarts = 40
    for seed in range(n_restarts):
        if time_left() < 5.0:
            break
        
        strategy = strategies[seed % len(strategies)]
        bary0 = _make_initial_bary(seed * 7919 + 1, strategy)
        
        # Quick local optimization
        bary_opt = _local_optimize(bary0, maxiter=300, gtol=1e-10)
        update_best(bary_opt)

    # Phase 2: Differential evolution in barycentric space
    if best_bary is not None and time_left() > 15.0:
        rng = np.random.RandomState(42)
        
        # Initialize population around best solution
        pop = [best_bary.flatten().copy() for _ in range(20)]
        for i in range(1, 20):
            scale = 0.1 + 0.3 * (i / 20.0)
            perturbation = rng.normal(0, scale, size=best_bary.shape)
            pop[i] = _clamp_bary(best_bary + perturbation).flatten()
        
        # Also add some diverse initial points
        for i in range(15, 20):
            bary0 = _make_initial_bary(i * 104729 + 7, 'random')
            pop[i] = bary0.flatten()
        
        fitness = [-_min_triangle_area_vectorized(_bary_to_cart(_clamp_bary(p.reshape(-1, 2)))) for p in pop]
        
        # DE iterations
        de_iter = 0
        while time_left() > 10.0 and de_iter < 80:
            pop, fitness = _de_step(pop, fitness, F=0.8, CR=0.9, rng=rng)
            
            # Track best
            best_idx = np.argmin(fitness)  # most negative = best
            best_de_bary = _clamp_bary(pop[best_idx].reshape(-1, 2))
            update_best(best_de_bary)
            
            # Periodically re-center population around best
            if de_iter % 10 == 9:
                for i in range(len(pop)):
                    if i != best_idx:
                        scale = 0.05 + 0.15 * rng.random()
                        pop[i] = _clamp_bary(best_de_bary + rng.normal(0, scale, size=best_de_bary.shape)).flatten()
                        fitness[i] = -_min_triangle_area_vectorized(_bary_to_cart(_clamp_bary(pop[i].reshape(-1, 2))))
            
            de_iter += 1

    # Phase 3: Intensive local polishing around best solution
    if best_bary is not None and time_left() > 8.0:
        rng = np.random.RandomState(999)
        n_polish = 15
        for i in range(n_polish):
            if time_left() < 3.0:
                break
            
            # Multi-scale perturbation
            scale = 0.02 + 0.08 * (i / max(n_polish, 1)) * rng.random()
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_pert = _clamp_bary(bary_pert)
            
            # Tight local optimization
            bary_opt = _local_optimize(bary_pert, maxiter=800, gtol=1e-13)
            update_best(bary_opt)

    # Phase 4: Final fine-tuning with very tight tolerance
    if best_bary is not None and time_left() > 3.0:
        bary_final = _local_optimize(best_bary, maxiter=1000, gtol=1e-14)
        update_best(bary_final)

    # Final fallback
    if best_points is None:
        bary0 = _make_initial_bary(0, 'grid')
        best_points = _bary_to_cart(_clamp_bary(bary0))

    return np.array(best_points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
