"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 115.0  # Safety margin


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def construct():
    start_time = time.time()
    
    # Genetic Algorithm Parameters
    pop_size = 200
    elite_size = 10
    mutation_rate = 0.1
    mutation_scale = 0.05
    crossover_rate = 0.8
    
    # Initialize population
    # Mix of random points and regular polygon perturbations to give it a head start
    # but the GA will explore freely.
    rng = np.random.RandomState(42)
    
    def init_individual():
        # 50% chance: Regular polygon with noise
        # 50% chance: Random points in unit disk
        if rng.rand() < 0.5:
            angles = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.1, 0.1, N)
            r = 1.0 + rng.uniform(-0.1, 0.1, N)
            pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        else:
            # Random in unit disk
            r = np.sqrt(rng.rand(N))
            theta = rng.rand(N) * 2 * np.pi
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        return pts

    population = [init_individual() for _ in range(pop_size)]
    scores = np.array([_score(p) for p in population])
    
    best_idx = np.argmax(scores)
    best_pts = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # GA Loop
    while time.time() - start_time < DEADLINE - 5:
        # Sort population by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = scores[order]
        
        # Elitism: Keep the best 'elite_size' individuals
        new_population = [population[i].copy() for i in range(elite_size)]
        new_scores = scores[:elite_size].copy()
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament Selection
            def select():
                idx1 = rng.randint(0, pop_size)
                idx2 = rng.randint(0, pop_size)
                if scores[idx1] > scores[idx2]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover
            if rng.rand() < crossover_rate:
                # Uniform Crossover
                mask = rng.rand(N, 2) > 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if rng.rand() < mutation_rate:
                # Gaussian Mutation
                noise = rng.normal(0, mutation_scale, (N, 2))
                child = child + noise
            
            # Evaluate Child
            child_score = _score(child)
            
            # Replacement Strategy: If child is better than worst in current new_population or just append
            # Simplest: Append and sort later, or replace worst if better.
            # For speed, just append and sort at end of generation.
            new_population.append(child)
            new_scores = np.append(new_scores, child_score)
            
            # Update best
            if child_score > best_score:
                best_score = child_score
                best_pts = child.copy()
        
        population = new_population
        scores = new_scores
        
        # Adaptive Mutation Scale: Decrease over time
        # Simple heuristic: reduce scale as time passes
        elapsed = time.time() - start_time
        progress = elapsed / (DEADLINE - 5)
        mutation_scale = 0.05 * (1.0 - progress) + 0.001
        
    # Final Local Refinement of the best GA result
    if time.time() - start_time < DEADLINE - 3:
        # Use Nelder-Mead for a quick polish
        from scipy.optimize import minimize
        
        def _neg_score(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            return -_score(pts)
        
        try:
            res = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
