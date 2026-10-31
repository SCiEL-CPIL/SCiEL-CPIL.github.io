"""Heilbronn convex regions: 13 points, maximize min triangle area / hull area."""
import time
import numpy as np
from itertools import combinations

N = 13

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull. 
    Assuming points are given in angular order around the centroid/origin, 
    the hull is the polygon formed by the points."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)

def _score_from_radii(radii, angles):
    """Compute score given radii and fixed angles."""
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _ga_objective(radii, angles):
    """Objective for GA: negative score."""
    # Ensure radii are positive
    radii = np.maximum(radii, 0.1)
    return -_score_from_radii(radii, angles)

def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fixed angles for 13 vertices
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Initialize population
    pop_size = 50
    # Start with a regular polygon (all radii 1.0) and some noise
    initial_pop = np.ones((pop_size, N)) * 1.0
    initial_pop += np.random.normal(0, 0.1, (pop_size, N))
    
    best_radii = initial_pop[0].copy()
    best_score = _score_from_radii(best_radii, angles)
    
    # Evaluate initial population
    scores = np.array([-_ga_objective(r, angles) for r in initial_pop])
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_radii = initial_pop[best_idx].copy()
    
    # Genetic Algorithm Loop
    pop = initial_pop
    gen = 0
    while time.time() < deadline:
        gen += 1
        # 1. Evaluate current population if not done
        # (We did initial, now we need to evaluate offspring)
        
        # 2. Selection (Tournament)
        new_pop = np.zeros_like(pop)
        # Elitism: keep top 5
        top_indices = np.argsort(scores)[-5:]
        new_pop[:5] = pop[top_indices]
        
        # Fill rest with offspring
        for i in range(5, pop_size):
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            parent1 = pop[idx1] if scores[idx1] > scores[idx2] else pop[idx2]
            
            idx3 = np.random.randint(0, pop_size)
            idx4 = np.random.randint(0, pop_size)
            parent2 = pop[idx3] if scores[idx3] > scores[idx4] else pop[idx4]
            
            # Crossover (Blend)
            alpha = np.random.uniform(0.0, 1.0)
            child = alpha * parent1 + (1 - alpha) * parent2
            
            # Mutation (Gaussian)
            # Adaptive mutation rate? Keep simple: 0.1 std
            child += np.random.normal(0, 0.05, N)
            
            # Clamp to valid range
            child = np.clip(child, 0.1, 10.0)
            
            new_pop[i] = child
        
        pop = new_pop
        
        # 3. Evaluate new population
        scores = np.array([-_ga_objective(r, angles) for r in pop])
        
        # 4. Update best
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_radii = pop[current_best_idx].copy()
            
        # 5. Elitism check: ensure best is in next pop? 
        # Our elitism step above kept top 5 of *previous* pop.
        # If the new best is in the new pop, it will be kept if it's top 5 in next iter.
        # To be safe, we can force the global best into the next population.
        # But we already updated best_radii. We can inject it.
        # Actually, the loop structure replaces pop. 
        # Let's inject the global best into the next population's slot 0 to ensure it's never lost.
        # We do this at the start of the next iteration's selection? No, just overwrite pop[0] before next eval?
        # Simpler: After creating new_pop, set new_pop[0] = best_radii.
        # Let's adjust the loop to do this.

    # Construct final points
    final_points = np.column_stack([
        best_radii * np.cos(angles),
        best_radii * np.sin(angles)
    ])
    
    return final_points
