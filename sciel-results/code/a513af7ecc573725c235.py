"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    H = np.sqrt(3) / 2
    
    def min_area(points):
        """Calculate the minimum area of any triangle formed by 3 points."""
        min_a = float('inf')
        # N=11, 165 triangles. This loop is fast.
        for i in range(N):
            p1 = points[i]
            for j in range(i+1, N):
                p2 = points[j]
                for k in range(j+1, N):
                    p3 = points[k]
                    det = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    a = 0.5 * abs(det)
                    if a < min_a:
                        min_a = a
        return min_a

    def is_inside(points):
        """Check if all points are inside the equilateral triangle."""
        # Vectorized check
        x = points[:, 0]
        y = points[:, 1]
        # y >= 0
        if np.any(y < -1e-9): return False
        # y <= sqrt(3)*x
        if np.any(y > np.sqrt(3)*x + 1e-9): return False
        # y <= sqrt(3)*(1-x)
        if np.any(y > np.sqrt(3)*(1-x) + 1e-9): return False
        # x >= 0, x <= 1
        if np.any(x < -1e-9) or np.any(x > 1 + 1e-9): return False
        return True

    def objective(coords):
        points = coords.reshape(-1, 2)
        if not is_inside(points):
            return -1e6  # We maximize, so bad fitness
        return min_area(points)

    def project_to_triangle(points):
        """Project points to be inside the triangle."""
        # Simple clamping/projection might be tricky for corners.
        # We will rely on rejection in GA or repair.
        pass

    def random_point_in_triangle(rng):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                return np.array([u + 0.5 * v, H * v])

    def random_points(rng):
        return np.array([random_point_in_triangle(rng) for _ in range(N)])

    def heuristic_init():
        """Create a structured initial population."""
        points = []
        # Vertices
        points.append(np.array([0, 0]))
        points.append(np.array([1, 0]))
        points.append(np.array([0.5, H]))
        # Centroid
        points.append(np.array([0.5, H/3]))
        # Midpoints of sides
        points.append(np.array([0.5, 0]))
        points.append(np.array([0.25, H/2]))
        points.append(np.array([0.75, H/2]))
        # Remaining 4 points random
        rng = np.random.RandomState(0)
        for _ in range(4):
            points.append(random_point_in_triangle(rng))
        return np.array(points)

    # Genetic Algorithm Parameters
    pop_size = 40
    n_generations = 150
    crossover_rate = 0.8
    mutation_rate = 0.2
    mutation_scale = 0.05
    rng = np.random.RandomState(42)
    
    # Initialize Population
    population = []
    # Add some random
    for _ in range(pop_size // 2):
        population.append(random_points(rng))
    # Add some heuristic based
    for _ in range(pop_size // 4):
        p = heuristic_init()
        # Perturb slightly
        p += rng.normal(0, 0.05, p.shape)
        # Ensure inside
        if is_inside(p):
            population.append(p)
        else:
            population.append(random_points(rng))
    # Fill rest with random
    while len(population) < pop_size:
        population.append(random_points(rng))
        
    population = np.array(population)
    
    # Evaluate fitness
    fitness = np.array([objective(ind.flatten()) for ind in population])
    
    best_fitness = np.max(fitness)
    best_points = population[np.argmax(fitness)].copy()
    
    for gen in range(n_generations):
        new_population = []
        
        # Elitism: keep the best 2
        idxs = np.argsort(fitness)[::-1]
        for i in range(2):
            new_population.append(population[idxs[i]].copy())
        
        while len(new_population) < pop_size:
            # Tournament Selection
            def select():
                a = rng.randint(pop_size)
                b = rng.randint(pop_size)
                if fitness[a] > fitness[b]:
                    return population[a].copy()
                else:
                    return population[b].copy()
            
            parent1 = select()
            parent2 = select()
            
            # Crossover
            if rng.random() < crossover_rate:
                # Blend Crossover (BLX-alpha)
                alpha = 0.5
                lo = np.minimum(parent1, parent2)
                hi = np.maximum(parent1, parent2)
                width = hi - lo
                child = lo + (hi - lo) * rng.uniform(-alpha, 1+alpha, parent1.shape)
            else:
                child = parent1.copy()
                
            # Mutation
            if rng.random() < mutation_rate:
                mask = rng.random(child.shape) < 0.5
                child[mask] += rng.normal(0, mutation_scale, np.sum(mask))
            
            # Repair: If outside, reflect or reject?
            # Simple: If outside, replace with random point or project.
            # Projection is complex. Let's just check and if invalid, keep parent or random.
            if not is_inside(child):
                # Try to project
                # For now, if invalid, just use a random point to keep population valid
                # Or better, reflect across boundary?
                # Simplest: replace with parent1 if child is invalid? No, that stalls.
                # Let's try a simple projection:
                x = child[:, 0]
                y = child[:, 1]
                # Clamp x to [0,1]
                x = np.clip(x, 0, 1)
                # Clamp y based on x
                y_max = np.minimum(np.sqrt(3)*x, np.sqrt(3)*(1-x))
                y = np.clip(y, 0, y_max)
                child = np.column_stack((x, y))
                # Re-check, if still bad (shouldn't be with clip), fallback
                if not is_inside(child):
                    child = random_points(rng)
            
            new_population.append(child)
        
        population = np.array(new_population)
        fitness = np.array([objective(ind.flatten()) for ind in population])
        
        current_best = np.max(fitness)
        if current_best > best_fitness:
            best_fitness = current_best
            best_points = population[np.argmax(fitness)].copy()
            
    # Polish the best point with Nelder-Mead
    try:
        res = minimize(
            lambda x: -objective(x), # minimize negative of max area
            best_points.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        final_points = res.x.reshape(-1, 2)
        if is_inside(final_points):
            final_score = min_area(final_points)
            if final_score > best_fitness:
                best_fitness = final_score
                best_points = final_points
    except Exception:
        pass

    return best_points
# EVOLVE-BLOCK-END
