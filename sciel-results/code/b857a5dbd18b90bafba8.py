<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Uses a hybrid approach: initial placement via grid, then local optimization using
    gradient-free methods to push circles apart and grow radii.
    """
    n = N
    # Initial placement: use a grid-like structure to avoid immediate overlaps
    # We'll use a simple heuristic initial placement
    side = int(np.ceil(np.sqrt(n)))
    # Create grid positions
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
                # Initial radius: small enough to not overlap with neighbors
                r = 0.5 / side * 0.9
                radii.append(r)
    
    # Ensure we have exactly n circles
    while len(positions) < n:
        positions.append([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
        radii.append(0.05)
    
    positions = np.array(positions[:n])
    radii = np.array(radii[:n])
    
    # Optimization: use a penalty-based approach with local search
    # We'll define a cost function that penalizes overlaps and boundary violations
    # and rewards larger radii
    
    def cost_and_grad(params):
        """
        params: array of shape (2*n,) where first n are x-coords, next n are y-coords
        We'll optimize positions and use a separate radius growth strategy.
        """
        x = params[:n]
        y = params[n:2*n]
        
        # Boundary penalty: circles must be inside [0,1]^2
        # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # For now, we'll use a fixed radius estimate and optimize positions
        # Then we'll compute max possible radii given positions
        
        # Compute pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # For each pair, the constraint is r_i + r_j <= dist_ij
        # We want to maximize sum of radii subject to these constraints
        # and boundary constraints
        
        # Use a simple approach: assume equal radii first, then refine
        # For equal radii r: 2r <= min_dist, and r <= min(x, 1-x, y, 1-y)
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                if dist[i, j] < min_dist:
                    min_dist = dist[i, j]
        
        # Boundary constraint
        boundary_min = np.min(np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
        
        # Max equal radius
        r_equal = min(min_dist / 2, boundary_min)
        r_equal = max(r_equal, 1e-6)
        
        # Penalty for overlaps if we try to use r_equal
        # Actually, let's just return the positions and compute radii separately
        
        # For optimization, we want to maximize the minimum distance between circles
        # and keep them away from boundaries
        # Let's define a cost that we minimize:
        # cost = -sum of log(min_dist_pair) + boundary_penalty
        
        # Simpler: minimize sum of overlap penalties
        cost = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # If circles overlap, penalize
                # We'll use a soft penalty
                d = dist[i, j]
                # Target: d >= 2*r, but r is unknown
                # Let's use a heuristic: penalize if d < 0.1 (too close)
                if d < 0.05:
                    cost += (0.05 - d)**2
                elif d < 0.1:
                    cost += (0.1 - d)**2 * 0.1
        
        # Boundary penalty
        for i in range(n):
            if x[i] < 0.05:
                cost += (0.05 - x[i])**2
            if x[i] > 0.95:
                cost += (x[i] - 0.95)**2
            if y[i] < 0.05:
                cost += (0.05 - y[i])**2
            if y[i] > 0.95:
                cost += (y[i] - 0.95)**2
        
        return cost
    
    # Use Nelder-Mead or Powell to optimize positions
    # We'll do a few rounds of optimization
    
    best_params = np.concatenate([positions[:, 0], positions[:, 1]])
    best_cost = cost_and_grad(best_params)
    
    # Try multiple restarts
    for restart in range(5):
        # Perturb initial positions
        if restart == 0:
            params = best_params.copy()
        else:
            params = best_params + np.random.randn(2*n) * 0.1
            params = np.clip(params, 0.01, 0.99)
        
        result = minimize(cost_and_grad, params, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x
    
    x = best_params[:n]
    y = best_params[n:2*n]
    
    # Now compute optimal radii given the positions
    # This is a linear programming problem: maximize sum(r_i) subject to:
    # r_i + r_j <= dist_ij for all i < j
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # r_i >= 0
    
    # Use a simple iterative approach: start with small radii and grow them
    # until constraints are violated
    
    radii = np.ones(n) * 0.01
    # Compute distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary constraints
    boundary_r = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    
    # Iteratively grow radii
    for iteration in range(100):
        updated = False
        for i in range(n):
            # Max radius for circle i given other circles
            max_r = boundary_r[i]
            for j in range(n):
                if i != j:
                    max_r = min(max_r, (dist[i, j] - radii[j]))
            max_r = max(max_r, 0.0)
            
            # Try to increase radius
            new_r = min(radii[i] * 1.1, max_r)
            if new_r > radii[i]:
                radii[i] = new_r
                updated = True
        
        if not updated:
            break
    
    # Final check: ensure no overlaps
    final_radii = np.zeros(n)
    for i in range(n):
        max_r = boundary_r[i]
        for j in range(n):
            if i != j:
                max_r = min(max_r, (dist[i, j] - final_radii[j]) / 2)
        final_radii[i] = max(max_r, 0.0)
    
    # Verify and adjust
    for i in range(n):
        for j in range(i+1, n):
            d = dist[i, j]
            if final_radii[i] + final_radii[j] > d:
                # Scale down
                scale = d / (final_radii[i] + final_radii[j]) * 0.999
                final_radii[i] *= scale
                final_radii[j] *= scale
    
    # Clip to boundary
    for i in range(n):
        final_radii[i] = min(final_radii[i], boundary_r[i])
    
    return np.column_stack([x, y, final_radii])
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import minimize


def compute_optimal_radii(x, y):
    """
    Given positions (x, y), compute the maximum radii for each circle
    subject to non-overlap and boundary constraints.
    Uses iterative relaxation.
    """
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary constraints: max radius for each circle
    boundary_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Initialize radii
    radii = np.minimum(boundary_r, 0.05)
    
    # Iteratively relax: for each circle, set radius to min of constraints
    for _ in range(200):
        new_radii = boundary_r.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    # r_i + r_j <= dist[i,j] => r_i <= dist[i,j] - r_j
                    max_r = dist[i, j] - radii[j]
                    new_radii[i] = min(new_radii[i], max(0.0, max_r))
        
        # Check convergence
        if np.allclose(radii, new_radii, atol=1e-8):
            radii = new_radii
            break
        radii = new_radii
    
    # Ensure feasibility with a final pass
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if radii[i] + radii[j] > dist[i, j] + 1e-10:
                    # Scale down both
                    scale = dist[i, j] / (radii[i] + radii[j]) * 0.999
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    
    # Final clip to boundary
    radii = np.minimum(radii, boundary_r)
    radii = np.maximum(radii, 0.0)
    
    return radii


def evaluate_solution(x, y):
    """
    Evaluate a solution: return (sum_of_radii, overlap_penalty, boundary_penalty)
    Lower is better for penalties, higher is better for sum_of_radii.
    """
    n = len(x)
    
    # Compute distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Compute optimal radii
    radii = compute_optimal_radii(x, y)
    sum_radii = np.sum(radii)
    
    # Check for overlaps (should be none if radii are computed correctly)
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if radii[i] + radii[j] > dist[i, j] + 1e-10:
                overlap_penalty += (radii[i] + radii[j] - dist[i, j]) ** 2
    
    # Boundary violations
    boundary_penalty = 0.0
    for i in range(n):
        boundary_penalty += max(0, radii[i] - x[i]) ** 2
        boundary_penalty += max(0, radii[i] - (1 - x[i])) ** 2
        boundary_penalty += max(0, radii[i] - y[i]) ** 2
        boundary_penalty += max(0, radii[i] - (1 - y[i])) ** 2
    
    return sum_radii, overlap_penalty, boundary_penalty


def nsga2_packing(n_circles, pop_size=100, n_generations=50, mutation_rate=0.1):
    """
    NSGA-II for circle packing.
    """
    n = n_circles
    
    def random_solution():
        """Generate a random initial solution."""
        # Use a grid-based initialization with some randomness
        side = int(np.ceil(np.sqrt(n)))
        x = np.zeros(n)
        y = np.zeros(n)
        
        # Place circles in a grid pattern with jitter
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < n:
                    x[idx] = (i + 0.5) / side + random.uniform(-0.02, 0.02)
                    y[idx] = (j + 0.5) / side + random.uniform(-0.02, 0.02)
                    idx += 1
        
        # Add any remaining circles randomly
        while idx < n:
            x[idx] = random.uniform(0.1, 0.9)
            y[idx] = random.uniform(0.1, 0.9)
            idx += 1
        
        # Clip to valid range
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        return x, y
    
    def crossover(parent1, parent2):
        """Uniform crossover."""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        mask = np.random.rand(2 * n) < 0.5
        child1[:n] = np.where(mask[:n], parent1[:n], parent2[:n])
        child1[n:] = np.where(mask[n:], parent1[n:], parent2[n:])
        child2[:n] = np.where(mask[:n], parent2[:n], parent1[:n])
        child2[n:] = np.where(mask[n:], parent2[n:], parent1[n:])
        return child1, child2
    
    def mutate(params, rate=mutation_rate):
        """Gaussian mutation."""
        child = params.copy()
        mutation_mask = np.random.rand(len(child)) < rate
        if np.any(mutation_mask):
            child[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.05
            child = np.clip(child, 0.01, 0.99)
        return child
    
    def dominates(a, b):
        """
        Check if solution a dominates solution b.
        a = (sum_radii, overlap_penalty, boundary_penalty)
        Higher sum_radii is better, lower penalties are better.
        """
        # a dominates b if a is at least as good in all objectives and strictly better in at least one
        if (a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2] and
                (a[0] > b[0] or a[1] < b[1] or a[2] < b[2])):
            return True
        return False
    
    def non_dominated_sort(population, objectives):
        """
        Non-dominated sorting.
        population: list of parameter arrays
        objectives: list of (sum_radii, overlap, boundary) tuples
        Returns: list of fronts, each front is a list of indices
        """
        n_pop = len(population)
        domination_count = np.zeros(n_pop, dtype=int)
        dominated_set = [set() for _ in range(n_pop)]
        fronts = []
        
        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                if dominates(objectives[i], objectives[j]):
                    dominated_set[i].add(j)
                    domination_count[j] += 1
                elif dominates(objectives[j], objectives[i]):
                    dominated_set[j].add(i)
                    domination_count[i] += 1
        
        # First front: non-dominated solutions
        current_front = [i for i in range(n_pop) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        # If there are still unassigned solutions, add them to the last front
        assigned = set()
        for front in fronts:
            for idx in front:
                assigned.add(idx)
        remaining = [i for i in range(n_pop) if i not in assigned]
        if remaining:
            fronts.append(remaining)
        
        return fronts
    
    def crowding_distance_assignment(front, objectives):
        """
        Assign crowding distances to solutions in a front.
        Returns a list of crowding distances.
        """
        n_front = len(front)
        if n_front <= 2:
            return [float('inf')] * n_front
        
        crowding = np.zeros(n_front)
        n_obj = 3  # sum_radii, overlap, boundary
        
        for m in range(n_obj):
            # Sort by m-th objective
            sorted_indices = sorted(range(n_front), key=lambda i: objectives[front[i]][m])
            crowding[sorted_indices[0]] = float('inf')
            crowding[sorted_indices[-1]] = float('inf')
            
            obj_range = objectives[front[sorted_indices[-1]]][m] - objectives[front[sorted_indices[0]]][m]
            if obj_range == 0:
                continue
            
            for i in range(1, n_front - 1):
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                crowding[sorted_indices[i]] += (
                    objectives[front[next_idx]][m] - objectives[front[prev_idx]][m]
                ) / obj_range
        
        return crowding
    
    # Initialize population
    population = []
    objectives = []
    for _ in range(pop_size):
        x, y = random_solution()
        params = np.concatenate([x, y])
        population.append(params)
        sum_r, overlap, boundary = evaluate_solution(x, y)
        objectives.append((sum_r, overlap, boundary))
    
    best_sum_r = max(obj[0] for obj in objectives)
    best_params = None
    for i, obj in enumerate(objectives):
        if obj[0] == best_sum_r and obj[1] < 1e-6 and obj[2] < 1e-6:
            best_params = population[i]
            break
    
    # Evolution loop
    for gen in range(n_generations):
        # Combine parents and offspring
        offspring = []
        offspring_obj = []
        
        # Selection and reproduction
        while len(offspring) < pop_size:
            # Tournament selection
            t1 = random.randint(0, len(population) - 1)
            t2 = random.randint(0, len(population) - 1)
            
            # Compare using non-dominated sorting + crowding
            # For simplicity, use a simple comparison
            if objectives[t1][0] > objectives[t2][0]:
                parent1 = population[t1]
            else:
                parent1 = population[t2]
            
            t3 = random.randint(0, len(population) - 1)
            t4 = random.randint(0, len(population) - 1)
            if objectives[t3][0] > objectives[t4][0]:
                parent2 = population[t3]
            else:
                parent2 = population[t4]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            x1, y1 = child1[:n], child1[n:]
            x2, y2 = child2[:n], child2[n:]
            
            obj1 = evaluate_solution(x1, y1)
            obj2 = evaluate_solution(x2, y2)
            
            offspring.append(child1)
            offspring_obj.append(obj1)
            if len(offspring) < pop_size:
                offspring.append(child2)
                offspring_obj.append(obj2)
        
        # Combine population and offspring
        combined_pop = population + offspring
        combined_obj = objectives + offspring_obj
        
        # Non-dominated sorting
        fronts = non_dominated_sort(combined_pop, combined_obj)
        
        # Select next generation
        new_population = []
        new_objectives = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                for idx in front:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_obj[idx])
            else:
                # Need to select from this front using crowding distance
                needed = pop_size - len(new_population)
                crowding = crowding_distance_assignment(front, combined_obj)
                # Sort by crowding distance (descending)
                sorted_front = sorted(front, key=lambda i: crowding[front.index(i)], reverse=True)
                for idx in sorted_front[:needed]:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_obj[idx])
                break
        
        population = new_population
        objectives = new_objectives
        
        # Update best
        for i, obj in enumerate(objectives):
            if obj[0] > best_sum_r and obj[1] < 1e-6 and obj[2] < 1e-6:
                best_sum_r = obj[0]
                best_params = population[i]
    
    # Final local optimization on the best solution
    if best_params is not None:
        x = best_params[:n]
        y = best_params[n:]
        
        def local_cost(params):
            x_l = params[:n]
            y_l = params[n:2*n]
            sum_r, overlap, boundary = evaluate_solution(x_l, y_l)
            # Minimize negative sum_radii + penalties
            return -sum_r + 10 * overlap + 10 * boundary
        
        result = minimize(local_cost, best_params, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
        
        if result.fun < local_cost(best_params):
            best_params = result.x
    
    x = best_params[:n]
    y = best_params[n:2*n]
    radii = compute_optimal_radii(x, y)
    
    return np.column_stack([x, y, radii])


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Uses NSGA-II evolutionary algorithm with local refinement.
    """
    n = N
    result = nsga2_packing(n, pop_size=80, n_generations=40, mutation_rate=0.1)
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
