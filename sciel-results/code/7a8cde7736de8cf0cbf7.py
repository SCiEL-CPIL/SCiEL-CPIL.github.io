"""Differential Evolution based circle packing in perimeter-4 rectangle."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def compute_radii(params):
        """
        params: array of shape (43,)
        [0]: W (width), H = 2 - W
        [1:43]: 21 x (x, y) positions
        
        Returns: sum of maximal feasible radii
        """
        W = params[0]
        H = 2.0 - W
        
        # Ensure valid rectangle
        if W <= 0.1 or H <= 0.1:
            return -1.0
            
        pts = params[1:].reshape(N, 2)
        
        # Check if points are inside the rectangle
        # If a point is outside, penalty
        for i in range(N):
            if pts[i, 0] < 0 or pts[i, 0] > W or pts[i, 1] < 0 or pts[i, 1] > H:
                return -1.0
        
        radii = np.zeros(N)
        
        for i in range(N):
            # Distance to walls
            d_wall = min(pts[i, 0], W - pts[i, 0], pts[i, 1], H - pts[i, 1])
            r_max = d_wall
            
            # Distance to other circles
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                # The radius of circle i is limited by half the distance to circle j
                # IF circle j has a radius such that they touch.
                # Actually, the maximal radius of circle i given fixed positions of others is:
                # r_i <= d_ij - r_j
                # But we are optimizing all radii simultaneously?
                # No, the objective is "maximize the sum of their radii".
                # For a given set of centers, the maximal sum of radii is not simply sum of (min dist to wall / 2 or neighbors).
                # Wait, if we fix centers, the maximal radius of circle i is min(d_wall, min_j (d_ij - r_j)).
                # This is a system of inequalities.
                # However, a common approximation or specific case is that we can set r_i = min(d_wall, min_j d_ij / 2).
                # Is this optimal?
                # If we set r_i = min(d_wall, min_j d_ij / 2), then for any pair i, j:
                # r_i + r_j <= d_ij/2 + d_ij/2 = d_ij. So they don't overlap.
                # And r_i <= d_wall. So they are inside.
                # Is it possible to get a larger sum?
                # Suppose two circles are far from walls but close to each other.
                # r_i = d/2, r_j = d/2. Sum = d.
                # If we made r_i smaller, we could make r_j larger? No, r_j is limited by d - r_i.
                # If r_i < d/2, r_j can be > d/2? No, r_j is also limited by its own walls and other neighbors.
                # Actually, for a fixed set of centers, the problem of maximizing sum r_i subject to r_i + r_j <= d_ij and r_i <= w_i is a linear programming problem.
                # The greedy assignment r_i = min(w_i, min_j d_ij/2) is NOT always optimal for the sum.
                # Example: 3 circles in a line. A-B-C. d_AB=1, d_BC=1. Walls far away.
                # Greedy: r_A=0.5, r_B=0.5, r_C=0.5. Sum=1.5.
                # LP: Max r_A+r_B+r_C.
                # r_A+r_B <= 1, r_B+r_C <= 1.
                # If r_B=0, r_A=1, r_C=1. Sum=2.
                # So the "half distance" heuristic is suboptimal.
                
                # However, implementing an LP solver inside DE for 21 circles (21 vars, 210 constraints) might be slow but feasible?
                # SciPy has linprog.
                # But DE evaluates hundreds/thousands of candidates. 1000 * LP might be too slow.
                
                # Alternative: Use the half-distance heuristic as a proxy. It's a lower bound.
                # Or, use a simpler relaxation: r_i = min(w_i, min_j d_ij / 2).
                # Let's stick to the lower bound for speed, but maybe refine it?
                # Actually, the "inspiration" mentioned differential evolution achieved 1.01.
                # Let's try to be slightly smarter or just accept the lower bound if it's fast enough.
                
                # Let's calculate the lower bound radius first.
                min_dist_neighbor = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(pts[i] - pts[j])
                    min_dist_neighbor = min(min_dist_neighbor, d)
                
                r_lower = min(d_wall, min_dist_neighbor / 2.0)
                radii[i] = r_lower
            
        return np.sum(radii)

    def evaluate_individual(ind):
        """
        ind: array of shape (43,)
        """
        W = ind[0]
        H = 2.0 - W
        if W < 0.1 or H < 0.1:
            return -1.0
        
        pts = ind[1:].reshape(N, 2)
        
        # Clip points to valid range [0, W] x [0, H]
        # This ensures circles are at least partially inside, though radius will be 0 if on boundary.
        # Actually, if we clip, the "distance to wall" becomes 0 for points on the edge.
        # It's better to penalize or just let the radius be small.
        # Let's just compute radii based on actual positions, even if slightly out?
        # No, circles must be inside.
        # If a center is outside, the circle cannot be placed.
        # So we should restrict the search space to valid centers.
        
        # In DE, we can repair the individual.
        # Repair: Clip x to [0, W], y to [0, H].
        pts[:, 0] = np.clip(pts[:, 0], 0, W)
        pts[:, 1] = np.clip(pts[:, 1], 0, H)
        
        # Recalculate radii
        radii = np.zeros(N)
        for i in range(N):
            d_wall = min(pts[i, 0], W - pts[i, 0], pts[i, 1], H - pts[i, 1])
            r_max = d_wall
            
            min_dist_neighbor = np.inf
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                min_dist_neighbor = min(min_dist_neighbor, d)
            
            r_max = min(r_max, min_dist_neighbor / 2.0)
            radii[i] = r_max
            
        return np.sum(radii)

    # Differential Evolution Parameters
    pop_size = 50
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    # Initial Population
    # W in [0.5, 1.5] (since H=2-W, and symmetry W<->H, we can restrict to W<=1 without loss of generality? 
    # No, the rectangle is not square necessarily, but W and H are symmetric in the problem definition 
    # except for the aspect ratio. We can search W in [0.5, 1.5].
    # Actually, due to symmetry, W in [0.5, 1.0] is sufficient if we allow swapping? 
    # The objective function is symmetric under W->H, x->y.
    # So we can restrict W to [0.5, 1.0].
    
    pop = np.zeros((pop_size, 43))
    for i in range(pop_size):
        # Random W
        pop[i, 0] = np.random.uniform(0.5, 1.0)
        # Random positions
        W = pop[i, 0]
        H = 2.0 - W
        pop[i, 1::2] = np.random.uniform(0, W, N)
        pop[i, 2::2] = np.random.uniform(0, H, N)
        
    # Evaluate initial population
    fit = np.array([evaluate_individual(ind) for ind in pop])
    
    best_fit = np.max(fit)
    best_idx = np.argmax(fit)
    best_ind = pop[best_idx].copy()
    
    max_gens = 200
    gen = 0
    
    while gen < max_gens:
        if time.time() > deadline - 10:
            break
            
        new_pop = np.zeros_like(pop)
        new_fit = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Select 3 distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = pop[i].copy()
            # Random j0
            j0 = np.random.randint(0, 43)
            for j in range(43):
                if np.random.uniform() < CR or j == j0:
                    trial[j] = mutant[j]
            
            # Repair trial individual
            W = trial[0]
            if W < 0.5 or W > 1.0:
                trial[0] = np.clip(trial[0], 0.5, 1.0)
                W = trial[0]
            H = 2.0 - W
            pts = trial[1:].reshape(N, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, W)
            pts[:, 1] = np.clip(pts[:, 1], 0, H)
            trial[1:] = pts.flatten()
            
            # Evaluate
            f_trial = evaluate_individual(trial)
            
            # Selection
            if f_trial >= fit[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
            else:
                new_pop[i] = pop[i]
                new_fit[i] = fit[i]
                
        pop = new_pop
        fit = new_fit
        
        current_best = np.max(fit)
        if current_best > best_fit:
            best_fit = current_best
            best_idx = np.argmax(fit)
            best_ind = pop[best_idx].copy()
            
        gen += 1
        
    # Construct final output
    # best_ind[0] is W
    W = best_ind[0]
    H = 2.0 - W
    pts = best_ind[1:].reshape(N, 2)
    
    # Final radius calculation (same as in evaluate)
    radii = np.zeros(N)
    for i in range(N):
        d_wall = min(pts[i, 0], W - pts[i, 0], pts[i, 1], H - pts[i, 1])
        r_max = d_wall
        min_dist_neighbor = np.inf
        for j in range(N):
            if i == j:
                continue
            d = np.linalg.norm(pts[i] - pts[j])
            min_dist_neighbor = min(min_dist_neighbor, d)
        r_max = min(r_max, min_dist_neighbor / 2.0)
        radii[i] = r_max
        
    result = np.column_stack([pts, radii])
    return result

# EVOLVE-BLOCK-END
