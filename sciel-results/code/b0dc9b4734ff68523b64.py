"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(h):
    """Compute the value to minimize: max correlate(h, 1-h) / N * 2"""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _is_valid(h):
    """Check constraints: h in [0,1], sum close to N/2, finite"""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True

def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-13:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) > 1e-11:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with seed
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    # Generate diverse initial population
    np.random.seed(42)
    pop = [best_h.copy()]
    
    # Add varied initial candidates
    for _ in range(50):
        # Random pattern with sum constraint
        h = np.random.uniform(0, 1, N)
        h = _project(h)
        if _is_valid(h):
            pop.append(h)
    
    # Add structured patterns
    # Sawtooth patterns
    for period in [2, 3, 4, 5, 7, 10]:
        h = np.zeros(N)
        for i in range(N):
            phase = (i % period) / period
            h[i] = 1.0 - 2.0 * abs(phase - 0.5)
        h = _project(h)
        if _is_valid(h):
            pop.append(h)
    
    # Binary-like patterns
    for density in [0.3, 0.4, 0.5]:
        for shift in range(5):
            h = np.zeros(N)
            target_ones = int(N * density)
            indices = np.sort(np.random.choice(N, target_ones, replace=False))
            h[indices] = 1.0
            h = _project(h)
            if _is_valid(h):
                pop.append(h)
    
    # Evaluate initial population
    for i, h in enumerate(pop):
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Genetic algorithm with deadline guard
    pop_size = 100
    if len(pop) < pop_size:
        while len(pop) < pop_size:
            h = np.random.uniform(0, 1, N)
            h = _project(h)
            if _is_valid(h):
                pop.append(h)
            else:
                pop.append(best_h.copy())
    
    pop = pop[:pop_size]
    pop_vals = [_objective(h) for h in pop]
    best_val = min(pop_vals)
    best_idx = np.argmin(pop_vals)
    best_h = pop[best_idx].copy()
    
    # Evolutionary loop
    elite_size = 10
    mutation_rate = 0.3
    crossover_rate = 0.5
    
    while time.time() < deadline:
        # Sort by fitness
        order = np.argsort(pop_vals)
        elites = [pop[i].copy() for i in order[:elite_size]]
        elite_vals = [pop_vals[i] for i in order[:elite_size]]
        
        # Update best
        if elite_vals[0] < best_val:
            best_val = elite_vals[0]
            best_h = elites[0].copy()
        
        # Generate new population
        new_pop = []
        new_vals = []
        
        # Keep elites
        for i in range(elite_size):
            new_pop.append(elites[i])
            new_vals.append(elite_vals[i])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 5.0:
                break
            
            r = np.random.random()
            if r < crossover_rate and len(elites) >= 2:
                # Crossover
                i1 = np.random.randint(0, elite_size)
                i2 = np.random.randint(0, elite_size)
                if i1 == i2:
                    i2 = (i2 + 1) % elite_size
                cross_point = np.random.randint(1, N)
                child = np.concatenate([elites[i1][:cross_point], elites[i2][cross_point:]])
            else:
                # Mutation from a random elite
                i = np.random.randint(0, elite_size)
                child = elites[i].copy()
            
            # Mutation
            if np.random.random() < mutation_rate:
                # Gaussian perturbation
                sigma = 0.1 * (1.0 - len(new_pop) / pop_size)
                child = child + np.random.normal(0, sigma, N)
            
            # Enforce constraints
            child = _project(child)
            
            if _is_valid(child):
                val = _objective(child)
                new_pop.append(child)
                new_vals.append(val)
            else:
                # Fallback to best
                new_pop.append(best_h.copy())
                new_vals.append(best_val)
        
        pop = new_pop
        pop_vals = new_vals
        
        # Check for convergence
        if len(set([round(v, 10) for v in pop_vals[:elite_size]])) == 1:
            # Try a larger perturbation
            for i in range(elite_size):
                h = elites[i] + np.random.normal(0, 0.3, N)
                h = _project(h)
                if _is_valid(h):
                    val = _objective(h)
                    if val < pop_vals[i]:
                        pop[i] = h
                        pop_vals[i] = val
    
    # L-BFGS-B refinement on best candidates
    if time.time() < deadline - 20.0:
        # Use L-BFGS-B to refine the best solution
        def smooth_objective(x):
            h = _project(x)
            corr = np.correlate(h, 1.0 - h, mode="full")
            # Use log-sum-exp for smooth max
            beta = 100.0
            max_c = np.max(corr)
            if max_c < -1000:
                return 0.0
            return np.log(np.sum(np.exp(beta * (corr - max_c)))) / beta + max_c - np.log(len(corr))
        
        # Refine best_h
        res = minimize(smooth_objective, best_h, method='L-BFGS-B', 
                      bounds=[(0, 1)] * N,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        refined_h = _project(res.x)
        if _is_valid(refined_h):
            refined_val = _objective(refined_h)
            if refined_val < best_val:
                best_val = refined_val
                best_h = refined_h.copy()
        
        # Also try refining top few elites
        for i in range(min(5, len(elites))):
            if time.time() > deadline - 10.0:
                break
            res = minimize(smooth_objective, elites[i], method='L-BFGS-B',
                          bounds=[(0, 1)] * N,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            refined_h = _project(res.x)
            if _is_valid(refined_h):
                refined_val = _objective(refined_h)
                if refined_val < best_val:
                    best_val = refined_val
                    best_h = refined_h.copy()
    
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
