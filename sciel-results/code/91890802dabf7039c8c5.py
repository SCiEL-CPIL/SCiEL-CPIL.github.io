"""
Strategy: Particle Swarm Optimization (PSO) with Diverse Analytical Initialization.

Rationale:
1. The current program uses L-BFGS-B, a deterministic local gradient-based method. 
   This often gets trapped in local minima for non-convex problems like this one.
2. PSO is a population-based stochastic global optimization algorithm. It explores 
   the search space using a swarm of particles that adjust their positions based 
   on their own best-known position and the global best-known position.
3. To ensure efficiency within the 120s time limit, we use a small swarm size (20) 
   and a limited number of iterations (200), relying on vectorized NumPy operations 
   for the evaluation of the objective function.
4. We initialize the swarm with a mix of random particles and particles based on 
   known smooth window functions (Gaussian, Triangular, Bimodal) to provide 
   diverse starting points that cover different basins of attraction.
5. The non-negativity constraint is handled by clipping the particle positions 
   to [0, infinity] after each update.
"""

import numpy as np
import time

N = 50

def evaluate(h):
    """Compute the value for a given set of step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Parameters for PSO
    swarm_size = 20
    max_iter = 200
    w_inertia = 0.7
    c1 = 1.5
    c2 = 1.5
    
    # Initialize particles with diverse strategies
    positions = np.zeros((swarm_size, N))
    velocities = np.zeros((swarm_size, N))
    personal_best_pos = np.zeros((swarm_size, N))
    personal_best_val = np.full(swarm_size, -np.inf)
    
    global_best_pos = np.zeros(N)
    global_best_val = -np.inf
    
    # Generate diverse initial positions
    x = np.linspace(0, 1, N)
    
    # 1. Uniform
    positions[0] = np.ones(N)
    
    # 2. Triangular
    positions[1] = 1 - np.abs(2*x - 1)
    
    # 3. Gaussian
    positions[2] = np.exp(-10*(x-0.5)**2)
    
    # 4. Bimodal
    positions[3] = np.exp(-20*(x-0.3)**2) + np.exp(-20*(x-0.7)**2)
    
    # 5. Cosine bump
    positions[4] = 0.5 * (1 + np.cos(2*np.pi*x))
    
    # 6-20. Random
    for i in range(5, swarm_size):
        positions[i] = np.random.uniform(0, 1, N)
    
    # Clip initial positions to non-negative
    positions = np.clip(positions, 0, None)
    
    # Evaluate initial population
    for i in range(swarm_size):
        val = evaluate(positions[i])
        personal_best_val[i] = val
        personal_best_pos[i] = positions[i].copy()
        if val > global_best_val:
            global_best_val = val
            global_best_pos = positions[i].copy()
    
    # PSO Loop
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        for i in range(swarm_size):
            # Update velocity
            r1 = np.random.rand(N)
            r2 = np.random.rand(N)
            
            velocities[i] = (w_inertia * velocities[i] + 
                             c1 * r1 * (personal_best_pos[i] - positions[i]) + 
                             c2 * r2 * (global_best_pos - positions[i]))
            
            # Update position
            positions[i] = positions[i] + velocities[i]
            
            # Enforce non-negativity
            positions[i] = np.clip(positions[i], 0, None)
            
            # Evaluate new position
            val = evaluate(positions[i])
            
            # Update personal best
            if val > personal_best_val[i]:
                personal_best_val[i] = val
                personal_best_pos[i] = positions[i].copy()
                
                # Update global best
                if val > global_best_val:
                    global_best_val = val
                    global_best_pos = positions[i].copy()
        
        # Adaptive inertia weight (optional, but helps convergence)
        # w_inertia = 0.9 - (it / max_iter) * 0.5
        # We keep it fixed for simplicity and stability in this short run
        
    # Final check: ensure the best solution is valid
    if global_best_val < 0:
        # Fallback to uniform
        global_best_pos = np.ones(N)
        global_best_val = evaluate(global_best_pos)
        
    return list(global_best_pos)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
