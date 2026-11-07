import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # length of g = convolve(h, h)

def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(0, h)
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    # y[i]^2 + y[i]*y[i+1] + y[i+1]^2
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def gmm_from_params(params):
    """
    Reconstruct h from GMM parameters.
    params: [amp1, cent1, width1, amp2, cent2, width2, amp3, cent3, width3, 
             amp4, cent4, width4, amp5, cent5, width5]
    """
    x = np.linspace(-1, 1, N)
    h = np.zeros(N)
    for i in range(5):
        amp = params[3*i]
        cent = params[3*i + 1]
        width = params[3*i + 2]
        # Ensure width is not too small to avoid numerical issues and sharp spikes
        width = max(width, 0.05)
        h += amp * np.exp(-((x - cent) ** 2) / (2 * width**2))
    return h

def neg_value_gmm(params):
    """Negative of compute_value for minimization of GMM parameters."""
    h = gmm_from_params(params)
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s
    
    # Parameter dimensions: 5 Gaussians * 3 params (amp, cent, width) = 15 dims
    n_params = 15
    
    rng = np.random.default_rng(42)
    
    # Initial population of GMM parameters
    pop_size = 30
    population = []
    scores = []
    
    def make_params():
        p = np.zeros(n_params)
        # Randomize amplitudes, centers, widths
        for i in range(5):
            p[3*i] = rng.uniform(0.1, 2.0)      # amp
            p[3*i+1] = rng.uniform(-0.8, 0.8)   # cent
            p[3*i+2] = rng.uniform(0.1, 0.5)    # width
        return p

    # Initialize with diverse starts
    for _ in range(pop_size):
        p = make_params()
        population.append(p)
        h = gmm_from_params(p)
        scores.append(compute_value(h))
    
    # Sort initial population
    sorted_indices = np.argsort(scores)[::-1]
    population = [population[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    best_idx = 0
    best_val = scores[0]
    best_params = population[0].copy()
    
    # Optimization loop
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Keep top half
        keep = pop_size // 2
        parents = population[:keep]
        
        children = []
        while len(children) < pop_size - keep:
            # Select two parents
            p1_idx = rng.integers(0, keep)
            p2_idx = rng.integers(0, keep)
            if p1_idx == p2_idx:
                continue
            
            p1 = parents[p1_idx]
            p2 = parents[p2_idx]
            
            # Uniform Crossover on parameters
            mask = rng.random(n_params) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            if rng.random() < 0.7:
                # Gaussian noise
                noise = rng.normal(0, 0.1, n_params)
                child += noise
            
            # Ensure valid ranges for parameters
            for i in range(5):
                child[3*i] = np.maximum(0, child[3*i])       # amp >= 0
                child[3*i+1] = np.clip(child[3*i+1], -1, 1)  # cent in [-1, 1]
                child[3*i+2] = np.maximum(0.05, child[3*i+2])# width >= 0.05
            
            # Local Refinement
            if time.time() < deadline - 15:
                try:
                    result = minimize(
                        neg_value_gmm,
                        child,
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-9}
                    )
                    child = result.x
                    # Re-apply constraints just in case optimizer drifted
                    for i in range(5):
                        child[3*i] = np.maximum(0, child[3*i])
                        child[3*i+1] = np.clip(child[3*i+1], -1, 1)
                        child[3*i+2] = np.maximum(0.05, child[3*i+2])
                except Exception:
                    pass
            
            children.append(child)
        
        # Combine
        population = parents + children
        scores = []
        for p in population:
            h = gmm_from_params(p)
            scores.append(compute_value(h))
        
        # Update best
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_val:
            best_val = scores[current_best_idx]
            best_params = population[current_best_idx].copy()
            
        # Sort and keep top
        sorted_indices = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_indices[:pop_size]]
        scores = [scores[i] for i in sorted_indices[:pop_size]]
        
    # Final refinement on best
    if time.time() < deadline - 2:
        try:
            result = minimize(
                neg_value_gmm,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            best_params = result.x
            for i in range(5):
                best_params[3*i] = np.maximum(0, best_params[3*i])
                best_params[3*i+1] = np.clip(best_params[3*i+1], -1, 1)
                best_params[3*i+2] = np.maximum(0.05, best_params[3*i+2])
        except Exception:
            pass
    
    best_h = gmm_from_params(best_params)
    return best_h.tolist()
