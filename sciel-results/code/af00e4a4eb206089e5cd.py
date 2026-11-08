import numpy as np
import time

N = 600

def _score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(conv(h, h)) / (sum(h))^2
    """
    # h is a numpy array of length N
    # Autoconvolution
    c = np.convolve(h, h, mode='full')
    max_c = np.max(c)
    sum_h = np.sum(h)
    if sum_h == 0:
        return float('inf')
    return max_c / (sum_h * sum_h)

def _optimize(h_init, time_limit=140.0):
    """
    Perform local search to minimize the score starting from h_init.
    """
    h = np.array(h_init, dtype=float)
    # Normalize to sum 1 to keep numbers stable, though score is scale invariant
    s = np.sum(h)
    if s > 0:
        h = h / s
    
    current_score = _score(h)
    best_h = h.copy()
    best_score = current_score
    
    start_time = time.time()
    sigma = 0.1
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        # Use a mix of small and large steps
        if no_improve_count > 100:
            sigma *= 1.5
            no_improve_count = 0
        else:
            sigma = max(0.01, sigma * 0.95)
            
        noise = np.random.normal(0, sigma, N)
        h_new = h + noise
        h_new = np.maximum(h_new, 0) # Enforce non-negativity
        
        # Optional: Re-normalize? 
        # Since score is scale invariant, we don't strictly need to, 
        # but it helps with numerical stability if values get too small/large.
        # However, changing scale doesn't change score. 
        # So we can just evaluate as is.
        
        new_score = _score(h_new)
        
        if new_score < current_score:
            h = h_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # Simple simulated annealing acceptance?
            # If we accept worse solutions, we might escape local minima.
            # P = exp(-(new_score - current_score) / T)
            # But we don't have a temperature. 
            # Let's stick to hill climbing with occasional restarts or larger jumps.
            # The sigma increase handles the "jump" part.
            
    return best_h, best_score

def construct():
    """
    Construct the step function heights.
    """
    start_time = time.time()
    hard_deadline = 145.0 # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Try different initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian-like (centered)
    x = np.linspace(-0.5, 0.5, N)
    inits.append(np.exp(-x**2 / 0.1))
    
    # 3. Cosine bump
    inits.append(np.cos(np.pi * x)) # x in [-0.5, 0.5] -> cos is positive
    
    # 4. Random
    inits.append(np.random.rand(N))
    
    # 5. Two peaks
    inits.append(np.exp(-(x-0.2)**2/0.05) + np.exp(-(x+0.2)**2/0.05))
    
    for h_init in inits:
        if time.time() - start_time > hard_deadline - 10:
            break
            
        # Allocate time for each optimization run
        # If we have 5 inits, and 145s total, we can spend ~20s each.
        # But we should stop early if we find a very good score.
        # Let's run each for a fixed short time, e.g., 5-10 seconds, or until deadline.
        
        # Simple strategy: Run all inits for a short burst, keep best.
        # Then take the best and run it longer.
        
        h_opt, score_opt = _optimize(h_init, time_limit=5.0)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
            
    # If we have time left, refine the best candidate further
    if best_h is not None:
        remaining_time = hard_deadline - (time.time() - start_time)
        if remaining_time > 1.0:
            # Run optimization again on the best found so far with remaining time
            # Note: _optimize restarts from the given init. 
            # We can pass best_h as init.
            h_opt, score_opt = _optimize(best_h, time_limit=remaining_time - 1.0)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
                
    if best_h is None:
        return [1.0] * N
        
    # Return as a list of floats
    return [float(x) for x in best_h]

