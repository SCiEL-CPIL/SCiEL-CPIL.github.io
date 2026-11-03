import time
import numpy as np
from scipy.optimize import differential_evolution

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of autoconvolution."""
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s == 0:
        return np.inf
    return np.max(conv) / (s * s)

def pulses_to_step(pulse_params):
    """Convert pulse parameters to step function.
    
    pulse_params: array of shape (n_pulses, 3) where each row is [center, width, height]
    center: position in [0, 1) (normalized)
    width: half-width in [0.001, 0.2]
    height: in [0, 2]
    """
    f = np.zeros(N)
    k = np.arange(N) / N  # normalized positions in [0, 1)
    
    for i in range(pulse_params.shape[0]):
        center = pulse_params[i, 0]
        width = pulse_params[i, 1]
        height = pulse_params[i, 2]
        
        # Rectangular pulse: 1 where |x - center| <= width, with wrap-around
        dist = np.abs(k - center)
        dist = np.minimum(dist, 1 - dist)  # wrap-around distance
        mask = dist <= width
        f[mask] += height
    
    return f

def objective_pulse(params_flat, n_pulses):
    """Objective function for differential evolution."""
    pulse_params = params_flat.reshape(n_pulses, 3)
    f = pulses_to_step(pulse_params)
    return compute_score(f)

def construct():
    start_time = time.time()
    
    # Start with constant function as baseline
    f_const = np.ones(N)
    best_score = compute_score(f_const)
    best_f = f_const.copy()
    
    # Try different numbers of pulses
    for n_pulses in [2, 3, 4, 5]:
        if time.time() > DEADLINE - 30:
            break
        
        # Define bounds for each pulse parameter
        # center in [0, 1), width in [0.001, 0.2], height in [0, 2]
        bounds = [(0, 1), (0.001, 0.2), (0, 2)] * n_pulses
        
        # Use differential evolution
        try:
            result = differential_evolution(
                objective_pulse,
                bounds=bounds,
                args=(n_pulses,),
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,  # We'll do our own refinement
                init='sobol'
            )
            
            if result.success or result.fun < best_score:
                f_found = pulses_to_step(result.fun.reshape(n_pulses, 3) if hasattr(result.fun, 'reshape') else result.x.reshape(n_pulses, 3))
                score_found = compute_score(f_found)
                if score_found < best_score:
                    best_score = score_found
                    best_f = f_found.copy()
        except Exception:
            continue
    
    # Local refinement on best found
    if time.time() < DEADLINE - 10:
        f = best_f.copy()
        current_score = best_score
        
        # Coordinate descent refinement
        for iteration in range(50):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                old_val = f[i]
                for delta in [0.01, -0.01, 0.001, -0.001]:
                    new_val = old_val + delta
                    if new_val < 0:
                        continue
                    f[i] = new_val
                    new_score = compute_score(f)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    f[i] = old_val
            
            if not improved:
                break
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
    
    return list(best_f)

if __name__ == "__main__":
    f = construct()
    score = compute_score(np.array(f))
    print(f"Score: {score}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
