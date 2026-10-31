import numpy as np
from scipy.fft import fft, ifft
from scipy import optimize
import time
import random

N = 600
DEADLINE = 140

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    size = 2 * N
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*N-1])
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Initialize with uniform
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)
    
    # Stage 1: Parametric optimization with more components and different structure
    # Use a mixture of shifted, scaled cosine bumps for smoother profiles
    K = 16  # Number of components
    
    i = np.arange(N, dtype=float)
    
    def params_to_h_v2(params):
        """Convert parameter vector to height array using cosine bumps."""
        h = np.zeros(N)
        for j in range(K):
            w_j = params[4*j]       # weight
            c_j = params[4*j+1]     # center
            s_j = params[4*j+2]     # width
            phase = params[4*j+3]   # phase shift for cosine modulation
            
            if s_j < 1e-6:
                s_j = 1e-6
            
            # Cosine bump: w * cos(pi * (x-c)/s) * exp(-((x-c)/s)^2) for |x-c| < s
            x = (i - c_j) / s_j
            mask = np.abs(x) < 1.0
            if np.any(mask):
                bump = np.zeros(N)
                bump[mask] = np.cos(np.pi * x[mask]) * np.exp(-x[mask]**2)
                h += w_j * bump
        return np.maximum(h, 0)
    
    def objective_v2(params):
        """Objective function for the cosine bump parameterization."""
        if time_left() < 2:
            return 1e10
        h = params_to_h_v2(params)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        return compute_autoconv_max(h_norm)
    
    # Define bounds
    bounds = []
    for j in range(K):
        bounds.append((0.01, 5.0))    # weight
        bounds.append((0.0, N-1))     # center
        bounds.append((5.0, N/3))     # width
        bounds.append((0.0, 2*np.pi)) # phase
    
    # Generate diverse initial points
    def generate_diverse_points(n_points):
        points = []
        for _ in range(n_points):
            p = np.zeros(4*K)
            strategy = random.randint(0, 4)
            
            if strategy == 0:
                # Symmetric bimodal
                for j in range(K):
                    if j < K//2:
                        p[4*j] = random.uniform(0.5, 2.0)
                        p[4*j+1] = random.uniform(N*0.1, N*0.3)
                        p[4*j+2] = random.uniform(N/12, N/6)
                        p[4*j+3] = 0
                    else:
                        p[4*j] = random.uniform(0.5, 2.0)
                        p[4*j+1] = random.uniform(N*0.7, N*0.9)
                        p[4*j+2] = random.uniform(N/12, N/6)
                        p[4*j+3] = 0
            elif strategy == 1:
                # Multi-modal spread
                for j in range(K):
                    p[4*j] = random.uniform(0.2, 1.0)
                    p[4*j+1] = j * (N-1) / (K-1) + random.uniform(-N/15, N/15)
                    p[4*j+2] = random.uniform(N/10, N/5)
                    p[4*j+3] = random.uniform(0, 2*np.pi)
            elif strategy == 2:
                # Central peak with wings
                for j in range(K):
                    if j < K//3:
                        p[4*j] = random.uniform(1.0, 3.0)
                        p[4*j+1] = random.uniform(N*0.4, N*0.6)
                        p[4*j+2] = random.uniform(N/8, N/4)
                    elif j < 2*K//3:
                        p[4*j] = random.uniform(0.3, 1.0)
                        p[4*j+1] = random.uniform(N*0.2, N*0.8)
                        p[4*j+2] = random.uniform(N/6, N/3)
                    else:
                        p[4*j] = random.uniform(0.1, 0.5)
                        p[4*j+1] = random.uniform(0, N)
                        p[4*j+2] = random.uniform(N/4, N/2)
                    p[4*j+3] = random.uniform(0, 2*np.pi)
            elif strategy == 3:
                # Ring-like (high at edges, low in center)
                for j in range(K):
                    if j < K//2:
                        p[4*j] = random.uniform(1.0, 3.0)
                        p[4*j+1] = random.uniform(0, N*0.15)
                        p[4*j+2] = random.uniform(N/10, N/5)
                    else:
                        p[4*j] = random.uniform(1.0, 3.0)
                        p[4*j+1] = random.uniform(N*0.85, N)
                        p[4*j+2] = random.uniform(N/10, N/5)
                    p[4*j+3] = random.uniform(0, 2*np.pi)
            else:
                # Random
                for j in range(K):
                    p[4*j] = random.uniform(0.01, 3.0)
                    p[4*j+1] = random.uniform(0, N-1)
                    p[4*j+2] = random.uniform(5, N/2)
                    p[4*j+3] = random.uniform(0, 2*np.pi)
            
            points.append(p)
        return points
    
    # Run differential evolution on the cosine bump parameterization
    if time_left() > 60:
        pop_size = min(40, max(20, int(time_left() * 0.3)))
        max_iter = int(min(80, time_left() * 0.4))
        
        try:
            init_points = generate_diverse_points(pop_size)
            
            result = optimize.differential_evolution(
                objective_v2,
                bounds=bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-8,
                seed=123,
                polish=False,
                init=np.array(init_points)
            )
            
            if result.fun < best_val:
                h_de = params_to_h_v2(result.x)
                s = np.sum(h_de)
                if s > 1e-10:
                    h_de = h_de / s
                    val_de = compute_autoconv_max(h_de)
                    if val_de < best_val:
                        best_val = val_de
                        best_h = h_de.copy()
        except Exception:
            pass
    
    # Stage 2: Local refinement with multiple restarts
    if time_left() > 30:
        # Create starting points for local optimization
        start_points = []
        
        # Add the DE result if available
        try:
            if 'result' in dir() and hasattr(result, 'x') and result.fun < best_val + 0.01:
                start_points.append(result.x.copy())
        except Exception:
            pass
        
        # Add diverse points
        start_points.extend(generate_diverse_points(4))
        
        for sp in start_points:
            if time_left() < 15:
                break
            
            try:
                res = optimize.minimize(
                    objective_v2,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.fun < best_val:
                    h_res = params_to_h_v2(res.x)
                    s = np.sum(h_res)
                    if s > 1e-10:
                        h_res = h_res / s
                        val_res = compute_autoconv_max(h_res)
                        if val_res < best_val:
                            best_val = val_res
                            best_h = h_res.copy()
            except Exception:
                pass
    
    # Stage 3: Direct grid optimization using coordinate descent with adaptive step sizes
    if time_left() > 25:
        current_h = best_h.copy()
        current_val = best_val
        
        n_iter = int(min(2000, time_left() * 30))
        step_sizes = np.ones(N) * 0.01
        
        for k in range(n_iter):
            if time_left() < 5:
                break
            
            # Select a random subset of indices to update
            n_update = random.randint(5, 20)
            indices = random.sample(range(N), n_update)
            
            improved = False
            for idx in indices:
                # Try several step sizes
                for step_mult in [0.1, 0.5, 1.0, 2.0, 5.0]:
                    delta = step_sizes[idx] * step_mult
                    if delta < 1e-10:
                        continue
                    
                    # Try positive and negative steps
                    for sign in [1, -1]:
                        new_h = current_h.copy()
                        new_h[idx] = max(0, current_h[idx] + sign * delta)
                        
                        s = np.sum(new_h)
                        if s < 1e-10:
                            continue
                        new_h = new_h / s
                        
                        new_val = compute_autoconv_max(new_h)
                        
                        if new_val < current_val - 1e-10:
                            current_h = new_h
                            current_val = new_val
                            step_sizes[idx] *= 1.5  # Increase step size if improved
                            improved = True
                            break
                    if improved:
                        break
                else:
                    # No improvement, decrease step size
                    step_sizes[idx] *= 0.8
            
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
    
    # Stage 4: Final smoothing pass to reduce spurious oscillations
    if time_left() > 10:
        # Apply a light Gaussian smoothing and re-normalize
        from scipy.ndimage import gaussian_filter1d
        h_smooth = gaussian_filter1d(best_h, sigma=2)
        h_smooth = np.maximum(h_smooth, 0)
        s = np.sum(h_smooth)
        if s > 1e-10:
            h_smooth = h_smooth / s
            val_smooth = compute_autoconv_max(h_smooth)
            if val_smooth < best_val:
                best_val = val_smooth
                best_h = h_smooth.copy()
    
    return list(best_h)
