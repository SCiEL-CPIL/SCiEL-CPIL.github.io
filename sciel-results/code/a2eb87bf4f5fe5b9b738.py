import numpy as np
import time
from scipy.optimize import differential_evolution
import sys

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

# Precompute grid
x_edges = np.linspace(-0.25, 0.25, N + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
dx = x_edges[1] - x_edges[0]

# Precompute FFT size for fast convolution
FFT_SIZE = 2048

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
        
    # Use FFT for convolution
    f_padded = np.zeros(FFT_SIZE)
    f_padded[:N] = f
    F = fft(f_padded)
    conv_padded = ifft(F * F)
    conv = np.real(conv_padded[:2*N-1])
    
    max_conv = np.max(conv)
    
    return max_conv / (s * s)

def geometric_to_step(params, num_pulses):
    """
    Convert geometric parameters to a step function on the fixed grid.
    params: array of shape (3 * num_pulses,)
             [center_1, width_1, height_1, center_2, width_2, height_2, ...]
    """
    f = np.zeros(N)
    
    for i in range(num_pulses):
        c = params[3*i]
        w = params[3*i+1]
        h = params[3*i+2]
        
        # Clip width to be non-negative and within bounds
        w = max(0.0, w)
        h = max(0.0, h)
        
        if w < 1e-6 or h < 1e-6:
            continue
            
        # Calculate interval indices covered by the pulse
        # Pulse is in [c - w/2, c + w/2]
        left = c - w / 2.0
        right = c + w / 2.0
        
        # Map to grid indices
        # x_edges[0] = -0.25, x_edges[N] = 0.25
        # Index j covers [x_edges[j], x_edges[j+1]]
        
        # Find start index: first j such that x_edges[j+1] > left
        # Find end index: last j such that x_edges[j] < right
        
        # Using searchsorted for efficiency
        # We want indices j where the interval [x_edges[j], x_edges[j+1]] overlaps [left, right]
        
        # Start: x_edges[j+1] > left  => j+1 > pos_left => j >= pos_left
        # pos_left is index in x_edges such that x_edges[pos_left] <= left < x_edges[pos_left+1]
        idx_left = np.searchsorted(x_edges, left, side='right') - 1
        start_idx = max(0, idx_left)
        
        # End: x_edges[j] < right => j < pos_right
        # pos_right is index in x_edges such that x_edges[pos_right] <= right < x_edges[pos_right+1]
        idx_right = np.searchsorted(x_edges, right, side='left') - 1
        end_idx = min(N - 1, idx_right)
        
        if start_idx <= end_idx:
            f[start_idx:end_idx+1] += h
            
    return f

def objective_geometric(params, num_pulses):
    """
    Objective function for differential evolution.
    """
    f = geometric_to_step(params, num_pulses)
    if np.sum(f) < 1e-12:
        return 1e10
    score = compute_score(f)
    return score

def construct():
    start_time = time.time()
    
    # Define search space
    # We will try a few different numbers of pulses.
    # More pulses = more flexible shape, but harder to optimize.
    # Start with 3 pulses (e.g., one wide central, two narrower side?) 
    # Actually, the optimal shape for minimizing peak autoconvolution with fixed L1 norm 
    # is often related to spreading out. 
    # Let's try num_pulses = 1, 2, 3, 4.
    
    # Bounds:
    # center: [-0.25, 0.25]
    # width: [0.0, 0.5] (max width of domain)
    # height: [0.0, 10.0] (scale invariant, but need upper bound for DE)
    
    best_score = np.inf
    best_f = None
    
    # Initial guess: uniform function
    f_uniform = np.ones(N)
    best_score = compute_score(f_uniform)
    best_f = f_uniform.copy()
    
    pulse_counts = [1, 2, 3, 4, 5]
    
    for num_pulses in pulse_counts:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        bounds = [(-0.25, 0.25), (0.0, 0.5), (0.0, 10.0)] * num_pulses
        
        # Initial population strategy:
        # We can provide a seed or let DE handle it.
        # To improve speed, we can limit maxiter and popsize.
        
        def wrapper(params):
            return objective_geometric(params, num_pulses)
            
        try:
            res = differential_evolution(
                wrapper,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False,  # Disable polish to save time, we'll do local refinement later if needed
                updating='immediate',
                workers=1
            )
            
            if res.success or res.fun < best_score:
                f_opt = geometric_to_step(res.x, num_pulses)
                s = np.sum(f_opt)
                if s > 1e-12:
                    f_opt = f_opt / s # Normalize, though score is invariant, good for consistency
                    score_opt = compute_score(f_opt)
                    if score_opt < best_score:
                        best_score = score_opt
                        best_f = f_opt.copy()
                        
        except Exception as e:
            pass
            
    # If we have time, do a local refinement using L-BFGS on the 600-dim vector
    # starting from the best geometric solution.
    # This combines the global search of DE with the local precision of L-BFGS.
    
    if time.time() - start_time < TIME_LIMIT - 10 and best_f is not None:
        f_local = best_f.copy()
        
        # Simple local refinement:
        # We can use the smooth max gradient from the original code logic, 
        # but simplified since we have a good starting point.
        
        # Let's implement a quick L-BFGS-B step.
        # We need the gradient function.
        
        def smooth_max_and_grad(f, beta=100.0):
            f = np.maximum(f, 0.0)
            s = np.sum(f)
            if s < 1e-15:
                return np.inf, np.zeros(N)
                
            f_padded = np.zeros(FFT_SIZE)
            f_padded[:N] = f
            F = fft(f_padded)
            conv_padded = ifft(F * F)
            conv = np.real(conv_padded[:2*N-1])
            
            beta_conv = beta * conv
            max_bc = np.max(beta_conv)
            if max_bc > 700:
                idx = np.argmax(beta_conv)
                grad_conv = np.zeros(len(conv))
                grad_conv[idx] = 1.0
                return max_c, grad_conv
                
            exp_bc = np.exp(beta_conv - max_bc)
            sum_exp = np.sum(exp_bc)
            M = (max_bc + np.log(sum_exp)) / beta
            grad_conv = exp_bc / sum_exp
            
            # Gradient calculation via FFT correlation
            FFT_SIZE_BIG = 4096
            G_big = np.zeros(FFT_SIZE_BIG)
            G_big[:len(grad_conv)] = grad_conv
            f_big = np.zeros(FFT_SIZE_BIG)
            f_big[:N] = f
            
            G_rev = grad_conv[::-1]
            G_rev_big = np.zeros(FFT_SIZE_BIG)
            G_rev_big[:len(G_rev)] = G_rev
            
            corr = ifft(fft(G_rev_big) * fft(f_big)).real
            J = np.arange(N)
            grad_f = 2.0 * corr[2 * N - 2 - J]
            
            return M, grad_f

        def obj_wrapper(f):
            val, grad = smooth_max_and_grad(f, beta=50.0)
            s = np.sum(f)
            penalty = 10.0 * (s - 1.0)**2
            grad_penalty = 20.0 * (s - 1.0) * np.ones(N)
            return val + penalty, grad + grad_penalty

        from scipy.optimize import minimize
        try:
            res_local = minimize(
                obj_wrapper, 
                f_local, 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            f_opt_local = res_local.x
            f_opt_local = np.maximum(f_opt_local, 0)
            s = np.sum(f_opt_local)
            if s > 1e-6:
                f_opt_local /= s
                score_local = compute_score(f_opt_local)
                if score_local < best_score:
                    best_score = score_local
                    best_f = f_opt_local.copy()
        except Exception:
            pass

    # Final check
    if best_f is None:
        best_f = np.ones(N)
        
    best_f = np.maximum(best_f, 0.0)
    s_final = np.sum(best_f)
    if s_final > 0:
        best_f /= s_final

    return [float(x) for x in best_f]

# Import FFT functions
from scipy.fft import fft, ifft

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
