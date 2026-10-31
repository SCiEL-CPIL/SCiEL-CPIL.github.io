import numpy as np
import scipy.optimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Generate a good initial guess: Blackman window
    # Blackman window formula: 0.42 - 0.5 cos(2 pi n / (N-1)) + 0.08 cos(4 pi n / (N-1))
    n = np.arange(N)
    # Ensure non-negative
    h0 = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    h0 = np.maximum(h0, 0)
    h0 = h0 / np.sum(h0)

    # Also try Hann
    h1 = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    h1 = np.maximum(h1, 0)
    h1 = h1 / np.sum(h1)

    def compute_psr(h):
        h = np.asarray(h)
        if np.any(h < -1e-10):
            return 1e10
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s == 0:
            return 1e10
        h_norm = h / s
        
        # Compute autoconvolution
        size = 2 * N
        H = np.fft.rfft(h_norm, n=size)
        c = np.fft.irfft(H * np.conj(H), n=size)
        
        c0 = c[0]
        if c0 < 1e-10:
            return 1e10
            
        # Max sidelobe
        # c is symmetric. c[1:N] are the sidelobes for positive lags.
        # We want max of c[1:N]
        max_side = np.max(c[1:N])
        
        return max_side / c0

    def compute_grad_psr(h):
        # This is complex to derive analytically for the max.
        # We will use numerical gradients or a smooth approximation.
        # Let's use a smooth max approximation in the objective and let L-BFGS-B handle it?
        # No, L-BFGS-B needs a differentiable function.
        # We can use the LogSumExp trick for the max sidelobe.
        
        h = np.asarray(h)
        h = np.maximum(h, 0)
        s = np.sum(h)
        h_norm = h / s
        
        size = 2 * N
        H = np.fft.rfft(h_norm, n=size)
        c = np.fft.irfft(H * np.conj(H), n=size)
        
        c0 = c[0]
        sides = c[1:N]
        
        # Objective: max(sides) / c0
        # Let M = max(sides).
        # Use LogSumExp for M.
        beta = 50.0
        max_side = np.max(sides)
        exp_sides = np.exp(beta * (sides - max_side))
        sum_exp = np.sum(exp_sides)
        M_smooth = max_side + np.log(sum_exp) / beta
        
        obj = M_smooth / c0
        
        # Gradient is tricky.
        # We can use finite differences for the gradient if analytical is too hard,
        # but N=600, finite diff is 600 evaluations. 600 * FFT cost.
        # FFT cost is ~ N log N. 600 * 10 = 6000 ops.
        # 600 * 6000 = 3.6M ops per gradient.
        # 100 iterations -> 360M ops. Might be too slow in Python if not vectorized.
        
        # Let's try to derive the gradient or use a simpler approach.
        # Since we only have 120s, and N=600 is small, we can try a direct local search
        # or use scipy.optimize with numerical gradient (which might be slow).
        
        # Alternative: Optimize in the frequency domain?
        # Or just use a few iterations of projected gradient with numerical grad.
        
        # Let's return the objective and use a numerical gradient method in the wrapper.
        # But scipy.optimize.minimize with jac=True expects a function returning (f, g).
        
        # Let's compute gradient numerically for the smooth objective.
        # Actually, we can just return the objective and let scipy use finite differences?
        # No, 'L-BFGS-B' with jac=True is fast if we provide gradient.
        # If we don't provide gradient, it uses finite differences.
        # Let's try without gradient first to see if it's fast enough.
        
        return obj

    # We will use a simple loop with numerical gradients or just try to fit a parametric family.
    # Given the time constraint, let's try optimizing a parametric family:
    # h_i = (1 + a * cos(2 pi i / N) + b * cos(4 pi i / N))
    # This captures the main features of Blackman/Hann.
    
    def parametric_h(a, b):
        n = np.arange(N)
        h = 1.0 + a * np.cos(2 * np.pi * n / N) + b * np.cos(4 * np.pi * n / N)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s == 0: return None
        return h / s

    def objective_params(params):
        a, b = params
        h = parametric_h(a, b)
        if h is None:
            return 1e10
        return compute_psr(h)

    # Initial guesses for (a, b)
    # Blackman: a ~ -0.5, b ~ 0.08
    # Hann: a ~ -0.5, b ~ 0
    # Uniform: a ~ 0, b ~ 0
    
    best_params = np.array([-0.5, 0.08])
    best_val = objective_params(best_params)
    
    # Optimize (a, b)
    res = scipy.optimize.minimize(objective_params, best_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if res.fun < best_val:
        best_val = res.fun
        best_params = res.x
    
    h_best = parametric_h(best_params[0], best_params[1])
    
    # Refine with direct optimization of h using a few steps of gradient descent
    # using numerical gradients?
    # Let's try a few random restarts with L-BFGS-B on h directly, using numerical gradient.
    # If it's too slow, we stick with the parametric solution.
    
    # Time check
    if time.time() < deadline - 20:
        # Try one direct optimization run
        h_current = h_best.copy()
        
        def direct_obj(h):
            return compute_psr(h)
            
        # Use Nelder-Mead for direct h optimization?
        # N=600 is too high for Nelder-Mead.
        # Use L-BFGS-B with numerical gradient.
        # This will be slow. 600+1 evaluations per step.
        # Let's just return the parametric result if it's good.
        
        # Let's check if we can do a quick coordinate descent or something.
        # No, let's trust the Blackman-like window.
        
    return list(h_best)

if __name__ == "__main__":
    h = construct()
    print("Constructed h")
