import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    """
    Minimizes the maximum overlap of h with shifted 1-h.
    h is a sequence of step heights in [0, 1] with sum N/2.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def compute_overlaps(h):
        """Compute C(k) = sum_i h_i * (1 - h_{i+k}) for all shifts k."""
        C = np.zeros(N)
        # Vectorized computation using FFT for speed if N is large, but N=100 is small.
        # Direct loop is fine and clear.
        for k in range(N):
            h_shifted = np.roll(h, -k)
            C[k] = np.sum(h * (1.0 - h_shifted))
        return C

    def objective(params, beta=50.0, lambda_sum=1000.0):
        """
        Objective function: Smooth max of overlaps + penalty for sum constraint.
        params are the step heights h.
        """
        h = params
        C = compute_overlaps(h)
        
        # LogSumExp for smooth maximum
        # Shift C by its max for numerical stability
        max_C = np.max(C)
        exp_C = np.exp(beta * (C - max_C))
        lse = max_C + np.log(np.sum(exp_C)) / beta
        
        # Penalty for sum constraint
        sum_penalty = lambda_sum * (np.sum(h) - N / 2.0) ** 2
        
        return lse + sum_penalty

    def evaluate_h(h):
        """Return the true max overlap for a given h."""
        C = compute_overlaps(h)
        return np.max(C)

    best_h = np.full(N, 0.5)
    best_score = evaluate_h(best_h)
    
    rng = np.random.default_rng(42)
    
    # Initializations to try
    initializations = []
    
    # 1. Uniform (already set as best_h)
    
    # 2. Sinusoidal patterns
    for freq in [1, 2, 3, 5]:
        for phase in [0, np.pi/2, np.pi]:
            x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
            h_init = 0.5 + 0.4 * np.sin(x + phase)
            h_init = np.clip(h_init, 0.0, 1.0)
            # Normalize to sum N/2
            s = np.sum(h_init)
            if s > 1e-10:
                h_init *= (N / 2.0) / s
            h_init = np.clip(h_init, 0.0, 1.0)
            initializations.append(h_init)
            
    # 3. Binary-like patterns (random permutations of 50 ones and 50 zeros, slightly perturbed)
    for _ in range(10):
        h_bin = np.zeros(N)
        h_bin[:N//2] = 1.0
        h_init = h_bin[rng.permutation(N)]
        # Add small noise to break symmetry
        h_init += rng.uniform(-0.05, 0.05, N)
        h_init = np.clip(h_init, 0.0, 1.0)
        s = np.sum(h_init)
        if s > 1e-10:
            h_init *= (N / 2.0) / s
        h_init = np.clip(h_init, 0.0, 1.0)
        initializations.append(h_init)
        
    # 4. Random smooth patterns
    for _ in range(10):
        # Low frequency noise
        h_noise = rng.uniform(0, 1, N)
        # Smooth it
        kernel = np.ones(5) / 5.0
        h_smooth = np.convolve(h_noise, kernel, mode='same')
        h_init = 0.5 + 0.4 * (h_smooth - np.mean(h_smooth)) / (np.std(h_smooth) + 1e-10)
        h_init = np.clip(h_init, 0.0, 1.0)
        s = np.sum(h_init)
        if s > 1e-10:
            h_init *= (N / 2.0) / s
        h_init = np.clip(h_init, 0.0, 1.0)
        initializations.append(h_init)

    # Optimization settings
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    n_restarts = 20
    for i in range(n_restarts):
        if elapsed() > time_limit - 10.0:
            break
            
        # Pick an initialization
        if i < len(initializations):
            x0 = initializations[i].copy()
        else:
            x0 = rng.uniform(0.1, 0.9, N)
            s = np.sum(x0)
            if s > 1e-10:
                x0 *= (N / 2.0) / s
            x0 = np.clip(x0, 0.0, 1.0)
            
        # Run L-BFGS-B
        try:
            # Increase beta for sharper max approximation in later iterations?
            # Start with moderate beta
            beta = 30.0 + 10.0 * i
            
            result = minimize(
                objective,
                x0,
                args=(beta, 1000.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = result.x
            # Project to ensure sum is exactly N/2 and bounds are respected
            h_opt = np.clip(h_opt, 0.0, 1.0)
            s = np.sum(h_opt)
            if s > 1e-10:
                h_opt *= (N / 2.0) / s
            h_opt = np.clip(h_opt, 0.0, 1.0)
            
            # Final adjustment for precision
            s2 = np.sum(h_opt)
            if abs(s2 - N/2.0) > 1e-12:
                h_opt += (N/2.0 - s2) / N
            h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = evaluate_h(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Final verification
    h_final = best_h.copy()
    s = np.sum(h_final)
    if abs(s - N/2.0) > 1e-10:
        h_final += (N/2.0 - s) / N
    h_final = np.clip(h_final, 0.0, 1.0)
    s2 = np.sum(h_final)
    if abs(s2 - N/2.0) > 1e-12:
        h_final += (N/2.0 - s2) / N
    h_final = np.clip(h_final, 0.0, 1.0)
    
    return h_final
