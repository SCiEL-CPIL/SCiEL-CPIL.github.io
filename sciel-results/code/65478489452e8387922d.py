"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Generate diverse initial points (random, structured, bump profiles).
    2. Use smooth surrogate (log-sum-exp) + L-BFGS-B gradient optimization.
    3. Final exact evaluation and polish.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def smooth_objective(x):
        """Smooth surrogate using log-sum-exp of conv entries."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        # Use log-sum-exp to approximate max smoothly
        max_c = np.max(conv)
        # Temperature-based smooth max
        beta = 50.0
        # Only consider positive entries for the max (since we want max, not min)
        shifted = conv - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val
    
    def smooth_grad(x):
        """Gradient of the smooth surrogate (numerical)."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        beta = 50.0
        shifted = conv - max_c
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        # d(log_sum_exp)/d(conv_i) = weights[i]
        # d(conv_i)/d(h_j) = h[i-j] if 0<=i-j<N else 0 (for full conv)
        # This is complex; use numerical gradient instead
        eps = 1e-7
        grad = np.zeros_like(h)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            val_plus = smooth_objective(h_plus)
            h_minus = h.copy()
            h_minus[i] -= eps
            val_minus = smooth_objective(h_minus)
            grad[i] = (val_plus - val_minus) / (2 * eps)
        return grad
    
    rng = np.random.RandomState(42)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(20):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(20):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))
    
    # Parameterized bump profiles (shown to work well)
    # h[i] = exp(-((i - center)^2) / (2 * width^2)) with sign
    for center in [N//4, N//2, 3*N//4]:
        for width in [20, 50, 100, 150, 200]:
            i = np.arange(N)
            h_bump = np.exp(-((i - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump
            h_double = np.exp(-((i - center)**2) / (2 * width**2)) - np.exp(-((i - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(5):
            freqs = rng.randint(1, 30, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try to optimize
        try:
            result = minimize(
                smooth_objective,
                init_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-4}
            )
            
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass
        
        # Also try exact evaluation of the initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()
    
    # --- Local polish using coordinate descent on the best solution ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.05
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in [lr, -lr, lr*0.5, -lr*0.5]:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.7
    
    return best_h
# EVOLVE-BLOCK-END
