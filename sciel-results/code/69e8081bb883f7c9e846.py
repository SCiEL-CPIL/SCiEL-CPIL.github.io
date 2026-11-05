# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

def construct():
    """
    Construct step heights h[0..N-1] to minimize value = abs(2*N*max(convolve(h,h)) / sum(h)^2).
    Model: h = Re(Zadoff-Chu(mu) * exp(i*pi*alpha*t^2/N)) * Window(t; width) + DC
    """
    N = 400
    best_h = np.ones(N)
    best_val = 2.0  # Seed value

    # Precompute constants
    t = np.arange(N, dtype=np.float64)
    ones = np.ones(N)
    T_conv = np.convolve(ones, ones) # Constant part for DC^2
    # Note: U_conv and C_conv depend on x, so they are computed in the loop

    def evaluate_full(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    def get_ac_signal(mu, alpha, beta, width):
        """
        Compute the AC component (zero mean approximately) of the signal.
        mu: Zadoff-Chu parameter
        alpha: Chirp rate
        beta: Phase
        width: Window width parameter (maps to standard deviation or Hann width)
        """
        # Zadoff-Chu sequence: exp(-i * pi * mu * t * (t+1) / N)
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        
        # Chirp: exp(i * pi * alpha * t^2 / N)
        # Combining phases: -pi*mu*t*(t+1)/N + pi*alpha*t^2/N
        # = pi/N * t^2 * (alpha - mu) - pi/N * mu * t
        # We can compute directly for clarity and to allow independent optimization if needed,
        # but combining is faster. Let's stick to the product form for clarity in code structure 
        # as requested by the "design" directive, though mathematically equivalent.
        
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        phase = np.exp(1j * beta)
        
        complex_signal = zc * chirp * phase
        
        # Window: Use a Gaussian window for smoothness which helps reduce sidelobes
        # width in [0.2, 1.0] maps to sigma
        # Center the window
        center = (N - 1) / 2.0
        sigma = 0.1 + 0.4 * width  # sigma in [0.1, 0.5] relative to N/2? 
        # Let's define width as a fraction of N. 
        # If width=1.0, sigma ~ N/4 (wide). If width=0.2, sigma ~ N/20 (narrow).
        sigma = N * (0.05 + 0.25 * width)
        
        w = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        
        h_ac = np.real(complex_signal) * w
        return h_ac

    def optimize_dc(x):
        """
        Find optimal DC offset c for signal x to minimize the objective.
        h = x + c
        conv(h) = conv(x) + 2c conv(x, 1) + c^2 conv(1, 1)
        sum(h) = sum(x) + Nc
        """
        S_x = np.sum(x)
        C_x = np.convolve(x, x)
        U_x = np.convolve(x, ones)
        
        def obj(c):
            conv_c = C_x + 2 * c * U_x + c * c * T_conv
            m = np.max(conv_c)
            s = S_x + N * c
            if abs(s) < 1e-12:
                return float('inf')
            val = abs(2 * N * m / (s * s))
            return val

        # Initial guesses for c
        # 1. Make sum zero? No, sum must not be zero.
        # 2. Make max conv zero?
        # 3. Small offsets
        candidates = [0.0, -S_x / N, 0.1, -0.1, 0.5, -0.5]
        
        best_c = 0.0
        best_obj = float('inf')
        
        # Evaluate candidates
        for c in candidates:
            v = obj(c)
            if v < best_obj:
                best_obj = v
                best_c = c

        # Refine around the best candidate
        # Use a bounded search around the best candidate
        bounds = [(best_c - 1.0, best_c + 1.0)]
        # Also try a global coarse grid if the local search fails or to ensure we don't miss a far optimum
        grid_c = np.linspace(-2, 2, 21)
        for c in grid_c:
            v = obj(c)
            if v < best_obj:
                best_obj = v
                best_c = c

        # Local refinement
        try:
            res = minimize_scalar(obj, bounds=(best_c - 0.5, best_c + 0.5), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        except Exception:
            pass

        return x + best_c

    def objective(params):
        mu, alpha, beta, width = params
        try:
            x = get_ac_signal(mu, alpha, beta, width)
            h = optimize_dc(x)
            return evaluate_full(h)
        except Exception:
            return float('inf')

    # Multi-start optimization
    np.random.seed(42)
    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    best_val_found = float('inf')
    best_h_found = None
    
    n_starts = 80
    
    for i in range(n_starts):
        if time.time() - start_time > max_time:
            break
            
        # Structured + Random starts
        if i % 4 == 0:
            # Vary mu systematically
            mu = (i // 4) * 1.0 + np.random.uniform(-0.5, 0.5)
            alpha = np.random.uniform(-2.0, 2.0)
            beta = np.random.uniform(0, 2 * np.pi)
            width = np.random.uniform(0.2, 0.8)
        elif i % 4 == 1:
            # Random mu, low chirp
            mu = np.random.uniform(0, 20)
            alpha = np.random.uniform(-0.5, 0.5)
            beta = np.random.uniform(0, 2 * np.pi)
            width = np.random.uniform(0.3, 0.7)
        elif i % 4 == 2:
            # Random mu, high chirp
            mu = np.random.uniform(0, 20)
            alpha = np.random.uniform(1.0, 3.0)
            beta = np.random.uniform(0, 2 * np.pi)
            width = np.random.uniform(0.2, 0.6)
        else:
            # Fully random
            mu = np.random.uniform(0, 30)
            alpha = np.random.uniform(-3.0, 3.0)
            beta = np.random.uniform(0, 2 * np.pi)
            width = np.random.uniform(0.1, 1.0)
            
        x0 = np.array([mu, alpha, beta, width])
        
        # Use Powell for robustness with non-smooth objective (due to max and abs)
        # L-BFGS-B might struggle with the kink in max/abs
        res = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 100, 'maxfev': 1500, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        
        if res.fun < best_val_found:
            best_val_found = res.fun
            # Reconstruct best h
            mu, alpha, beta, width = res.x
            x = get_ac_signal(mu, alpha, beta, width)
            best_h_found = optimize_dc(x)
            
            # Early exit if very good
            if best_val_found < 1.05:
                # Try to refine further with more iterations
                res2 = minimize(
                    objective,
                    res.x,
                    method='Powell',
                    options={'maxiter': 300, 'maxfev': 5000, 'xatol': 1e-5, 'fatol': 1e-8}
                )
                if res2.fun < best_val_found:
                    best_val_found = res2.fun
                    mu, alpha, beta, width = res2.x
                    x = get_ac_signal(mu, alpha, beta, width)
                    best_h_found = optimize_dc(x)
                break

    if best_h_found is not None and best_val_found < best_val:
        return best_h_found
    return best_h
# EVOLVE-BLOCK-END
