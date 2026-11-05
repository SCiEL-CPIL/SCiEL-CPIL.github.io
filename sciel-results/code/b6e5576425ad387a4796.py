import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute basis functions for the parameterization
    # We use h = c0 * ones + sum_{k in K} a_k * sin(2*pi*k*i/N)
    # Sine modes are antisymmetric around the center (for N even, sin(pi*k*i) shifts phase but keeps zero mean)
    # Actually, for N=400, sin(2*pi*k*i/N) has zero sum.
    # This ensures sum(h) = N * c0.
    # The autoconvolution of h is determined by the power spectrum.
    # By using few modes, we constrain the shape to smooth, oscillatory patterns.
    
    # Choose a set of low-frequency modes to explore
    # Low frequencies allow for large-scale structures that might reduce the max of the autoconvolution
    # relative to the square of the sum.
    modes_to_test = [
        [1],
        [2],
        [3],
        [1, 2],
        [1, 3],
        [2, 3],
        [1, 2, 3],
        [1, 4],
        [2, 5],
        [1, 2, 4],
        [1, 3, 5],
        [2, 3, 4],
        [1, 2, 3, 4],
        [1, 5],
        [2, 6],
        [1, 2, 5],
        [1, 3, 6],
    ]

    # Precompute sine vectors for all potential modes up to max mode index
    max_mode = 10
    i = np.arange(N)
    sine_bases = {}
    for k in range(1, max_mode + 1):
        sine_bases[k] = np.sin(2 * np.pi * k * i / N)

    def build_h(params, modes):
        """
        params: [c0, a1, a2, ...]
        modes: list of mode integers
        """
        c0 = params[0]
        h = np.full(N, c0)
        for idx, k in enumerate(modes):
            h += params[idx + 1] * sine_bases[k]
        return h

    def objective(params, modes):
        h = build_h(params, modes)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        c = np.convolve(h, h)
        # The value is abs(2 * N * max(c) / s^2)
        val = abs(2.0 * N * np.max(c) / (s * s))
        return val

    best_val = np.inf
    best_h = np.ones(N) # Seed

    # Initial value for seed
    seed_val = objective(np.array([1.0]), [1]) # This is just c0=1, no modes
    # Wait, seed is np.ones(400). sum=400, conv max = 400 (at lag 0).
    # val = 2*400*400 / 400^2 = 2.0.
    # My parameterization: h = c0 * ones. sum = N*c0. conv max = N*c0^2.
    # val = 2*N*(N*c0^2)/(N^2*c0^2) = 2. Correct.
    
    if seed_val < best_val:
        best_val = seed_val
        best_h = np.ones(N)

    # We will optimize for each mode combination
    # To speed up, we use L-BFGS-B or Powell. L-BFGS-B is fast but needs gradients.
    # Powell is derivative-free and robust for low dimensions.
    # Given N=400, convolve is O(N^2) ~ 160k ops. Very fast.
    # We can afford many function evaluations.
    
    for modes in modes_to_test:
        if time.time() > deadline:
            break
            
        # Initial guess: c0=1, amplitudes=0
        x0 = np.zeros(len(modes) + 1)
        x0[0] = 1.0
        
        # Try a few different initial amplitude settings to avoid local minima
        # The objective is scale invariant, so c0=1 is a good anchor.
        # We can try small positive/negative amplitudes.
        
        inits = [
            x0.copy(),
            np.array([1.0] + [0.1] * len(modes)),
            np.array([1.0] + [-0.1] * len(modes)),
            np.array([1.0] + [0.5] * len(modes)),
            np.array([1.0] + [-0.5] * len(modes)),
        ]
        
        # Add some random inits
        rng = np.random.default_rng(42 + len(modes))
        for _ in range(5):
            inits.append(np.array([1.0] + rng.uniform(-1, 1, len(modes))))

        for x_start in inits:
            if time.time() > deadline:
                break
                
            # Use Powell method for robustness without gradient calculation
            # Bounds: c0 should be non-zero. Let's say c0 in [0.1, 10].
            # Amplitudes can be anything, but too large makes sum small or zero.
            # Let's bound amplitudes to [-5, 5] to keep sum(h) reasonable.
            bounds = [(0.1, 10.0)] + [(-5.0, 5.0)] * len(modes)
            
            try:
                res = minimize(
                    lambda p: objective(p, modes),
                    x_start,
                    method='Powell',
                    bounds=bounds,
                    options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                
                if res.success or res.fun < best_val:
                    val = res.fun
                    if val < best_val:
                        best_val = val
                        best_h = build_h(res.x, modes)
            except Exception:
                pass

    # Final check: if best_h is still the seed, return seed.
    # Otherwise return the optimized h.
    # Ensure h is finite
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    s = np.sum(h)
    c = np.convolve(h, h)
    val = abs(2 * N * np.max(c) / s**2)
    print(f"Best value found: {val}")
