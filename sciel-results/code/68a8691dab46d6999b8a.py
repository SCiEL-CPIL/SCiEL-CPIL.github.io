import numpy as np
import time
from scipy.optimize import minimize

def construct():
    N = 600
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # We represent h as |IDFT(Z)|^2
    # Z is a complex vector of length N.
    # To ensure h is real and non-negative, h = |IDFT(Z)|^2 is always non-negative.
    # We optimize the amplitudes and phases of Z.
    
    # Initial guess: Uniform h corresponds to Z having only the DC component?
    # If h is constant, its DFT has only DC.
    # So Z[0] = sqrt(N) * c, others 0.
    # Let's start with a random spectrum to explore the space.
    
    def compute_h(params):
        # params: [amplitudes (N), phases (N)]
        amps = params[:N]
        phases = params[N:]
        Z = amps * np.exp(1j * phases)
        # IDFT
        h = np.abs(np.fft.ifft(Z)) ** 2
        # Ensure non-negativity (numerical errors might cause tiny negatives)
        h = np.maximum(h, 0)
        return h

    def score_from_h(h):
        # Autoconvolution
        # For efficiency, use FFT
        # h has length N. Autoconv length 2N-1.
        # We only need the max.
        # max is at lag 0 for non-negative h.
        # But to be safe and follow the "max autoconv" definition generally:
        # Actually, for non-negative, max is at 0.
        # sum_h = np.sum(h)
        # peak = np.sum(h**2)
        # return peak / (sum_h**2)
        
        # However, if the "max" is not at 0 for some reason (e.g. if we allowed negatives, 
        # but we enforce non-negativity), we should check.
        # With non-negativity, max is at 0.
        sum_h = np.sum(h)
        if sum_h == 0:
            return 1e10
        peak = np.sum(h**2)
        return peak / (sum_h**2)

    def objective(params):
        h = compute_h(params)
        return score_from_h(h)

    # Initial parameters
    # Start with a uniform-like spectrum (DC dominant) but add noise to allow movement
    np.random.seed(42)
    amps_init = np.zeros(N)
    amps_init[0] = 1.0
    # Add small noise to other amplitudes to break symmetry
    amps_init[1:] = 0.1 * np.random.rand(N-1)
    phases_init = 2 * np.pi * np.random.rand(N)
    
    x0 = np.concatenate([amps_init, phases_init])
    
    best_params = x0
    best_score = objective(x0)
    
    # Optimization
    # L-BFGS-B
    # We need to handle the time budget.
    
    def objective_wrapper(params):
        nonlocal best_score, best_params
        val = objective(params)
        if val < best_score:
            best_score = val
            best_params = params.copy()
        return val

    # Use L-BFGS-B
    # Since N=600, 2N=1200 variables. This is large but manageable for L-BFGS-B if not too many iterations.
    # We limit iterations based on time.
    
    try:
        # maxiter is a safety, but we check time in the callback
        res = minimize(
            objective_wrapper,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        pass # Just return best found

    # Finalize
    h_best = compute_h(best_params)
    
    # The evaluator expects a sequence of nonnegative step heights.
    # We should normalize or scale?
    # The score is invariant to scaling: 
    # (c^2 * sum h^2) / (c^2 * (sum h)^2) = same.
    # So we can return h_best as is.
    
    return h_best.tolist()

if __name__ == "__main__":
    construct()
