import numpy as np
import time
from scipy import optimize
from scipy.signal import dpss

N = 600
DELTA = 0.5 / N  # interval width

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Precompute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    def compute_score(h):
        """
        Compute max(autoconv) / ||h||_2^2.
        h: array of shape (N,)
        """
        h = np.asarray(h, dtype=np.float64)
        if h.size != N:
            h = np.resize(h, N)
        h = np.maximum(h, 0.0)
        
        l2_sq = np.sum(h**2) * DELTA
        if l2_sq < 1e-15:
            return float('inf')
            
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.rfft(h_padded)
        autoconv_circ = np.fft.irfft(H * H, n_fft)
        
        # Linear autoconvolution is in the first 2N-1 entries
        autoconv_vals = autoconv_circ[:2*N-1] * DELTA
        max_autoconv = np.max(autoconv_vals)
        
        return max_autoconv / l2_sq

    def smooth_positive(x, eps=1e-4):
        """Smooth approximation of max(x, 0)."""
        return 0.5 * (x + np.sqrt(x**2 + eps**2))

    best_score = float('inf')
    best_h = None

    # Try different DPSS configurations
    # K: number of modes, W: time-bandwidth product
    configs = [
        (3, 2.0),
        (5, 2.0),
        (5, 4.0),
        (7, 3.0),
        (10, 5.0),
        (10, 8.0),
        (15, 10.0),
    ]

    for K, W in configs:
        if time.time() > deadline - 20:
            break

        try:
            # Generate DPSS basis
            # dpss returns the first K modes
            basis = dpss(N, W, K)
            # Ensure basis is float64
            basis = basis.astype(np.float64)
            
            def objective(params):
                # params are the coefficients c_k
                # h_raw = sum c_k * basis_k
                h_raw = basis @ params
                # Apply smooth positive projection
                h = smooth_positive(h_raw, eps=1e-4)
                return compute_score(h)

            def params_to_h(params):
                h_raw = basis @ params
                return smooth_positive(h_raw, eps=1e-4)

            # Initial guess: dominant first mode
            # The first DPSS mode is the most concentrated (Gaussian-like)
            x0 = np.zeros(K)
            x0[0] = 1.0
            
            # Bounds: allow negative coefficients, but the projection handles non-negativity
            # We can bound them to avoid exploding values
            bounds = [(-10.0, 10.0)] * K

            # Run L-BFGS-B
            res = optimize.minimize(
                objective, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_h = params_to_h(res.x).copy()
                
        except Exception as e:
            pass

    # If DPSS didn't find anything good, try a simple Gaussian fallback
    if best_h is None or best_score > 1.5:
        centers = np.arange(N) - N // 2
        for sigma in [5, 10, 20, 50]:
            h = np.exp(-centers**2 / (2 * sigma**2))
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if best_h is None:
            best_h = np.ones(N)
            best_score = compute_score(best_h)

    # Optional: Local refinement with coordinate descent on the best DPSS result
    # This helps if the smooth approximation left some "flat" zeros that could be adjusted
    if best_h is not None and time.time() < deadline - 10:
        h_ref = best_h.copy()
        current_score = best_score
        step_size = 0.1
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(0, N, 2): # Step by 2 for speed
                if time.time() > deadline - 5:
                    break
                # Try increasing
                h_trial = h_ref.copy()
                h_trial[i] += step_size
                trial_score = compute_score(h_trial)
                if trial_score < current_score:
                    h_ref = h_trial
                    current_score = trial_score
                    improved = True
                    continue
                # Try decreasing
                h_trial = h_ref.copy()
                h_trial[i] = max(0, h_trial[i] - step_size)
                trial_score = compute_score(h_trial)
                if trial_score < current_score:
                    h_ref = h_trial
                    current_score = trial_score
                    improved = True
            if not improved:
                step_size *= 0.9
                if step_size < 1e-4:
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_ref

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
