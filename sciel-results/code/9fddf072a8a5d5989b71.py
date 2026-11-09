import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100

def compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h using FFT."""
    # overlap(s) = sum_i h[i] * (1 - h[(i+s) % N])
    #            = sum_i h[i] - sum_i h[i] * h[(i+s) % N]
    #            = N/2 - autocorr(s)
    # where autocorr(s) = sum_i h[i] * h[(i+s) % N]
    # We want to minimize max_s overlap(s) = N/2 - min_s autocorr(s)
    # So we want to maximize min_s autocorr(s)
    
    H = np.fft.rfft(h)
    # autocorr via FFT: IFFT(|H|^2) gives circular autocorrelation
    acf = np.fft.irfft(np.abs(H)**2, n=N)
    # acf[s] = sum_i h[i] * h[(i-s) % N] (circular)
    # Due to symmetry, acf[s] = acf[N-s]
    min_acf = np.min(acf)
    max_overlap = N / 2.0 - min_acf
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Strategy: h[i] = 0.5 + sum_k a_k * cos(2*pi*k*i/N + phi_k)
    # This guarantees sum(h) = N*0.5 = N/2 (DC term only from baseline)
    # and 0 <= h[i] <= 1 constrains the amplitudes.
    
    # We'll parameterize with a few harmonics and optimize (a_k, phi_k)
    # to minimize max overlap.
    
    # Number of harmonics to use
    n_harmonics = 8
    
    def params_to_h(params):
        """Convert (a_1, phi_1, a_2, phi_2, ...) to h array."""
        h = np.full(N, 0.5)
        for k in range(1, n_harmonics + 1):
            a = params[2*(k-1)]
            phi = params[2*(k-1)+1]
            h += a * np.cos(2.0 * np.pi * k * np.arange(N) / N + phi)
        # Clip to [0, 1] and renormalize to keep sum = N/2
        h = np.clip(h, 0.0, 1.0)
        # Adjust to ensure sum is exactly N/2
        current_sum = np.sum(h)
        if current_sum > 0:
            # Scale toward 0.5
            h = h * (N / 2.0) / current_sum
            h = np.clip(h, 0.0, 1.0)
            # Fine-tune: add/remove small amounts to hit exact sum
            diff = N / 2.0 - np.sum(h)
            if abs(diff) > 1e-10:
                # Distribute difference uniformly
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                # Final correction on non-boundary elements
                diff2 = N / 2.0 - np.sum(h)
                if abs(diff2) > 1e-12:
                    # Add to elements that can absorb it
                    can_add = h < 1.0
                    can_sub = h > 0.0
                    if diff2 > 0:
                        idx = np.where(can_add)[0]
                        if len(idx) > 0:
                            per = diff2 / len(idx)
                            h[idx] = np.minimum(h[idx] + per, 1.0)
                    else:
                        idx = np.where(can_sub)[0]
                        if len(idx) > 0:
                            per = diff2 / len(idx)
                            h[idx] = np.maximum(h[idx] + per, 0.0)
        return h
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        h = params_to_h(params)
        if np.any(h < -1e-10) or np.any(h > 1.0 + 1e-10):
            return 1e6
        return compute_score(h)
    
    # Bounds: amplitude in [0, 0.5], phase in [0, 2*pi]
    bounds = []
    for k in range(1, n_harmonics + 1):
        bounds.append((0.0, 0.5))
        bounds.append((0.0, 2.0 * np.pi))
    
    best_score = 1e10
    best_h = None
    
    # Try several random initializations
    np.random.seed(42)
    
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        x0 = np.random.uniform(low=[b[0] for b in bounds], 
                               high=[b[1] for b in bounds])
        
        # Use BFGS or L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=bounds, 
                             options={'maxiter': 200, 'maxfun': 500})
            if result.fun < best_score:
                best_score = result.fun
                best_h = params_to_h(result.x)
        except Exception:
            pass
        
        # Also try a few gradient-free steps from this point
        if time.time() > deadline:
            break
    
    # Also try differential evolution with fewer harmonics for robustness
    if time.time() < deadline - 20:
        n_harm_de = 4
        bounds_de = []
        for k in range(1, n_harm_de + 1):
            bounds_de.append((0.0, 0.5))
            bounds_de.append((0.0, 2.0 * np.pi))
        
        def params_to_h_de(params):
            h = np.full(N, 0.5)
            for k in range(1, n_harm_de + 1):
                a = params[2*(k-1)]
                phi = params[2*(k-1)+1]
                h += a * np.cos(2.0 * np.pi * k * np.arange(N) / N + phi)
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            if current_sum > 0:
                h = h * (N / 2.0) / current_sum
                h = np.clip(h, 0.0, 1.0)
                diff = N / 2.0 - np.sum(h)
                if abs(diff) > 1e-10:
                    h += diff / N
                    h = np.clip(h, 0.0, 1.0)
                    diff2 = N / 2.0 - np.sum(h)
                    if abs(diff2) > 1e-12:
                        can_add = h < 1.0
                        can_sub = h > 0.0
                        if diff2 > 0:
                            idx = np.where(can_add)[0]
                            if len(idx) > 0:
                                per = diff2 / len(idx)
                                h[idx] = np.minimum(h[idx] + per, 1.0)
                        else:
                            idx = np.where(can_sub)[0]
                            if len(idx) > 0:
                                per = diff2 / len(idx)
                                h[idx] = np.maximum(h[idx] + per, 0.0)
            return h
        
        def objective_de(params):
            if time.time() > deadline:
                return 1e6
            h = params_to_h_de(params)
            if np.any(h < -1e-10) or np.any(h > 1.0 + 1e-10):
                return 1e6
            return compute_score(h)
        
        try:
            de_result = differential_evolution(objective_de, bounds_de, 
                                              maxiter=30, popsize=15,
                                              seed=42, tol=1e-6,
                                              polish=True)
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_h = params_to_h_de(de_result.x)
        except Exception:
            pass
    
    # Fallback: if no valid solution found, use uniform
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Final verification and correction
    best_h = np.clip(best_h, 0.0, 1.0)
    s = np.sum(best_h)
    if abs(s - N/2.0) > 1e-12:
        best_h += (N/2.0 - s) / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
