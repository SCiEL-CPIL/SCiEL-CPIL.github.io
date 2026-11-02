import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def get_max_overlap(h):
        """
        Calculates the maximum cyclic overlap of h with 1-h.
        Overlap k = sum h_i (1 - h_{i+k}) = N/2 - autocorr[k]
        Max Overlap = N/2 - min(autocorr[1:])
        """
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(np.abs(H)**2).real
        min_autocorr = np.min(autocorr[1:])
        return N / 2.0 - min_autocorr

    def spectral_init(n_restarts=5):
        """
        Generate initial solutions using spectral shaping.
        We want to maximize min autocorrelation of zero-mean component.
        """
        candidates = []
        for _ in range(n_restarts):
            # Random phase spectrum
            freqs = np.fft.fftfreq(N)
            # Zero out DC component
            mask = np.abs(freqs) > 1e-10
            phases = 2 * np.pi * np.random.rand(N)
            # Random amplitude spectrum with some structure
            amps = np.random.rand(N) ** 0.5
            amps[~mask] = 0
            
            # Create spectrum
            spectrum = amps * np.exp(1j * phases)
            spectrum[0] = 0  # Zero mean
            
            # Inverse FFT to get zero-mean component
            u = np.fft.ifft(spectrum).real
            u = u - np.mean(u)  # Ensure zero mean
            
            # Scale to fit in [0,1] range
            u_min, u_max = u.min(), u.max()
            if u_max - u_min > 1e-10:
                u_scaled = (u - u_min) / (u_max - u_min)
            else:
                u_scaled = np.full(N, 0.5)
            
            # Add small random perturbation
            u_scaled += 0.05 * np.random.randn(N)
            u_scaled = np.clip(u_scaled, 0, 1)
            
            # Fix sum to N/2
            diff = N/2.0 - np.sum(u_scaled)
            u_scaled[N-1] += diff
            u_scaled = np.clip(u_scaled, 0, 1)
            
            if 0 <= u_scaled[N-1] <= 1:
                candidates.append(u_scaled)
        
        return candidates

    def simulated_annealing(h_init, max_iter=5000):
        """
        Simulated annealing to minimize max overlap.
        """
        h = h_init.copy()
        current_score = get_max_overlap(h)
        best_h = h.copy()
        best_score = current_score
        
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling = 0.9995
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Generate perturbation
            # Mix of single-point and multi-point perturbations
            if np.random.random() < 0.7:
                # Single point perturbation
                idx = np.random.randint(N)
                delta = np.random.uniform(-0.1, 0.1)
                h_new = h.copy()
                h_new[idx] += delta
            else:
                # Multi-point perturbation
                n_pert = np.random.randint(2, 6)
                idxs = np.random.choice(N, n_pert, replace=False)
                deltas = np.random.uniform(-0.05, 0.05, n_pert)
                h_new = h.copy()
                h_new[idxs] += deltas
            
            # Project to [0,1] and fix sum
            h_new = np.clip(h_new, 0, 1)
            diff = N/2.0 - np.sum(h_new)
            # Distribute correction
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            
            # If sum is still off, adjust the element closest to 0.5
            diff = N/2.0 - np.sum(h_new)
            if abs(diff) > 1e-10:
                closest_to_half = np.argmin(np.abs(h_new - 0.5))
                h_new[closest_to_half] += diff
                h_new = np.clip(h_new, 0, 1)
            
            if np.any(h_new < -1e-9) or np.any(h_new > 1 + 1e-9):
                continue
            
            new_score = get_max_overlap(h_new)
            
            # Accept with probability
            if new_score < current_score:
                h = h_new
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new.copy()
            else:
                delta_score = new_score - current_score
                accept_prob = np.exp(-delta_score / T)
                if np.random.random() < accept_prob:
                    h = h_new
                    current_score = new_score
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_h, best_score

    best_h = None
    best_score = np.inf

    # Phase 1: Spectral initialization + simulated annealing
    init_candidates = spectral_init(n_restarts=8)
    
    for h_init in init_candidates:
        if time.time() > deadline - 30:
            break
        
        h_sa, score_sa = simulated_annealing(h_init, max_iter=3000)
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()

    # Phase 2: Also try binary initialization
    for r in range(5):
        if time.time() > deadline - 30:
            break
        
        h_bin = np.zeros(N)
        ones_indices = np.random.choice(N, N//2, replace=False)
        h_bin[ones_indices] = 1.0
        
        h_sa, score_sa = simulated_annealing(h_bin, max_iter=3000)
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()

    # Phase 3: Local polish with scipy
    if best_h is not None:
        h0 = best_h.copy()
        
        def objective(x):
            h = np.empty(N)
            h[:N-1] = x
            h[N-1] = N/2.0 - np.sum(x)
            
            if np.any(h < -1e-6) or np.any(h > 1 + 1e-6):
                penalty = np.sum(np.maximum(0, -h)**2 + np.maximum(0, h-1)**2)
                return 1000.0 + penalty
            
            return get_max_overlap(h)
            
        bounds = [(0, 1)] * (N-1)
        
        try:
            res = minimize(objective, h0[:N-1], method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7})
            
            if res.fun < best_score:
                h_candidate = np.empty(N)
                h_candidate[:N-1] = res.x
                h_candidate[N-1] = N/2.0 - np.sum(res.x)
                
                h_candidate = np.clip(h_candidate, 0, 1)
                diff = N/2.0 - np.sum(h_candidate)
                h_candidate[N-1] += diff
                
                if 0 <= h_candidate[N-1] <= 1:
                    final_score = get_max_overlap(h_candidate)
                    if final_score < best_score:
                        best_score = final_score
                        best_h = h_candidate
        except Exception:
            pass

    # Fallback
    if best_h is None:
        best_h = np.full(N, 0.5)

    # Ensure exact sum
    diff = N/2.0 - np.sum(best_h)
    best_h[N-1] += diff
    
    if best_h[N-1] < -1e-9 or best_h[N-1] > 1 + 1e-9:
        best_h = np.full(N, 0.5)

    return best_h
