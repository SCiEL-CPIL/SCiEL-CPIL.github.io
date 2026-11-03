import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # We'll use a Fourier-based parameterization.
    # h_i = 1/2 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N + phi_k)
    # This ensures sum h_i = N/2 automatically.
    # We need h_i in [0,1].
    
    # Objective: minimize max_s O(s) where O(s) = sum_i h_i*(1-h_{(i+s)%N})
    
    def compute_overlap(h):
        # h is array of length N
        # O(s) for s in 1..N-1 (and maybe s=0)
        # Use FFT for efficiency
        # O(s) = sum_i h_i * (1 - h_{i+s})
        # = sum_i h_i - sum_i h_i h_{i+s}
        # = N/2 - R(s) where R(s) = sum_i h_i h_{i+s} (cyclic cross-corr)
        
        # R(s) = circular correlation of h with h
        # Use FFT: R = ifft(fft(h) * conj(fft(h)))  -- this gives R(s) = sum_i h_{i+s} * conj(h_i)
        # For real h, R(s) = sum_i h_{i+s} h_i
        H = np.fft.fft(h)
        R = np.fft.ifft(H * np.conj(H)).real
        # R[s] = sum_i h[(i+s)%N] * h[i]
        # O(s) = N/2 - R[s]
        O = N/2.0 - R
        return O
    
    def score_from_h(h):
        O = compute_overlap(h)
        # The "maximum overlap" - do we include s=0?
        # At s=0: O(0) = sum h_i(1-h_i) = N/2 - sum h_i^2
        # For h_i in [0,1], O(0) <= N/4 = 25 (max at h_i=0.5)
        # For binary, O(0) = 0
        # The task says "shifted 1-h", so probably s != 0
        # Let's consider s in 1..N-1
        max_overlap = np.max(O[1:])
        return max_overlap
    
    # Parameterize in Fourier domain
    # h_i = 1/2 + sum_{k=1}^{K} amp_k * cos(2*pi*k*i/N + phase_k)
    # Use K modes
    
    K = 10  # number of Fourier modes to use
    
    def fourier_to_h(amps, phases):
        """amps: shape (K,), phases: shape (K,)"""
        h = np.full(N, 0.5)
        for k in range(K):
            h += amps[k] * np.cos(2*np.pi*k*np.arange(N)/N + phases[k])
        return h
    
    def project_to_feasible(h):
        """Project h to [0,1] with sum = N/2"""
        # Simple: clip to [0,1], then adjust to fix sum
        h = np.clip(h, 0, 1)
        # Adjust sum
        diff = N/2.0 - np.sum(h)
        if abs(diff) > 1e-10:
            # Add/subtract uniformly where possible
            if diff > 0:
                # Need to increase sum; add to elements < 1
                mask = h < 1.0 - 1e-10
                if np.sum(mask) > 0:
                    add_per = diff / np.sum(mask)
                    h[mask] = np.clip(h[mask] + add_per, 0, 1)
            else:
                mask = h > 1e-10
                if np.sum(mask) > 0:
                    sub_per = (-diff) / np.sum(mask)
                    h[mask] = np.clip(h[mask] - sub_per, 0, 1)
            # Second pass for any residual
            diff2 = N/2.0 - np.sum(h)
            if abs(diff2) > 1e-10:
                if diff2 > 0:
                    mask = h < 1.0 - 1e-10
                    if np.sum(mask) > 0:
                        add_per = diff2 / np.sum(mask)
                        h[mask] = np.clip(h[mask] + add_per, 0, 1)
                else:
                    mask = h > 1e-10
                    if np.sum(mask) > 0:
                        sub_per = (-diff2) / np.sum(mask)
                        h[mask] = np.clip(h[mask] - sub_per, 0, 1)
        return h
    
    def objective(params):
        """params: [amps(0..K-1), phases(0..K-1)]"""
        amps = params[:K]
        phases = params[K:]
        h = fourier_to_h(amps, phases)
        h = project_to_feasible(h)
        return score_from_h(h)
    
    # Use scipy optimize
    from scipy.optimize import minimize
    
    best_h = np.full(N, 0.5)
    best_score = score_from_h(best_h)
    
    # Multi-restart with different initial conditions
    rng = np.random.default_rng(42)
    
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initial amplitudes and phases
        amps = rng.uniform(0, 0.3, size=K)
        phases = rng.uniform(0, 2*np.pi, size=K)
        params0 = np.concatenate([amps, phases])
        
        # Try L-BFGS-B
        try:
            res = minimize(objective, params0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or res.fun < best_score:
                h_new = fourier_to_h(res.x[:K], res.x[K:])
                h_new = project_to_feasible(h_new)
                s_new = score_from_h(h_new)
                if s_new < best_score:
                    best_score = s_new
                    best_h = h_new.copy()
        except Exception:
            pass
        
        # Also try with larger amplitudes for some restarts
        if r % 5 == 0:
            amps = rng.uniform(0, 0.5, size=K)
            phases = rng.uniform(0, 2*np.pi, size=K)
            params0 = np.concatenate([amps, phases])
            try:
                res = minimize(objective, params0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-8})
                if res.success or res.fun < best_score:
                    h_new = fourier_to_h(res.x[:K], res.x[K:])
                    h_new = project_to_feasible(h_new)
                    s_new = score_from_h(h_new)
                    if s_new < best_score:
                        best_score = s_new
                        best_h = h_new.copy()
            except Exception:
                pass
    
    # Final refinement: try to improve by local search on the best solution
    if time.time() < deadline - 5:
        # Try a few more random perturbations of the best
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            h_pert = best_h + rng.normal(0, 0.05, size=N)
            h_pert = project_to_feasible(h_pert)
            s_pert = score_from_h(h_pert)
            if s_pert < best_score:
                best_score = s_pert
                best_h = h_pert.copy()
    
    return best_h
