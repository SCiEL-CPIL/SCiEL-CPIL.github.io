import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 145.0

    # Spectral phase optimization:
    # h_i = 1/2 + sum_{k=1}^{N/2} a_k * cos(2*pi*k*i/N + phi_k)
    # Fix magnitudes a_k, optimize phases phi_k.
    # Overlap O(s) = N/2 - R(s) where R(s) = sum_i h_i * h_{i+s}
    # R(s) depends on phases through products of cos terms.
    # Minimize max_s O(s) = maximize min_s R(s) (for s in 1..N-1)
    
    # Use log-sum-exp smoothing for differentiable max
    # R(s) = N/4 + sum_{k} a_k^2/2 * cos(2*pi*k*s/N) + cross terms
    # Actually let's just compute R via FFT of h
    
    def h_from_phases(phases, mags):
        """phases: shape (K,), mags: shape (K,)"""
        K = len(mags)
        h = np.full(N, 0.5)
        idx = np.arange(N)
        for k in range(K):
            h += mags[k] * np.cos(2*np.pi*(k+1)*idx/N + phases[k])
        return h

    def compute_R(h):
        """Circular autocorrelation R[s] = sum_i h_i * h[(i+s)%N]"""
        H = np.fft.fft(h)
        R = np.fft.ifft(H * np.conj(H)).real
        return R

    def smooth_max_overlap(h, beta=5.0):
        """Smooth approximation of max overlap using log-sum-exp"""
        R = compute_R(h)
        O = N/2.0 - R
        # Only consider shifts 1..N-1
        O_shifts = O[1:]
        # Log-sum-exp for smooth max
        m = np.max(O_shifts)
        if m < 1e-10:
            return m
        LSE = m + np.log(np.sum(np.exp(beta * (O_shifts - m)))) / beta
        return LSE

    def true_max_overlap(h):
        R = compute_R(h)
        O = N/2.0 - R
        return np.max(O[1:])

    # Choose magnitudes: use a power spectrum that gives reasonable h in [0,1]
    # h_i = 1/2 + sum a_k cos(...) so we need sum a_k <= 0.5 for h in [0,1]
    # Use geometrically decaying magnitudes
    K = 20
    mags = np.array([0.4 / (k+1) for k in range(K)])
    # Scale so max possible deviation is ~0.45
    max_dev = np.sum(mags)
    if max_dev > 0.45:
        mags = mags * 0.45 / max_dev

    rng = np.random.default_rng(42)

    best_h = np.full(N, 0.5)
    best_score = true_max_overlap(best_h)

    # Gradient descent on phases with multiple restarts
    def grad_descent_phases(phases0, lr=0.01, n_steps=500):
        """Simple gradient descent on phase angles"""
        phases = phases0.copy()
        best_local = true_max_overlap(h_from_phases(phases, mags))
        best_local_phases = phases.copy()
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            h = h_from_phases(phases, mags)
            # Clip to [0,1] for feasibility check
            h_clipped = np.clip(h, 0, 1)
            # Check if we need to rescale
            if np.any(h < -1e-6) or np.any(h > 1.0 + 1e-6):
                # Scale down magnitudes temporarily
                h = np.clip(h, 0, 1)
            
            # Compute gradient numerically (or analytically)
            # O(s) = N/2 - R(s), R(s) = sum_i h_i h_{i+s}
            # dR(s)/dphi_k = -sum_i a_k sin(2*pi*(k+1)*i/N + phi_k) * h_{i+s}
            #                - sum_i a_k sin(2*pi*(k+1)*(i+s)/N + phi_k) * h_i
            
            # Use log-sum-exp gradient for smooth max
            R = compute_R(h)
            O = N/2.0 - R
            O_shifts = O[1:]
            m = np.max(O_shifts)
            if m > 1e-10:
                weights = np.exp(5.0 * (O_shifts - m)) / np.sum(np.exp(5.0 * (O_shifts - m)))
            else:
                weights = np.ones(N-1) / (N-1)
            
            # Gradient of smooth max w.r.t. phases
            # d(smooth_max)/dphi_k = sum_s weights[s] * dO(s)/dphi_k
            # dO(s)/dphi_k = -dR(s)/dphi_k
            # R(s) = sum_i h_i * h_{i+s}
            # dh_i/dphi_k = -a_k * sin(2*pi*(k+1)*i/N + phi_k)
            # dR(s)/dphi_k = sum_i [dh_i/dphi_k * h_{i+s} + h_i * dh_{i+s}/dphi_k]
            
            grad = np.zeros(K)
            idx = np.arange(N)
            for k in range(K):
                sin_k = np.sin(2*np.pi*(k+1)*idx/N + phases[k])
                # dR(s)/dphi_k for each s
                # = -a_k * sum_i [sin_k[i]*h[(i+s)%N] + h[i]*sin_k[(i+s)%N]]
                # = -2*a_k * sum_i sin_k[i]*h[(i+s)%N]  (by symmetry of circular corr)
                # Actually R(s) = sum_i h_i h_{i+s}, so:
                # dR(s)/dphi_k = sum_i [(-a_k sin_k[i]) h_{i+s} + h_i (-a_k sin_k[(i+s)%N])]
                # By substitution j=i+s: second term = sum_j h_{j-s} (-a_k sin_k[j])
                # = -a_k * [sum_i sin_k[i] h_{i+s} + sum_j sin_k[j] h_{j-s}]
                # = -2*a_k * (sin_k * circular_corr_h_shifted)[s] ... 
                # Let's just do it directly but vectorized
                # For efficiency, use FFT
                # dR/dphi_k = -a_k * ifft(fft(sin_k) * conj(fft(h)) + conj(fft(sin_k)) * fft(h))
                # Actually simpler: 
                # R(s) = sum_i h[i] h[(i+s)%N]
                # dR(s)/dphi_k = -a_k * sum_i [sin_k[i]*h[(i+s)%N] + h[i]*sin_k[(i+s)%N]]
                # = -2*a_k * sum_i sin_k[i]*h[(i+s)%N]  (by reindexing)
                # sum_i sin_k[i]*h[(i+s)%N] is the circular cross-correlation of sin_k and h
                # = ifft(fft(sin_k) * conj(fft(h)))... let me verify
                # C[s] = sum_i sin_k[i] * h[(i+s)%N]
                # This is circular correlation. C = ifft(fft(h) * conj(fft(sin_k)))? 
                # Let's just compute it directly for K=20, N=100 (fast enough)
                corr = np.zeros(N)
                for s in range(N):
                    corr[s] = np.sum(sin_k * np.roll(h, -s))
                dR_dphi_k = -2.0 * mags[k] * corr
                dO_dphi_k = -dR_dphi_k  # O = N/2 - R
                # Weighted sum over shifts 1..N-1
                grad[k] = np.sum(weights * dO_dphi_k[1:])
            
            # Update
            phases -= lr * grad
            # Wrap phases to [0, 2*pi]
            phases = phases % (2*np.pi)
            
            # Check improvement
            h_new = h_from_phases(phases, mags)
            s_new = true_max_overlap(h_new)
            if s_new < best_local:
                best_local = s_new
                best_local_phases = phases.copy()
            
            # Adaptive learning rate
            if step % 50 == 0:
                lr = max(0.001, lr * 0.9)
        
        return best_local_phases, best_local

    # Multi-restart gradient descent
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        phases0 = rng.uniform(0, 2*np.pi, size=K)
        phases_opt, local_score = grad_descent_phases(phases0, lr=0.02, n_steps=400)
        
        h_new = h_from_phases(phases_opt, mags)
        h_new = np.clip(h_new, 0, 1)
        # Fix sum if needed
        diff = N/2.0 - np.sum(h_new)
        if abs(diff) > 1e-10:
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
        
        s_new = true_max_overlap(h_new)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
    
    # Final local refinement: try small perturbations
    if time.time() < deadline - 3:
        for _ in range(30):
            if time.time() > deadline - 1:
                break
            h_pert = best_h + rng.normal(0, 0.03, size=N)
            h_pert = np.clip(h_pert, 0, 1)
            diff = N/2.0 - np.sum(h_pert)
            h_pert += diff / N
            h_pert = np.clip(h_pert, 0, 1)
            s_pert = true_max_overlap(h_pert)
            if s_pert < best_score:
                best_score = s_pert
                best_h = h_pert.copy()

    return best_h
