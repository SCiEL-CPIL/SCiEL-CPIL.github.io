"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _overlap_max(h, N):
    """Compute max over shifts of sum_i h[i] * (1 - h[(i+t) % N]) / N."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    overlap = np.real(np.fft.ifft(np.conj(H) * C))
    return np.max(overlap)


def construct():
    N = 100
    
    def project(h):
        """Project h to [0,1] with sum exactly N/2."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(100):
            diff = N / 2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        # Final fix: if still off, adjust uniformly
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        return h
    
    # Parameterize h as a sum of harmonics: h[i] = 0.5 + sum_k a_k * cos(2*pi*k*i/N + phi_k)
    # We'll use harmonics k = 1, 2, 3, 4, 5, 7, 8, 9 (avoid k=0 which is the mean)
    harmonics = [1, 2, 3, 4, 5, 7, 8, 9]
    n_harm = len(harmonics)
    
    # Simulated annealing over (amplitudes, phases)
    np.random.seed(42)
    
    # Initial: small random amplitudes and random phases
    best_params = None
    best_score = float('inf')
    
    # Multiple restarts with different initial conditions
    for restart in range(20):
        if restart == 0:
            # Start with a structured pattern
            amplitudes = np.array([0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01])
            phases = np.zeros(n_harm)
        else:
            amplitudes = np.random.uniform(0.0, 0.15, n_harm)
            phases = np.random.uniform(0, 2 * np.pi, n_harm)
        
        current_params = np.concatenate([amplitudes, phases])
        current_score = float('inf')
        
        # Build h from params
        def build_h(params):
            amps = params[:n_harm]
            phis = params[n_harm:]
            i = np.arange(N)
            h = np.full(N, 0.5)
            for k_idx, k in enumerate(harmonics):
                h += amps[k_idx] * np.cos(2 * np.pi * k * i / N + phis[k_idx])
            return project(h)
        
        # Simulated annealing
        T = 0.05
        T_min = 0.001
        cooling = 0.97
        max_iter = 3000
        
        for it in range(max_iter):
            h = build_h(current_params)
            score = _overlap_max(h, N)
            
            if score < current_score:
                current_score = score
                if score < best_score:
                    best_score = score
                    best_params = current_params.copy()
            
            if T < T_min:
                break
            
            # Propose a move: perturb one amplitude or phase
            dim = np.random.randint(0, 2 * n_harm)
            new_params = current_params.copy()
            if dim < n_harm:
                # Perturb amplitude
                delta = np.random.normal(0, T * 0.5)
                new_params[dim] = np.clip(current_params[dim] + delta, -0.3, 0.3)
            else:
                # Perturb phase
                delta = np.random.normal(0, T * 0.5)
                new_params[dim] = current_params[dim] + delta
            
            new_h = build_h(new_params)
            new_score = _overlap_max(new_h, N)
            
            # Metropolis criterion
            if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
                current_params = new_params
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_params = current_params.copy()
            
            T *= cooling
        
        # Cooldown: local refinement
        for _ in range(500):
            h = build_h(best_params)
            score = _overlap_max(h, N)
            
            # Try small perturbations
            improved = False
            for trial in range(10):
                dim = np.random.randint(0, 2 * len(harmonics))
                new_params = best_params.copy()
                if dim < n_harm:
                    delta = np.random.normal(0, 0.01)
                    new_params[dim] = np.clip(best_params[dim] + delta, -0.3, 0.3)
                else:
                    delta = np.random.normal(0, 0.01)
                    new_params[dim] = best_params[dim] + delta
                
                new_h = build_h(new_params)
                new_score = _overlap_max(new_h, N)
                
                if new_score < score:
                    best_params = new_params
                    score = new_score
                    improved = True
                    break
            
            if not improved:
                break
    
    # Final: fine-tune with direct spatial optimization from the best harmonic solution
    h = build_h(best_params)
    
    # Projected gradient descent refinement
    for step in range(2000):
        h = project(h)
        comp = 1.0 - h
        H = np.fft.fft(h)
        C = np.fft.fft(comp)
        overlap = np.real(np.fft.ifft(np.conj(H) * C))
        
        # Use top few shifts to build a smoother subgradient
        top_k = 5
        top_indices = np.argsort(overlap)[-top_k:]
        
        grad = np.zeros(N)
        for t_star in top_indices:
            for i in range(N):
                grad[i] += 1.0 - h[(i + t_star) % N] - h[(i - t_star) % N]
        grad /= top_k
        
        # Adaptive learning rate
        lr = 0.01 / (1.0 + step / 1000.0)
        h_new = project(h - lr * grad)
        
        if _overlap_max(h_new, N) < _overlap_max(h, N) + 1e-10:
            h = h_new
        else:
            lr *= 0.5
    
    h = project(h)
    return h
# EVOLVE-BLOCK-END
