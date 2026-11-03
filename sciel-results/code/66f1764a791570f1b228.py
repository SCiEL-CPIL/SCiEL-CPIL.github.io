"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def compute_overlap(h):
        """Compute max overlap of h with shifted (1-h)."""
        max_ov = 0.0
        for k in range(1, N):
            shifted_1h = 1.0 - np.roll(h, k)
            ov = np.sum(h * shifted_1h)
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    def make_h(f, phase, a, b):
        """Create h from sinusoidal wave clipped to [0,1], scaled to sum to N/2."""
        t = np.arange(N)
        raw = a + b * np.sin(2 * np.pi * f * t / N + phase)
        h = np.clip(raw, 0.0, 1.0)
        # Scale to ensure sum is exactly N/2
        s = np.sum(h)
        if s < 1e-10:
            return np.full(N, 0.5)
        h = h / s * (N / 2.0)
        # Re-clip and adjust
        h = np.clip(h, 0.0, 1.0)
        # Fine-tune to get exact sum
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-10:
            # Distribute adjustment
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
            # Final correction
            diff2 = N / 2.0 - np.sum(h)
            if abs(diff2) > 1e-12:
                idx = np.argsort(h)
                if diff2 > 0:
                    h[idx[-1]] += diff2
                else:
                    h[idx[0]] += diff2
                h = np.clip(h, 0.0, 1.0)
        return h
    
    best_h = np.full(N, 0.5)
    best_score = compute_overlap(best_h)
    
    # Grid search over parameters
    freqs = np.arange(1, 20)
    phases = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    amplitudes = np.linspace(0.1, 0.9, 15)
    offsets = np.linspace(0.3, 0.7, 10)
    
    count = 0
    for f in freqs:
        if time.time() > deadline:
            break
        for phase in phases:
            if time.time() > deadline:
                break
            for a in offsets:
                if time.time() > deadline:
                    break
                for b in amplitudes:
                    if time.time() > deadline:
                        break
                    h = make_h(f, phase, a, b)
                    score = compute_overlap(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
                    count += 1
    
    # Local refinement using gradient-free optimization on best region
    if time.time() < deadline - 30:
        # Do a local search around best parameters found
        # Use simple random perturbation
        # We need to track which params gave best
        # Let's do a more systematic local search
        
        # Try Nelder-Mead style on (f, phase, a, b)
        def objective(params):
            f, phase, a, b = params
            f = max(0.5, min(25.0, f))
            a = max(0.0, min(1.0, a))
            b = max(0.0, min(1.0, b))
            h = make_h(f, phase, a, b)
            return compute_overlap(h)
        
        # Simple local search
        rng = np.random.default_rng(42)
        current_params = [5.0, 0.0, 0.5, 0.5]  # initial guess
        current_obj = objective(current_params)
        
        step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        for ss in step_sizes:
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                perturbation = rng.normal(0, ss, 4)
                new_params = np.array(current_params) + perturbation
                new_params[0] = max(0.5, min(25.0, new_params[0]))
                new_params[1] = (new_params[1] % (2 * np.pi))
                new_params[2] = max(0.0, min(1.0, new_params[2]))
                new_params[3] = max(0.0, min(1.0, new_params[3]))
                new_obj = objective(new_params)
                if new_obj < current_obj:
                    current_params = new_params.tolist()
                    current_obj = new_obj
        
        h_final = make_h(*current_params)
        score_final = compute_overlap(h_final)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final
    
    return best_h
# EVOLVE-BLOCK-END
