import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill


def _evaluate(h):
    """
    Computes the objective value exactly as defined in the task.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    if not np.all(np.isfinite(h)):
        return np.inf
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (S * S))


def _zadoff_chu(N, root):
    """
    Generate a Zadoff-Chu sequence of length N with given root.
    For N even, root must be coprime with N.
    zc[i] = exp(-j * pi * root * i * (i+1) / N)
    We take the real part for our real-valued step heights.
    """
    i = np.arange(N)
    # Standard Zadoff-Chu: zc[n] = exp(-j*pi*alpha*n*(n+1)/N) for N even
    # We use the real part to get a real signal
    zc = np.exp(-1j * np.pi * root * i * (i + 1) / N)
    return np.real(zc)


def _construct_h(params):
    """
    Construct h from parameters:
    params = [root1, root2, shift1, shift2, phase1, phase2, amp1, amp2, offset]
    
    h = amp1 * real(rotate(zc1_shifted, phase1)) + amp2 * real(rotate(zc2_shifted, phase2)) + offset
    """
    root1 = int(round(params[0]))
    root2 = int(round(params[1]))
    shift1 = int(round(params[2])) % N
    shift2 = int(round(params[3])) % N
    phase1 = params[4]
    phase2 = params[5]
    amp1 = params[6]
    amp2 = params[7]
    offset = params[8]
    
    # Ensure roots are valid (coprime with N for even N)
    # For N=400, valid roots are odd numbers not divisible by 5
    # Actually for Zadoff-Chu of length N (even), alpha must be coprime to N
    # Let's just use them and take real part; the structure still helps
    zc1 = _zadoff_chu(N, root1)
    zc2 = _zadoff_chu(N, root2)
    
    # Time shift
    zc1_shifted = np.roll(zc1, shift1)
    zc2_shifted = np.roll(zc2, shift2)
    
    # Phase rotation (multiply by exp(j*phase) then take real)
    # This is equivalent to a time shift in frequency domain
    # But for real signals, we just modulate
    h1 = amp1 * (zc1_shifted * np.cos(phase1) - np.imag(zc1_shifted * np.exp(1j*phase1)))
    # Simpler: just use the real part with a phase offset as a mixing
    # Actually, let's keep it simple: use the real Zadoff-Chu directly
    # and add a sinusoidal modulation for the "phase" parameter
    
    # Simpler approach: h = amp1 * zc1_shifted + amp2 * zc2_shifted + offset
    # The "phase" parameters will be used to modulate with a carrier
    # to create more flexibility
    
    # Let's use a cleaner formulation:
    # h[i] = amp1 * zc1[(i + shift1) % N] * cos(2*pi*phase1*i/N) 
    #       + amp2 * zc2[(i + shift2) % N] * cos(2*pi*phase2*i/N) + offset
    
    i = np.arange(N)
    h = (amp1 * zc1_shifted * np.cos(2 * np.pi * phase1 * i / N) +
         amp2 * zc2_shifted * np.cos(2 * np.pi * phase2 * i / N) +
         offset)
    
    return h


def _objective(params):
    h = _construct_h(params)
    return _evaluate(h)


def _directed_refinement(h, max_iter=300, deadline=140.0):
    """
    Directed perturbation search: exploit the structure of the autoconvolution.
    For each candidate index, try to reduce the peak of the autoconvolution.
    Use gradient-free but informed moves: perturb indices that contribute most
    to the current peak of the autoconvolution.
    """
    best_h = h.copy()
    best_val = _evaluate(best_h)
    
    scale = np.std(best_h)
    if scale < 1e-6:
        scale = 1.0
    step = 0.05 * scale
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute current autoconvolution to find where the peak is
        conv = np.convolve(best_h, best_h)
        peak_idx = np.argmax(conv)
        
        # The peak at lag k means h[i] * h[i+k] is large for many i
        # To reduce it, we can perturb h at positions that contribute to this peak
        # For lag k, the contributing pairs are (i, i+k) for i=0..N-1-k
        # Perturb h at positions near the "center" of these pairs
        
        # Strategy: pick a few indices near the peak-contributing region
        # and try to reduce the product
        k = peak_idx  # lag of the peak
        if k < N:
            # Indices that contribute: i and i+k for i in range(N-k)
            # Perturb h[i] for random i in this range
            n_contrib = N - k
            if n_contrib > 0:
                n_perturb = min(5, n_contrib)
                indices = np.random.choice(n_contrib, size=n_perturb, replace=False)
                for idx in indices:
                    # Try to reduce h[idx] * h[idx+k] by adjusting h[idx]
                    # If h[idx] * h[idx+k] > 0, decrease |h[idx]|
                    # If h[idx] * h[idx+k] < 0, increase |h[idx]|
                    partner = idx + k
                    if partner < N:
                        prod = best_h[idx] * best_h[partner]
                        if prod > 0:
                            # Try decreasing h[idx] if positive, increasing if negative
                            direction = -np.sign(best_h[idx])
                        else:
                            direction = np.sign(best_h[idx])
                        
                        h_test = best_h.copy()
                        h_test[idx] += direction * step
                        val = _evaluate(h_test)
                        if val < best_val:
                            best_h = h_test
                            best_val = val
        
        # Also try random perturbations for exploration
        for _ in range(3):
            idx = np.random.randint(N)
            h_test = best_h.copy()
            h_test[idx] += np.random.choice([-1, 1]) * step
            val = _evaluate(h_test)
            if val < best_val:
                best_h = h_test
                best_val = val
        
        # Adaptive step size
        if iteration % 20 == 19:
            # Check if we're making progress
            step *= 0.9
            if step < 1e-7:
                break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    
    # Baseline
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    np.random.seed(42)
    
    # Zadoff-Chu roots that are coprime with N=400
    # 400 = 2^4 * 5^2, so roots must be odd and not divisible by 5
    valid_roots = [r for r in range(1, N, 2) if r % 5 != 0 and np.gcd(r, N) == 1]
    
    # Try several configurations of two Zadoff-Chu sequences
    configs = []
    
    # Configuration 1: Two different roots, no shifts
    if len(valid_roots) >= 2:
        configs.append([valid_roots[0], valid_roots[1], 0, 0, 0.0, 0.0, 1.0, 1.0, 0.0])
        configs.append([valid_roots[0], valid_roots[2], 0, 0, 0.0, 0.0, 1.0, -1.0, 0.0])
    
    # Configuration 2: Same root, different shifts (creates beat pattern)
    if len(valid_roots) >= 1:
        r = valid_roots[0]
        configs.append([r, r, 0, N//4, 0.0, 0.0, 1.0, 1.0, 0.0])
        configs.append([r, r, 0, N//3, 0.0, np.pi, 1.0, -1.0, 0.0])
    
    # Configuration 3: With phase modulation
    if len(valid_roots) >= 2:
        configs.append([valid_roots[0], valid_roots[1], N//8, 3*N//8, 0.5, 0.3, 1.0, 1.0, 0.1])
        configs.append([valid_roots[1], valid_roots[3] if len(valid_roots)>3 else valid_roots[0], 
                        N//5, 2*N//5, 0.2, 0.7, 0.8, 1.2, -0.1])
    
    # Configuration 4: Larger amplitudes with offset
    if len(valid_roots) >= 2:
        configs.append([valid_roots[0], valid_roots[1], 0, 0, 0.0, 0.0, 2.0, 1.0, 0.5])
        configs.append([valid_roots[2], valid_roots[4] if len(valid_roots)>4 else valid_roots[1], 
                        N//6, N//3, 0.1, 0.4, 1.5, 0.8, -0.2])
    
    # Random configurations
    for _ in range(5):
        r1 = valid_roots[np.random.randint(len(valid_roots))]
        r2 = valid_roots[np.random.randint(len(valid_roots))]
        s1 = np.random.randint(N)
        s2 = np.random.randint(N)
        p1 = np.random.rand()
        p2 = np.random.rand()
        a1 = np.random.uniform(0.5, 2.0)
        a2 = np.random.uniform(0.5, 2.0)
        off = np.random.uniform(-0.5, 0.5)
        configs.append([r1, r2, s1, s2, p1, p2, a1, a2, off])
    
    # Optimize each configuration
    for config in configs:
        if time.time() > DEADLINE - 30:
            break
        
        # Use Nelder-Mead for the continuous parameters (shifts, phases, amps, offset)
        # Keep roots fixed (they're discrete)
        x0 = np.array(config, dtype=float)
        
        # Define bounds
        bounds = [
            (1, N-1),      # root1 (will be rounded)
            (1, N-1),      # root2
            (0, N-1),      # shift1
            (0, N-1),      # shift2
            (0, 1.0),      # phase1
            (0, 1.0),      # phase2
            (0.1, 5.0),    # amp1
            (0.1, 5.0),    # amp2
            (-2.0, 2.0)    # offset
        ]
        
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            h_candidate = _construct_h(res.x)
            val = _evaluate(h_candidate)
            
            if val < best_val and np.all(np.isfinite(h_candidate)):
                best_val = val
                best_h = h_candidate.copy()
                
        except Exception:
            # Fallback: just use the initial config
            h_candidate = _construct_h(x0)
            val = _evaluate(h_candidate)
            if val < best_val and np.all(np.isfinite(h_candidate)):
                best_val = val
                best_h = h_candidate.copy()
    
    # Directed refinement on the best candidate
    if time.time() < DEADLINE - 10:
        refined_h, refined_val = _directed_refinement(best_h, max_iter=200, deadline=DEADLINE - 3)
        if refined_val < best_val:
            best_val = refined_val
            best_h = refined_h
    
    # Final validation
    final_val = _evaluate(best_h)
    if final_val != np.inf and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-12:
        return best_h
    else:
        return np.ones(N)


if __name__ == "__main__":
    h = construct()
    val = _evaluate(h)
    print(f"Final Value: {val}")
    print(f"Score: {2.0 / val}")
