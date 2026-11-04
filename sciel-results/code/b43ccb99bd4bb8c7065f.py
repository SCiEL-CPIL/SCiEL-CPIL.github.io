"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _spectral_design_optimize(deadline, rng):
    """
    Optimize the magnitude spectrum directly.
    Representation: |X(k)| for k=0..N//2 (31 parameters for N=400).
    Use Differential Evolution to search for a flat-ish spectrum that minimizes autoconv peaks.
    """
    n_freqs = N // 2 + 1
    # Initial guess: constant magnitude (flat spectrum)
    # We scale magnitudes such that the resulting time-domain signal has a reasonable sum.
    # If |X(k)| = C for all k, then x(t) is a delta-like impulse in time (for DFT).
    # But we want a step function.
    # Actually, let's just optimize the magnitudes and reconstruct with random phases first,
    # then refine phases? No, let's stick to a simpler model:
    # We optimize the magnitudes M[k].
    # To reconstruct x, we need phases. 
    # A common trick: assume random phases, or optimize phases too?
    # Optimizing 400 phases is hard.
    # Let's try: Optimize magnitudes M[k]. Reconstruct x using random phases from a fixed seed.
    # Then, do a local refinement on x itself.
    
    # Better: Use the property that we can fix the phase to be 0? No, that gives a symmetric peak.
    # Let's use the "random phase" approach but optimize the magnitudes to make the autoconvolution flat.
    
    # Differential Evolution Parameters
    pop_size = 50
    F = 0.8  # Mutation scale factor
    CR = 0.9 # Crossover rate
    
    # Bounds for magnitudes
    # If sum(x) = S, then X(0) = S. Other X(k) should be small to reduce peaks.
    # Let's normalize x such that sum(x) = N (mean 1).
    # Then X(0) = N.
    # We optimize M[1..N//2]. M[0] is fixed to N.
    # M[k] range: [0, N]
    
    # Initialize population
    pop = np.zeros((pop_size, n_freqs))
    pop[:, 0] = N
    # Random initial magnitudes for k>0
    pop[:, 1:] = rng.uniform(0, N * 0.5, size=(pop_size, n_freqs - 1))
    
    # Fixed phases for reconstruction (random but deterministic)
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    phases[0] = 0.0
    
    def eval_spectrum(mags):
        # Construct complex spectrum
        X = mags * np.exp(1j * phases)
        # Inverse FFT
        x = np.fft.irfft(X, _FFT_SIZE)[:N]
        # Check sum
        s = np.sum(x)
        if abs(s) < 1e-6:
            return 1e10
        # Metric
        fx = np.fft.rfft(x, _FFT_SIZE)
        conv = np.fft.irfft(fx * fx, _FFT_SIZE)
        max_abs = np.max(np.abs(conv[:2 * N - 1]))
        return max_abs / (s * s)
    
    # Evaluate initial population
    scores = np.array([eval_spectrum(ind) for ind in pop])
    best_idx = np.argmin(scores)
    best_mags = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    # DE Loop
    max_generations = 200
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        for i in range(pop_size):
            if time.time() > deadline:
                break
            
            # Select 3 random distinct individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            
            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            # Keep M[0] fixed
            v[0] = N
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(n_freqs)
            for j in range(n_freqs):
                if j == j_rand or rng.random() < CR:
                    trial[j] = v[j]
            
            # Evaluate
            trial_score = eval_spectrum(trial)
            
            # Selection
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_mags = trial.copy()
    
    # Reconstruct best x
    X = best_mags * np.exp(1j * phases)
    x_best = np.fft.irfft(X, _FFT_SIZE)[:N]
    
    return x_best, best_score


def _try_local_search_fast(x, deadline, rng, max_rounds=200000):
    """Fast local search with in-place modifications and early termination."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.3
    total_steps = 0
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 200:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.0005:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        # Try multiple perturbations per round
        best_delta = 0
        best_idx = None
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
            total_steps += 1
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Coordinate descent with adaptive step sizes
    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            for delta in step_sizes:
                for sign in [1, -1]:
                    x_test = best_x.copy()
                    x_test[idx] += sign * delta
                    v = _metric(x_test)
                    if v < best_val:
                        best_val = v
                        best_x = x_test
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0

    rng = np.random.default_rng(2024)

    # Start with a reasonable baseline
    best_x = np.ones(N)
    best_val = _metric(best_x)

    # Phase 1: Spectral Design Optimization (Differential Evolution on Magnitudes)
    # This is the main structural change.
    if time.time() < deadline - 40:
        x_spec, val_spec = _spectral_design_optimize(deadline - 40, rng)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec

    # Phase 2: Local Search to refine the spectral result
    # The spectral result might have "jagged" time-domain values due to fixed random phases.
    # Local search can smooth these out.
    if time.time() < deadline - 10:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 10, rng, max_rounds=100000)

    # Phase 3: Final Polish
    if time.time() < deadline - 1:
        best_x, best_val = _final_polish(best_x, deadline - 1, rng)

    return best_x
# EVOLVE-BLOCK-END
