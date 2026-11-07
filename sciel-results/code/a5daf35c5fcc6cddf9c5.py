import numpy as np
from scipy.optimize import minimize, least_squares
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE_MARGIN = 15.0  # seconds before 150s kill

def _correlation_max(h):
    """Compute max of np.correlate(h, 1-h, mode='full')."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr)

def _project_feasible(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(500):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-13:
            break
        # Adjust only interior points to avoid hitting bounds
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        n_mask = np.sum(mask)
        if n_mask == 0:
            # If all at bounds, spread uniformly
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
            continue
        h[mask] += diff / n_mask
        h = np.clip(h, 0.0, 1.0)
    return h

def _solve_poisson(sources, num_sources):
    """
    Solve discrete Poisson equation: -Laplacian(h) = sources
    with periodic boundary conditions.
    
    The discrete Laplacian on a 1D periodic grid:
    (Laplacian h)[i] = h[i+1] - 2*h[i] + h[i-1]
    
    So -Laplacian(h) = sources means:
    2*h[i] - h[i+1] - h[i-1] = sources[i]
    
    In Fourier space, the Laplacian eigenvalues are:
    lambda_k = -4 * sin^2(pi * k / N)
    
    So -Laplacian has eigenvalues 4*sin^2(pi*k/N) for k=1..N-1,
    and 0 for k=0 (constant mode).
    
    We solve in Fourier space:
    For k=0: h_0 is free (determined by sum constraint)
    For k=1..N-1: h_k = sources_k / (4*sin^2(pi*k/N))
    
    But sources should be zero-mean (k=0 component = 0) for consistency.
    """
    # FFT of sources
    src_fft = np.fft.rfft(sources)
    
    # Eigenvalues of -discrete Laplacian
    k = np.arange(N)
    eigvals = 4.0 * np.sin(np.pi * k / N) ** 2
    eigvals[0] = 1.0  # Avoid division by zero; we'll set h_0 separately
    
    # Solve in Fourier space
    h_fft = src_fft / eigvals
    h_fft[0] = 0.0  # Zero mean for now; sum constraint handled separately
    
    h = np.fft.irfft(h_fft, n=N)
    return h

def _h_from_sources(sources):
    """Convert sparse source vector to height sequence via Poisson solve + projection."""
    # The sources drive the shape; we also add a constant to satisfy sum=TARGET_SUM
    h_shape = _solve_poisson(sources, len(sources))
    
    # Add constant to make sum = TARGET_SUM
    const = TARGET_SUM / N - np.mean(h_shape)
    h = h_shape + const
    
    # Project to feasible set
    h = _project_feasible(h)
    return h

def _objective_from_sources(sources):
    """Objective: max correlation of h derived from sources."""
    h = _h_from_sources(sources)
    return _correlation_max(h)

def _objective_and_h_from_sources(sources):
    h = _h_from_sources(sources)
    return _correlation_max(h), h

def _generate_source_seeds():
    """Generate diverse sparse source vectors."""
    rng = np.random.RandomState(42)
    seeds = []
    
    # Zero source (flat)
    seeds.append(np.zeros(N))
    
    # Single low-frequency sinusoidal sources
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for amp in [0.5, 1.0, 1.5, 2.0]:
            src = np.zeros(N)
            src[freq] = amp
            src[N - freq] = amp if freq != 0 else 0
            seeds.append(src.copy())
            
            # Also cosine-like (real part)
            src_cos = np.zeros(N)
            src_cos[freq] = amp
            src_cos[N - freq] = amp
            seeds.append(src_cos.copy())
    
    # Sparse random sources (few non-zero entries)
    for n_sparse in [2, 3, 5, 8, 12]:
        for _ in range(3):
            src = np.zeros(N)
            idx = rng.choice(N, size=n_sparse, replace=False)
            src[idx] = rng.uniform(-2.0, 2.0, size=n_sparse)
            # Ensure zero mean for consistency
            src -= np.mean(src)
            seeds.append(src)
    
    # Low-pass filtered random sources
    for _ in range(5):
        src = rng.randn(N)
        # Keep only low frequencies
        src_fft = np.fft.rfft(src)
        cutoff = 8
        src_fft[cutoff:] = 0
        src = np.fft.irfft(src_fft, n=N)
        src -= np.mean(src)
        src *= 0.5
        seeds.append(src)
    
    # Two-point dipole sources
    for d in [5, 10, 15, 20, 25]:
        for start in [0, 10, 20, 30, 40]:
            src = np.zeros(N)
            i1 = start % N
            i2 = (start + d) % N
            src[i1] = 1.0
            src[i2] = -1.0
            seeds.append(src.copy())
    
    return seeds

def _polish_h_with_slsqp(h0, time_limit):
    """Polish a height vector using SLSQP with true max objective."""
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
    ]
    
    def obj(x):
        return _correlation_max(x)
    
    try:
        res = minimize(
            obj,
            h0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
        )
        if res.success or (hasattr(res, 'x') and np.all(np.isfinite(res.x))):
            h_res = _project_feasible(res.x)
            return h_res
    except Exception:
        pass
    return h0

def _polish_sources_with_slsqp(sources0, time_limit):
    """Polish source vector using SLSQP."""
    def obj(s):
        return _objective_from_sources(s)
    
    bounds = [(-5.0, 5.0) for _ in range(N)]
    
    try:
        res = minimize(
            obj,
            sources0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or (hasattr(res, 'x') and np.all(np.isfinite(res.x))):
            return res.x
    except Exception:
        pass
    return sources0

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE_MARGIN - elapsed()
    
    # Start with flat seed
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    # Generate diverse source seeds
    source_seeds = _generate_source_seeds()
    
    # Evaluate each source seed and keep best
    for src in source_seeds:
        if elapsed() > DEADLINE_MARGIN - 30:
            break
        h = _h_from_sources(src)
        val = _correlation_max(h)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h.copy()
    
    # Polish the best height directly with SLSQP
    if remaining() > 15:
        best_h = _polish_h_with_slsqp(best_h, remaining())
        best_val = _correlation_max(best_h)
    
    # Try polishing from multiple diverse starting points
    diverse_starts = []
    # Add some sinusoidal starts
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.3, 0.45, 0.49]:
            h_s = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            diverse_starts.append(_project_feasible(h_s))
            h_c = 0.5 + amp * np.cos(2 * np.pi * freq * np.arange(N) / N)
            diverse_starts.append(_project_feasible(h_c))
    
    # Add alternating pattern
    diverse_starts.append(_project_feasible(np.where(np.arange(N) % 2 == 0, 0.25, 0.75)))
    
    # Add from best source-derived h
    diverse_starts.append(best_h.copy())
    
    # Polish a subset of diverse starts
    for i, h0 in enumerate(diverse_starts):
        if elapsed() > DEADLINE_MARGIN - 10:
            break
        if i % 3 == 0:  # Polish every 3rd to save time
            h_polished = _polish_h_with_slsqp(h0, remaining())
            val = _correlation_max(h_polished)
            if val < best_val - 1e-12:
                best_val = val
                best_h = h_polished.copy()
    
    # Final high-precision polish on best_h
    if remaining() > 5:
        best_h = _polish_h_with_slsqp(best_h, remaining())
        best_val = _correlation_max(best_h)
    
    # Final projection to guarantee validity
    best_h = _project_feasible(best_h)
    
    # Strict sum enforcement
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-15:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        best_h = _project_feasible(best_h)
    
    return best_h
