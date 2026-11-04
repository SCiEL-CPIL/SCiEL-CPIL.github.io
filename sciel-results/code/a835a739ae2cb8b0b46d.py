"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max |f*f| over all lags using FFT."""
    n = len(f)
    conv = np.fft.ifft(np.fft.fft(f, 2*n) ** 2).real
    return np.max(np.abs(conv))


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s
    
    best_f = None
    best_val = np.inf
    
    # Structural change: try multiple diverse initializations and sparsity levels
    # rather than just one alternating pattern
    
    np.random.seed(42)
    
    # List of initializations to try
    initializations = []
    
    # 1. All ones
    initializations.append(np.ones(N))
    
    # 2. Alternating ±1
    alt = np.ones(N)
    alt[::2] = -1
    initializations.append(alt)
    
    # 3. Block patterns of different sizes
    for block_size in [4, 8, 16, 32]:
        f = np.ones(N)
        for start in range(0, N, block_size * 2):
            f[start:start+block_size] = -1
        initializations.append(f.copy())
    
    # 4. Sparse random patterns with different densities
    for density in [0.1, 0.2, 0.3, 0.5]:
        f = np.zeros(N)
        active = np.random.choice(N, size=int(N * density), replace=False)
        f[active] = np.random.choice([-1, 1], size=len(active))
        initializations.append(f)
    
    # 5. Pure random ±1
    initializations.append(np.random.choice([-1, 1], size=N))
    
    # 6. Comb pattern with gaps
    for gap in [3, 5, 7]:
        f = np.zeros(N)
        f[::gap] = np.random.choice([-1, 1], size=len(range(0, N, gap)))
        initializations.append(f)
    
    # Evaluate each initialization and keep the best
    for f_init in initializations:
        if time.time() > deadline - 5:
            break
        val = _autoconvolution_max_abs(f_init)
        if val < best_val:
            best_val = val
            best_f = f_init.copy()
    
    if best_f is None:
        return np.ones(N)
    
    f = best_f.copy()
    
    # Local search: flip individual positions
    # Use a more aggressive approach with multiple passes
    improved = True
    max_sweeps = 50
    sweep = 0
    while improved and sweep < max_sweeps and time.time() < deadline - 5:
        improved = False
        sweep += 1
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = f[i]
            f[i] = -old_val
            val = _autoconvolution_max_abs(f)
            if val < best_val - 1e-12:
                best_val = val
                improved = True
            else:
                f[i] = old_val
    
    # Second phase: try swapping pairs of positions (structural change)
    if time.time() < deadline - 10:
        improved = True
        max_swap_sweeps = 20
        swap_sweep = 0
        while improved and swap_sweep < max_swap_sweeps and time.time() < deadline - 10:
            improved = False
            swap_sweep += 1
            # Try a limited number of random pair swaps per sweep
            num_swaps = min(200, N * 2)
            for _ in range(num_swaps):
                if time.time() > deadline - 10:
                    break
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                old_i, old_j = f[i], f[j]
                f[i], f[j] = old_j, old_i
                val = _autoconvolution_max_abs(f)
                if val < best_val - 1e-12:
                    best_val = val
                    improved = True
                else:
                    f[i], f[j] = old_i, old_j
    
    return f
# EVOLVE-BLOCK-END
