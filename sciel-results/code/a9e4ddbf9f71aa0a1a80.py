"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

def _compute_ratio(heights):
    """Compute the normalized max autoconvolution ratio."""
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    # Autoconvolution via FFT
    # Pad to avoid circular convolution
    size = 2 * n
    f = np.zeros(size)
    f[:n] = h
    conv = np.real(np.fft.ifft(np.fft.fft(f) ** 2))
    # The autoconvolution at lag k is conv[k] for k=0..n-1 (and symmetric)
    # Actually, for linear autoconvolution of a sequence of length n,
    # the result has length 2n-1. Let's use the proper indexing.
    # conv[i] = sum_j h[j]*h[j+i-n+1]... this is getting complex.
    # Simpler: use np.convolve
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    l1_norm = np.sum(h)
    if l1_norm <= 0:
        return 0.0
    return max_conv / (l1_norm * l1_norm)

def _param_to_h(width_frac, power):
    """Generate step heights from a parametric bump function.
    width_frac: fraction of [-1/4,1/4] where the bump is nonzero (0.5 to 1.0)
    power: exponent on the cosine bump
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 1.0/(2*N)  # centers of intervals
    half_width = 0.25 * width_frac
    # Cosine bump: max(0, cos(pi * x / (2*half_width)))^power
    arg = np.abs(x) / (2.0 * half_width)
    arg = np.clip(arg, 0, 1)
    bump = np.cos(np.pi * arg / 2.0) ** power
    bump[arg > 1.0] = 0.0
    return bump.tolist()

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_ratio = _compute_ratio([1.0] * N)
    best_h = [1.0] * N
    
    # Optimize width_frac and power
    # Grid search first
    width_fracs = np.linspace(0.3, 1.0, 20)
    powers = np.linspace(0.5, 4.0, 20)
    
    for wf in width_fracs:
        if time.time() > deadline:
            break
        for pw in powers:
            if time.time() > deadline:
                break
            h = _param_to_h(wf, pw)
            ratio = _compute_ratio(h)
            if ratio < best_ratio:
                best_ratio = ratio
                best_h = h
    
    # Local refinement around best found
    # Find which grid point was best
    best_wf = 0.3
    best_pw = 0.5
    min_ratio = best_ratio
    for wf in width_fracs:
        for pw in powers:
            h = _param_to_h(wf, pw)
            ratio = _compute_ratio(h)
            if ratio < min_ratio - 1e-15:
                min_ratio = ratio
                best_wf = wf
                best_pw = pw
    
    # Refine with finer grid
    wf_range = np.linspace(max(0.2, best_wf - 0.1), min(1.0, best_wf + 0.1), 30)
    pw_range = np.linspace(max(0.1, best_pw - 1.0), min(6.0, best_pw + 1.0), 30)
    
    for wf in wf_range:
        if time.time() > deadline:
            break
        for pw in pw_range:
            if time.time() > deadline:
                break
            h = _param_to_h(wf, pw)
            ratio = _compute_ratio(h)
            if ratio < best_ratio:
                best_ratio = ratio
                best_h = h
                best_wf = wf
                best_pw = pw
    
    # Second refinement
    wf_range2 = np.linspace(max(0.2, best_wf - 0.03), min(1.0, best_wf + 0.03), 30)
    pw_range2 = np.linspace(max(0.1, best_pw - 0.5), min(6.0, best_pw + 0.5), 30)
    
    for wf in wf_range2:
        if time.time() > deadline:
            break
        for pw in pw_range2:
            if time.time() > deadline:
                break
            h = _param_to_h(wf, pw)
            ratio = _compute_ratio(h)
            if ratio < best_ratio:
                best_ratio = ratio
                best_h = h
                best_wf = wf
                best_pw = pw
    
    return best_h
# EVOLVE-BLOCK-END
