import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _soft_max(x, beta):
    """Soft max approximation using log-sum-exp."""
    m = np.max(x)
    if m < -1000:
        return 0.0
    return m + np.log(np.sum(np.exp((x - m) * beta)) / len(x)) / beta

def _soft_max_grad(x, beta):
    """Gradient of soft max."""
    m = np.max(x)
    exp_x = np.exp((x - m) * beta)
    weights = exp_x / np.sum(exp_x)
    return weights

def _spectral_waterfilling(h_init, time_limit=5.0):
    """
    Use the spectral representation of the autoconvolution to optimize.
    The autoconvolution of h is related to the squared magnitude of the DFT of h.
    We want to maximize ||h||_2^2 / (||h||_1 * sup|h*h|).
    
    A key insight: if we think of h as a vector, the autoconvolution peak is bounded by
    the squared l2 norm in the frequency domain. Specifically, for circular convolution,
    the max value is related to the energy distribution.
    
    For linear convolution, we can still use the DFT as a proxy for the spectral content.
    If the energy is spread uniformly in frequency, the time-domain peak is minimized.
    This suggests we want h to be "band-limited" or have a specific spectral shape.
    
    Water-filling approach:
    We iterate by adjusting h to flatten the spectral magnitude |H(f)|^2.
    """
    h = h_init.copy()
    if np.sum(h) < 1e-15:
        return h
    
    start_time = time.time()
    
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Normalize to avoid numerical issues
        l1 = np.sum(h)
        if l1 < 1e-15:
            break
        h_norm = h / l1
        
        # Compute DFT
        H = np.fft.fft(h_norm)
        P = np.abs(H)**2
        
        # We want to equalize P (water-filling)
        # Target: constant power spectrum
        # Update rule: h_new = h_old * (1/sqrt(P)) or similar
        # But we need to keep h non-negative.
        
        # Let's try a gradient step on the log-spectrum
        # We want to minimize the variance of log(P)
        # Or simply, scale h by 1/sqrt(P) in frequency domain
        # H_target = sqrt(target_power) * H / |H|
        # But this changes the phase, which we want to keep.
        # So H_new = H / |H| * sqrt(mean(P))
        
        mean_p = np.mean(P)
        if mean_p < 1e-15:
            break
            
        H_phase = H / np.abs(H)
        H_new = H_phase * np.sqrt(mean_p)
        
        h_new = np.real(np.fft.ifft(H_new))
        
        # Ensure non-negativity
        h_new = np.maximum(h_new, 0)
        
        # Normalize back to original l1
        l1_new = np.sum(h_new)
        if l1_new < 1e-15:
            break
        h_new = h_new * (l1 / l1_new)
        
        # Check for convergence
        if np.max(np.abs(h_new - h)) < 1e-8:
            h = h_new
            break
        else:
            h = h_new
            
    return h

def _optimize_smooth(h_init, time_limit=10.0):
    """
    Optimize using a smooth surrogate for the max in the denominator.
    """
    h0 = h_init.copy()
    if np.sum(h0) < 1e-15:
        return h0, 0.0
    
    def objective(amps, beta=100.0):
        h = np.maximum(amps, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        # Use soft max to approximate sup
        sup_conv = _soft_max(conv, beta)
        if sup_conv < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_conv)
        
    def neg_objective(amps):
        return -objective(amps)
        
    def jac(amps):
        h = np.maximum(amps, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return np.zeros_like(amps)
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        beta = 100.0
        sup_conv = _soft_max(conv, beta)
        if sup_conv < 1e-15:
            return np.zeros_like(amps)
            
        # Gradient of ratio = (2*l1*sup_conv*l2_sq - l2_sq*(l1*grad_sup + l2_sq*grad_l1)) / (l1*sup_conv)^2
        # This is complex. Let's use numerical gradient for simplicity in this step,
        # or just use L-BFGS-B with numerical gradient.
        return None
        
    try:
        res = minimize(neg_objective, h0, method='L-BFGS-B', bounds=[(0, 10)]*N,
                       options={'maxiter': 200, 'ftol': 1e-12})
        h_opt = np.maximum(res.x, 0)
        return h_opt, -res.fun
    except Exception:
        return h0, objective(h0)

def _coordinate_descent(h, max_passes=5, time_limit=10.0):
    """
    Coordinate descent on individual coordinates.
    """
    h = h.copy()
    start_time = time.time()
    
    def optimize_coord(i, h):
        # Fix all other coordinates
        h_fixed = h.copy()
        val = h_fixed[i]
        h_fixed[i] = 0
        
        A = np.sum(h_fixed**2)
        B = np.sum(h_fixed)
        
        if B < 1e-12:
            return val
            
        C = np.convolve(h_fixed, h_fixed, mode='full')
        
        # Contribution of h[i] to convolution
        # conv = C + h[i] * (2 * h_fixed shifted) + h[i]^2 * delta
        # The term 2*h_fixed shifted is a vector of length 2N-1
        D = np.zeros(2*N - 1)
        D[i:i+N] = 2.0 * h_fixed
        
        E = np.zeros(2*N - 1)
        E[2*i] = 1.0
        
        def ratio_func(x):
            if x < 0:
                x = 0.0
            l2_val = A + x*x
            l1_val = B + x
            if l1_val < 1e-15:
                return 0.0
            conv_vals = C + x * D + x*x * E
            sup_conv = np.max(conv_vals)
            if sup_conv < 1e-15:
                return 0.0
            return l2_val / (l1_val * sup_conv)
            
        # Golden Section Search
        gr = (np.sqrt(5) - 1) / 2
        a = 0.0
        b = max(1.0, 2.0 * np.sqrt(A + 1.0) + B)
        
        fa = ratio_func(a)
        fb = ratio_func(b)
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = ratio_func(c)
        fd = ratio_func(d)
        
        for _ in range(50):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = ratio_func(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = ratio_func(c)
                
            if abs(b - a) < 1e-9:
                break
                
        x_opt = (a + b) / 2.0
        return x_opt

    coords = np.arange(N)
    for _ in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        np.random.shuffle(coords)
        for i in coords:
            new_val = optimize_coord(i, h)
            h[i] = new_val
            
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    np.random.seed(42)
    
    # Strategy 1: Sidon-like sets (spikes)
    # We want to maximize the number of non-zero elements while keeping autoconvolution peak low.
    # A Sidon set has distinct differences, which minimizes the off-peak autoconvolution.
    
    def generate_sidon_set(size, max_retries=50):
        best_set = None
        best_energy = float('inf')
        
        for _ in range(max_retries):
            S = set()
            S.add(np.random.randint(0, N))
            
            while len(S) < size:
                added = False
                for _ in range(100):
                    idx = np.random.randint(0, N)
                    if idx in S:
                        continue
                    new_S = S | {idx}
                    diffs = []
                    for i in new_S:
                        for j in new_S:
                            if i != j:
                                diffs.append(i - j)
                    from collections import Counter
                    counts = Counter(diffs)
                    max_count = max(counts.values()) if counts else 0
                    if max_count < 2:
                        S = new_S
                        added = True
                        break
                    elif max_count < 3:
                        # Allow some collisions if we can't find a perfect one
                        if np.random.random() < 0.1:
                            S = new_S
                            added = True
                            break
                
                if not added:
                    break
            
            if len(S) == size:
                h_test = np.zeros(N)
                for i in S:
                    h_test[i] = 1.0
                conv = np.convolve(h_test, h_test, mode='full')
                peak = conv[N-1]
                off_peak = max(np.max(conv[:N-1]), np.max(conv[N:]))
                if off_peak < best_energy:
                    best_energy = off_peak
                    best_set = S
                    
        if best_set is None:
            best_set = set(np.linspace(0, N-1, size, dtype=int))
            
        return sorted(best_set)

    # Try different support sizes
    for size in range(1, 11):
        if time.time() > deadline - 20.0:
            break
            
        for attempt in range(3):
            if time.time() > deadline - 10.0:
                break
                
            support = generate_sidon_set(size)
            h_init = np.zeros(N)
            for i in support:
                h_init[i] = 1.0
            
            # Optimize amplitudes
            h_opt, ratio = _optimize_smooth(h_init, time_limit=5.0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
                
            # Coordinate descent polish
            h_cd = _coordinate_descent(h_opt, max_passes=2, time_limit=2.0)
            ratio_cd = _compute_ratio(h_cd)
            if ratio_cd > best_ratio:
                best_ratio = ratio_cd
                best_h = h_cd.copy()

    # Strategy 2: Spectral Water Filling
    # Start with a few random initializations
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
            
        # Random non-negative initialization
        h_init = np.random.rand(N)
        h_init = h_init / np.sum(h_init)
        
        # Spectral water filling
        h_wf = _spectral_waterfilling(h_init, time_limit=3.0)
        ratio_wf = _compute_ratio(h_wf)
        if ratio_wf > best_ratio:
            best_ratio = ratio_wf
            best_h = h_wf.copy()
            
        # Coordinate descent on the result
        h_cd = _coordinate_descent(h_wf, max_passes=3, time_limit=3.0)
        ratio_cd = _compute_ratio(h_cd)
        if ratio_cd > best_ratio:
            best_ratio = ratio_cd
            best_h = h_cd.copy()

    # Final polish on the best found
    if time.time() < deadline - 5.0:
        h_final = _coordinate_descent(best_h, max_passes=5, time_limit=5.0)
        ratio_final = _compute_ratio(h_final)
        if ratio_final > best_ratio:
            best_ratio = ratio_final
            best_h = h_final.copy()

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
