"""Rewritten program: Fourier-domain convex optimization for Erdos minimum overlap."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0


def _overlaps_fft(h):
    """S(k) = sum_i h[i] * (1 - h[(i+k) mod N]) for all shifts k using FFT."""
    n = len(h)
    one_minus = 1.0 - h
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus)
    S = np.fft.ifft(np.conj(H) * O).real
    return S


def _project(h, n):
    """Project h onto [0,1]^n with sum = n/2."""
    h = np.clip(h, 0.0, 1.0)
    h += (n / 2.0 - h.sum()) / n
    h = np.clip(h, 0.0, 1.0)
    h += (n / 2.0 - h.sum()) / n
    return h


def _fourier_to_h(c, n):
    """Convert Fourier coefficients c (complex, length n) to real h of length n."""
    h = np.fft.ifft(c).real
    return h


def _h_to_fourier(h, n):
    """Convert real h to Fourier coefficients."""
    return np.fft.fft(h)


def _optimize_fourier(n, max_iter=5000, tol=1e-8):
    """
    Optimize in Fourier domain.
    
    Key insight: The overlap S(k) = sum_i h[i]*(1-h[(i+k)%n]) can be written
    in terms of Fourier coefficients. We minimize the maximum S(k) subject to
    0 <= h[i] <= 1 and sum(h) = n/2.
    
    We use a projected gradient method in Fourier space with careful handling
    of the box constraints via subgradient projection.
    """
    # Initialize with a structured pattern in real space, then go to Fourier
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = 0.5 + 0.4 * np.sin(2 * t)  # Period-2 pattern
    h = _project(h, n)
    
    # Work in Fourier domain
    c = _h_to_fourier(h, n)
    
    best_h = h.copy()
    best_max_overlap = np.max(_overlaps_fft(h))
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        h = _fourier_to_h(c, n)
        h = _project(h, n)
        
        S = _overlaps_fft(h)
        max_ov = np.max(S)
        
        if max_ov < best_max_overlap:
            best_max_overlap = max_ov
            best_h = h.copy()
        
        # Gradient of max(S) w.r.t. h: use softmax approximation
        beta = 50.0
        m = np.max(S)
        e = np.exp(beta * (S - m))
        e = e / e.sum()
        
        # dS(k)/dh[i] = (1 - h[(i+k)%n]) - h[(i-k)%n]
        grad_h = np.zeros(n)
        for k in range(n):
            ek = e[k]
            if ek < 1e-10:
                continue
            grad_h += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
        grad_h -= grad_h.mean()  # Remove mean component (sum constraint)
        
        # Transform gradient to Fourier domain
        grad_c = _h_to_fourier(grad_h, n)
        
        # Gradient descent step in Fourier domain
        c -= lr * grad_c
        
        # Periodically project back to feasible region
        if it % 10 == 0:
            h_check = _fourier_to_h(c, n)
            h_check = _project(h_check, n)
            c = _h_to_fourier(h_check, n)
        
        # Adaptive learning rate
        S_new = _overlaps_fft(_project(_fourier_to_h(c, n), n))
        if np.max(S_new) < max_ov:
            lr = min(lr * 1.05, 0.05)
        else:
            lr = max(lr * 0.7, 0.001)
    
    return best_h, best_max_overlap


def _optimize_real_space(n, max_iter=3000):
    """
    Optimize in real space with a different approach: coordinate descent
    with overlap-aware updates.
    """
    # Start with a good initial pattern
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = 0.5 + 0.45 * np.sin(2 * t)
    h = _project(h, n)
    
    best_h = h.copy()
    best_max_ov = np.max(_overlaps_fft(h))
    
    lr = 0.02
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        S = _overlaps_fft(h)
        max_ov = np.max(S)
        
        if max_ov < best_max_ov:
            best_max_ov = max_ov
            best_h = h.copy()
        
        # Softmax-weighted gradient
        beta = 30.0
        m = np.max(S)
        e = np.exp(beta * (S - m))
        e = e / e.sum()
        
        grad = np.zeros(n)
        for k in range(n):
            ek = e[k]
            if ek < 1e-10:
                continue
            grad += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
        grad -= grad.mean()
        
        # Line search
        for step_factor in [1.0, 0.5, 0.25, 0.1]:
            h_try = _project(h - lr * step_factor * grad, n)
            S_try = _overlaps_fft(h_try)
            if np.max(S_try) < max_ov:
                h = h_try
                break
        
        # Adaptive lr
        S_curr = _overlaps_fft(h)
        if np.max(S_curr) < max_ov:
            lr = min(lr * 1.03, 0.1)
        else:
            lr = max(lr * 0.8, 0.001)
    
    return best_h, best_max_ov


def _optimize_mixed(n):
    """
    Mixed approach: alternate between Fourier and real-space updates.
    """
    # Initialize
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = 0.5 + 0.4 * np.cos(2 * t) + 0.1 * np.sin(4 * t)
    h = _project(h, n)
    
    best_h = h.copy()
    best_max_ov = np.max(_overlaps_fft(h))
    
    lr_f = 0.005  # Fourier learning rate
    lr_r = 0.015  # Real learning rate
    
    for it in range(4000):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Every 20 iterations, do a Fourier-space step
        if it % 20 == 0:
            c = _h_to_fourier(h, n)
            S = _overlaps_fft(h)
            beta = 40.0
            m = np.max(S)
            e = np.exp(beta * (S - m))
            e = e / e.sum()
            
            grad_h = np.zeros(n)
            for k in range(n):
                ek = e[k]
                if ek < 1e-10:
                    continue
                grad_h += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
            grad_h -= grad_h.mean()
            
            grad_c = _h_to_fourier(grad_h, n)
            c -= lr_f * grad_c
            h = _project(_fourier_to_h(c, n), n)
        else:
            # Real-space step
            S = _overlaps_fft(h)
            beta = 30.0
            m = np.max(S)
            e = np.exp(beta * (S - m))
            e = e / e.sum()
            
            grad = np.zeros(n)
            for k in range(n):
                ek = e[k]
                if ek < 1e-10:
                    continue
                grad += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
            grad -= grad.mean()
            
            h_new = _project(h - lr_r * grad, n)
            S_new = _overlaps_fft(h_new)
            if np.max(S_new) < np.max(S):
                h = h_new
                lr_r = min(lr_r * 1.02, 0.05)
            else:
                lr_r = max(lr_r * 0.85, 0.001)
        
        max_ov = np.max(_overlaps_fft(h))
        if max_ov < best_max_ov:
            best_max_ov = max_ov
            best_h = h.copy()
    
    return best_h, best_max_ov


def _try_structured_patterns(n):
    """
    Try several analytically-motivated patterns and pick the best.
    """
    candidates = []
    
    # Pattern 1: Pure period-2 (alternating)
    h1 = np.zeros(n)
    h1[::2] = 0.7
    h1[1::2] = 0.3
    h1 = _project(h1, n)
    candidates.append(h1)
    
    # Pattern 2: Sine with period 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h2 = 0.5 + 0.45 * np.sin(2 * t)
    h2 = _project(h2, n)
    candidates.append(h2)
    
    # Pattern 3: Cosine with period 2
    h3 = 0.5 + 0.45 * np.cos(2 * t)
    h3 = _project(h3, n)
    candidates.append(h3)
    
    # Pattern 4: Multi-frequency
    h4 = 0.5 + 0.35 * np.sin(2 * t) + 0.15 * np.sin(4 * t)
    h4 = _project(h4, n)
    candidates.append(h4)
    
    # Pattern 5: Sawtooth-like
    h5 = np.zeros(n)
    for i in range(n):
        frac = (i % 4) / 4.0
        h5[i] = 0.5 + 0.4 * (2 * frac - 1)
    h5 = _project(h5, n)
    candidates.append(h5)
    
    # Pattern 6: Block pattern
    h6 = np.zeros(n)
    block = n // 4
    for i in range(n):
        if (i % (2 * block)) < block:
            h6[i] = 0.65
        else:
            h6[i] = 0.35
    h6 = _project(h6, n)
    candidates.append(h6)
    
    best_h = candidates[0]
    best_ov = np.max(_overlaps_fft(best_h))
    for h in candidates[1:]:
        ov = np.max(_overlaps_fft(h))
        if ov < best_ov:
            best_ov = ov
            best_h = h
    
    return best_h, best_ov


def construct():
    t0 = time.time()
    n = N
    
    best_h = np.full(n, 0.5)
    best_val = np.max(_overlaps_fft(best_h))
    
    # Phase 1: Try structured patterns
    h_struct, ov_struct = _try_structured_patterns(n)
    if ov_struct < best_val:
        best_val = ov_struct
        best_h = h_struct.copy()
    
    # Phase 2: Fourier-domain optimization
    if time.time() < _DEADLINE - 30.0:
        h_f, ov_f = _optimize_fourier(n, max_iter=3000)
        if ov_f < best_val:
            best_val = ov_f
            best_h = h_f.copy()
    
    # Phase 3: Real-space optimization from best so far
    if time.time() < _DEADLINE - 25.0:
        h_r, ov_r = _optimize_real_space(n, max_iter=2000)
        if ov_r < best_val:
            best_val = ov_r
            best_h = h_r.copy()
    
    # Phase 4: Mixed optimization
    if time.time() < _DEADLINE - 20.0:
        h_m, ov_m = _optimize_mixed(n)
        if ov_m < best_val:
            best_val = ov_m
            best_h = h_m.copy()
    
    # Phase 5: Polish from the best solution found
    if time.time() < _DEADLINE - 10.0:
        h = best_h.copy()
        for it in range(2000):
            if time.time() > _DEADLINE - 3.0:
                break
            
            S = _overlaps_fft(h)
            max_ov = np.max(S)
            
            if max_ov < best_val:
                best_val = max_ov
                best_h = h.copy()
            
            beta = 80.0
            m = np.max(S)
            e = np.exp(beta * (S - m))
            e = e / e.sum()
            
            grad = np.zeros(n)
            for k in range(n):
                ek = e[k]
                if ek < 1e-10:
                    continue
                grad += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
            grad -= grad.mean()
            
            for step_factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
                h_try = _project(h - 0.005 * step_factor * grad, n)
                S_try = _overlaps_fft(h_try)
                if np.max(S_try) < max_ov:
                    h = h_try
                    break
    
    # Final projection
    best_h = _project(best_h, n)
    
    return best_h
# EVOLVE-BLOCK-END
