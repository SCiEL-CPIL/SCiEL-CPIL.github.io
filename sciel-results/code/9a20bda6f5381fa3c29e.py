"""
Strategy: Spectral design via inverse FFT of a target amplitude spectrum.

Instead of optimizing in the time domain (h-space), we work in the frequency domain.
The autoconvolution c = conv(h, h) has a DFT that is |H|^2 where H = DFT(h).
The peak of c corresponds to the peak of the inverse FFT of |H|^2.
We want to minimize max(c) / sum(h)^2.

Key insight: 
- sum(h) = H[0] (DC component)
- c[0] = sum(h^2) is always positive and is the first element of the full convolution
- Actually, for full convolution of length 2N-1, c[k] for k=0..2N-2
- The peak is often at k=0 (the "center" of the autocorrelation if we think of it that way)
  Wait, conv(h,h) for h of length N gives length 2N-1. 
  c[0] = h[0]^2, c[1] = 2*h[0]*h[1], ..., c[N-1] = sum h[i]^2, ...
  Actually no: c[k] = sum_{j} h[j] h[k-j]
  c[0] = h[0]*h[0] = h[0]^2
  c[1] = h[0]*h[1] + h[1]*h[0] = 2*h[0]*h[1]
  ...
  c[N-1] = sum_{j=0}^{N-1} h[j]*h[N-1-j] ... no, that's correlation.
  
  Let me re-check: np.convolve(h, h) for h=[a,b,c] gives [a^2, 2ab, b^2+2ac, 2bc, c^2]
  So c[0]=h[0]^2, c[1]=2*h[0]*h[1], c[2]=h[1]^2+2*h[0]*h[2], etc.
  
  The maximum is often near the center (k=N-1) where the sum of products is largest.
  c[N-1] = sum_{j=0}^{N-1} h[j]*h[N-1-j] ... no wait.
  
  For h of length N, conv(h,h)[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j]*h[k-j]
  
  At k=N-1: sum_{j=0}^{N-1} h[j]*h[N-1-j] ... no, k-j ranges from N-1 down to 0.
  Actually: j from 0 to N-1, k-j = N-1-j. So c[N-1] = sum_{j=0}^{N-1} h[j]*h[N-1-j].
  This is the "center" of the convolution.
  
  The DC component H[0] = sum(h).
  The energy sum(h^2) relates to sum(|H|^2)/N by Parseval.
  
  Strategy: 
  1. Parameterize H (the DFT of h) directly.
  2. We want to minimize max(ifft(|H|^2)) / H[0]^2 (up to constants).
  3. Note that |H|^2 is the power spectrum. The inverse FFT of |H|^2 is the autocorrelation.
  4. We want the autocorrelation to be as flat as possible (low sidelobes relative to peak).
  
  A classic result: if |H|^2 is constant (all frequencies have equal power), then the 
  autocorrelation is a delta function (only the center peak is non-zero). This is the 
  ideal case, achieved by Zadoff-Chu sequences or similar.
  
  However, we also need H[0] = sum(h) to be large (to make the denominator large).
  
  So the problem becomes: choose a complex vector H of length N (with H[0] real for 
  real h, and H[N-k] = conj(H[k]) for real h) to minimize:
     max(ifft(|H|^2)) / H[0]^2
  
  Since h = ifft(H), we need h to be real. This means H must satisfy conjugate symmetry.
  
  We can parameterize:
  - H[0] is real (DC)
  - H[k] for k=1..N/2 are complex, and H[N-k] = conj(H[k])
  
  Objective: minimize max(ifft(|H|^2)) / H[0]^2
  
  Let's normalize: let H[0] = 1 (we can scale h later, and the objective is scale-invariant
  up to the square... wait, let's check).
  
  value = abs(2*N*max(c) / sum(h)^2)
  c = conv(h,h). If we scale h by alpha, c scales by alpha^2, sum(h) scales by alpha.
  So value scales by alpha^2 / alpha^2 = 1. It's scale-invariant!
  
  So we can fix H[0] = 1 (or any non-zero real value) and optimize the other components.
  
  Actually, let's be more careful. h = ifft(H). sum(h) = H[0]. 
  c = conv(h, h). The DFT of c (padded to length 2N-1) is |H_padded|^2 where H_padded 
  is H zero-padded. But actually, the autocorrelation is ifft(|H|^2) when we use circular 
  convolution. For linear convolution, it's a bit different.
  
  Let's use the fact that for the purpose of this task, we can just compute h = ifft(H)
  and then evaluate the objective directly. The optimization is over the complex 
  amplitudes of H.
  
  We'll use scipy.optimize.minimize with L-BFGS-B on the real and imaginary parts of H.
  
  Constraints: H[0] is real (fix it to 1), H satisfies conjugate symmetry.
  
  Let's define the parameter vector:
  - H[0] = 1 (fixed)
  - For k=1 to floor(N/2): H[k] = a[k] + i*b[k]
  - H[N-k] = a[k] - i*b[k]
  - If N is even, H[N/2] is real: H[N/2] = c[N/2]
  
  N=400 is even. So:
  - H[0] = 1
  - H[1..199] = a[1..199] + i*b[1..199]
  - H[200] = a[200] (real)
  - H[201..399] = conj(H[399..1])
  
  Total parameters: 199*2 + 1 = 399.
  
  Objective: 
    h = ifft(H) (should be real up to numerical precision)
    value = abs(2*N*max(conv(h,h)) / sum(h)^2)
    But sum(h) = H[0] = 1, so value = abs(2*N*max(conv(h,h)))
    
  Wait, that's not right. sum(h) = H[0] only if H[0] is the DC component.
  Actually, ifft(H)[k] = (1/N) * sum_{m=0}^{N-1} H[m] * exp(2*pi*i*m*k/N)
  sum(h) = sum_k h[k] = (1/N) * sum_m H[m] * sum_k exp(2*pi*i*m*k/N)
  The inner sum is N if m=0, else 0. So sum(h) = H[0]. Yes.
  
  So if we fix H[0]=1, value = abs(2*N*max(conv(h,h))).
  
  We want to minimize this. The minimum possible is 0 if we can make max(conv(h,h)) = 0,
  but that's impossible since conv(h,h) has at least one non-zero element (unless h=0).
  
  Actually, we can make the peak very small relative to... wait, no, the value is 
  2*N*max(c) since sum(h)=1. We want to minimize max(c).
  
  But c[0] = h[0]^2 >= 0. And c is the autoconvolution. The sum of all c[k] = (sum h)^2 = 1.
  So the average of c[k] is 1/(2N-1). The maximum must be at least 1/(2N-1).
  
  So the minimum possible value is 2*N/(2N-1) ≈ 1. For N=400, that's 800/799 ≈ 1.00125.
  
  This is a great lower bound! If we can achieve a "flat" autocorrelation, we get close to 1.
  
  Let's implement this: optimize the frequency components to make the autocorrelation 
  as flat as possible.
"""

import numpy as np
import scipy.optimize
import time

N = 400
_DEADLINE = 145.0

def _exact_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _build_H(params):
    """
    Build the DFT vector H from parameters.
    params: array of length 399
      params[0:199] = real parts of H[1..199]
      params[199:398] = imag parts of H[1..199]
      params[398] = H[200] (real)
    """
    H = np.zeros(N, dtype=complex)
    H[0] = 1.0
    for k in range(1, N//2):
        H[k] = params[k-1] + 1j * params[N//2 - 1 + k - 1]
        H[N-k] = np.conj(H[k])
    H[N//2] = params[N-2]
    return H

def _objective(params):
    """Compute the objective value from frequency parameters."""
    H = _build_H(params)
    h = np.fft.ifft(H).real
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _objective_and_grad(params):
    """Compute objective and gradient via finite differences (for simplicity)."""
    val = _objective(params)
    grad = np.zeros_like(params)
    eps = 1e-7
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _objective(params_plus)
        grad[i] = (val_plus - val) / eps
    return val, grad

def construct():
    """
    Construct step heights using spectral design.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initialize with a Zadoff-Chu-like sequence
    # Zadoff-Chu: h[n] = exp(-i*pi*n*(n+1)/N) for N odd, or similar
    # For real sequences, we can use a chirp: h[n] = cos(2*pi*alpha*n^2/N)
    
    rng = np.random.default_rng(42)
    
    # Try several initializations in frequency domain
    best_params = None
    best_val = np.inf
    
    # Init 1: All zeros except DC (gives h = constant)
    params0 = np.zeros(N-1)
    v0 = _objective(params0)
    if v0 < best_val:
        best_val = v0
        best_params = params0.copy()
    
    # Init 2: Random
    for init_idx in range(5):
        if time.time() > deadline - 60:
            break
        params_r = rng.normal(0, 0.1, N-1)
        v_r = _objective(params_r)
        if v_r < best_val:
            best_val = v_r
            best_params = params_r.copy()
    
    # Init 3: Chirp-like spectrum
    # We want |H|^2 to be as flat as possible
    # Start with flat spectrum: |H[k]| = 1 for all k
    # H[k] = exp(i*phase[k])
    # For real h, H must be conjugate symmetric
    # Let's try H[k] = exp(i*2*pi*k*alpha/N) for a linear phase (chirp in time)
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        if time.time() > deadline - 60:
            break
        H_flat = np.ones(N, dtype=complex)
        for k in range(N):
            H_flat[k] = np.exp(2j * np.pi * alpha * k * k / N)
        # Enforce conjugate symmetry
        H_sym = np.zeros(N, dtype=complex)
        H_sym[0] = 1.0
        for k in range(1, N//2):
            H_sym[k] = H_flat[k]
            H_sym[N-k] = np.conj(H_flat[k])
        H_sym[N//2] = H_flat[N//2].real
        
        # Extract params from H_sym
        params_c = np.zeros(N-1)
        for k in range(1, N//2):
            params_c[k-1] = H_sym[k].real
            params_c[N//2 - 1 + k - 1] = H_sym[k].imag
        params_c[N-2] = H_sym[N//2].real
        
        v_c = _objective(params_c)
        if v_c < best_val:
            best_val = v_c
            best_params = params_c.copy()
    
    # Optimize using L-BFGS-B
    if best_params is not None and time.time() < deadline - 30:
        # Use numerical gradient for simplicity
        def obj_func(p):
            return _objective(p)
        
        # Run L-BFGS-B
        try:
            result = scipy.optimize.minimize(
                obj_func,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Try a few more random restarts
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            params_r = rng.normal(0, 0.05, N-1)
            try:
                result = scipy.optimize.minimize(
                    obj_func,
                    params_r,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
    
    # Convert best params to h
    if best_params is not None:
        H = _build_H(best_params)
        h = np.fft.ifft(H).real
    else:
        h = np.ones(N)
    
    # Validate and return
    final_val = _exact_value(h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        h = np.ones(N)
    
    return h
