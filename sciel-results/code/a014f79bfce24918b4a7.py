"""
Spectral-domain gradient descent for C3-prime minimization.

Strategy:
  - Represent the signed step sequence as a real vector x of length N.
  - The autoconvolution peak is the DC component of |X(k)|^2 summed? No.
    Actually, autoconvolution c[n] = sum_k x[k] x[k+n].
    The maximum absolute value of c[n] is what we want to minimize, normalized by (sum x)^2.
    
    Key insight: The autoconvolution c is the inverse FFT of |X|^2.
    So max|c| = max_n |IFFT(|X|^2)[n]|.
    
    We can write a smooth surrogate for max|c| using log-sum-exp.
    
    Objective: F(x) = logsumexp(beta * |c|) / (sum x)^2  -- but this is tricky because of the normalization.
    
    Alternative: minimize max|c| subject to sum x = 1 (or fix the integral).
    Since the score is max|c| / (sum x)^2, we can normalize x so that sum x = 1, then just minimize max|c|.
    
    But x must be signed steps: x[i] in {-1, 1}. This is a discrete constraint.
    
    Approach: 
    1. Start with continuous relaxation: x in R^N, minimize soft-max surrogate of max|c| with sum x = 1.
    2. Use projected gradient descent or just gradient descent with normalization.
    3. At the end, round to signs.
    4. Then do local refinement on the discrete domain.

    The gradient of the soft-max surrogate w.r.t. x can be computed via FFT.
    
    Let c = IFFT(|X|^2) where X = FFT(x).
    c[n] = (1/N_fft) sum_k X[k] conj(X[k]) exp(2pi i k n / N_fft)
    
    Actually, let's use the FFT-based autoconvolution.
    
    For the soft-max surrogate:
    S(x) = (1/beta) log(sum_n exp(beta * |c[n]|))
    
    But |c[n]| is not differentiable at 0. We can use c[n]^2 instead:
    S(x) = (1/beta) log(sum_n exp(beta * c[n]^2))
    
    This is differentiable. The gradient w.r.t. x:
    dS/dx_i = (1/beta) * (sum_n 2*beta*c[n] * (d c[n]/d x_i) * exp(beta c[n]^2)) / (sum_n exp(beta c[n]^2))
    
    d c[n]/d x_i: c[n] = sum_j x[j] x[j+n] (with wraparound or linear?).
    For linear autoconvolution, it's more complex. Let's use circular autoconvolution via FFT for simplicity in gradient, then verify with linear.
    
    Actually, the task says "autoconvolution" which typically means linear. But the FFT method in the seed uses linear (with zero-padding).
    
    Let me use a different approach: directly compute the gradient of the soft-max of c[n]^2 where c is the linear autoconvolution.
    
    c[n] = sum_{k} x[k] x[k+n] for n in range -(N-1) to N-1.
    
    This is O(N^2) to compute the gradient directly, which is too slow for N=400 (160k operations per gradient step, times many steps).
    
    Alternative: use the FFT-based circular autoconvolution as a surrogate, optimize in that space, then evaluate the true linear autoconvolution.
    
    Let me try: 
    1. Use circular autoconvolution (via FFT) for the smooth optimization.
    2. The circular autoconvolution c_circ[n] = sum_k x[k] x[(k+n) mod N].
    3. C_circ = IFFT(FFT(x) * conj(FFT(x))).
    4. Optimize soft-max of c_circ[n]^2.
    5. Then round to signs and do local refinement on the true linear autoconvolution.
"""
import numpy as np
import time

N = 400
N_FFT = 1
while N_FFT < 2 * N - 1:
    N_FFT *= 2

def _autoconv_fft(sig):
    """Compute linear autoconvolution of a signal using FFT."""
    f = np.fft.rfft(sig, N_FFT)
    ac = np.fft.irfft(f * np.conj(f), N_FFT)[:2*N-1]
    return ac

def _exact_score(sig):
    """Compute C3' = max|autoconv| / (integral)^2."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(sig)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _circular_autoconv(x):
    """Compute circular autoconvolution via FFT."""
    f = np.fft.fft(x, N)
    c = np.fft.ifft(f * np.conj(f)).real
    return c

def _soft_max_obj_and_grad(x, beta):
    """
    Compute soft-max surrogate objective and gradient for circular autoconv.
    Objective: (1/beta) * log(sum_n exp(beta * c[n]^2))
    where c = circular autoconv of x.
    """
    f = np.fft.fft(x, N)
    c = np.fft.ifft(f * np.conj(f)).real
    
    # Soft-max of c^2
    c2 = c * c
    max_c2 = np.max(c2)
    # Numerical stability
    exps = np.exp(beta * (c2 - max_c2))
    log_sum = max_c2 + np.log(np.sum(exps))
    obj = (1.0 / beta) * log_sum
    
    # Gradient
    # d obj / d x_i
    # c[n] = sum_k x[k] x[(k+n) mod N]
    # d c[n] / d x[i] = x[(i+n) mod N] + x[(i-n) mod N]  (since c[n] = sum x[k]x[k+n], and x[i] appears when k=i or k+n=i)
    # Actually, c[n] = sum_k x[k] x[(k+n) mod N]
    # d c[n]/d x[i] = x[(i+n) mod N] + (if i-n is in range) ... 
    # Let me think again. c[n] = sum_{k=0}^{N-1} x[k] x[(k+n) mod N]
    # For a fixed i, x[i] appears in term k=i: x[i] x[(i+n) mod N]
    # and in term k such that (k+n) mod N = i, i.e., k = (i-n) mod N: x[(i-n) mod N] x[i]
    # So d c[n]/d x[i] = x[(i+n) mod N] + x[(i-n) mod N]
    
    # Gradient of obj w.r.t. c[n]:
    # d obj / d c[n] = (1/beta) * (exp(beta c[n]^2) * 2 beta c[n]) / sum(exp(beta c[m]^2))
    # = 2 c[n] exp(beta c[n]^2) / sum(exp(beta c[m]^2))
    
    weights = 2.0 * c * exps / np.sum(exps)
    
    # grad_x[i] = sum_n weights[n] * (x[(i+n) mod N] + x[(i-n) mod N])
    # This is a convolution: grad_x = weights * x + weights * x[::-1] (circularly)
    # Actually: sum_n weights[n] x[(i+n) mod N] = (weights * x)[i] circular convolution
    # sum_n weights[n] x[(i-n) mod N] = (weights * x_flip)[i] where x_flip[j] = x[-j mod N]
    
    w = np.fft.fft(weights, N)
    xf = np.fft.fft(x, N)
    xflip = np.fft.fft(np.roll(x, -np.arange(N)), N)  # x[-j mod N]
    
    grad_x = np.fft.ifft(w * xf + w * xflip).real
    
    return obj, grad_x

def _spectral_init(seed=0):
    """Generate initial signal from random phase."""
    rng = np.random.RandomState(seed)
    mag = np.ones(N)
    phase = rng.uniform(0, 2*np.pi, size=N)
    phase[0] = 0
    spectrum = mag * np.exp(1j * phase)
    sig = np.fft.ifft(spectrum).real
    integral = np.sum(sig)
    if abs(integral) > 1e-10:
        sig = sig / integral
    sig = np.sign(sig)
    sig[sig == 0] = 1.0
    return sig

def _local_refinement(sig, time_limit=20.0):
    """Local search: flip individual elements to improve score."""
    start_time = time.time()
    best_score = _exact_score(sig)
    
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        order = np.random.permutation(N)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            new_sig = sig.copy()
            new_sig[i] = -new_sig[i]
            s = _exact_score(new_sig)
            
            if s < best_score - 1e-12:
                best_score = s
                sig = new_sig
                improved = True
        
        if not improved:
            break
    
    return sig, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_sig = None
    best_score = float('inf')
    
    # Phase 1: Spectral-domain gradient descent on continuous relaxation
    for seed in range(8):
        if time.time() > deadline - 30.0:
            break
        
        sig = _spectral_init(seed)
        x = sig.astype(float)  # Continuous relaxation
        
        # Gradient descent with adaptive beta
        beta = 5.0
        lr = 0.1
        for step in range(200):
            if time.time() > deadline - 30.0:
                break
            
            obj, grad = _soft_max_obj_and_grad(x, beta)
            
            # Normalize to keep sum ~ 1
            x = x - lr * grad
            s = np.sum(x)
            if abs(s) > 1e-10:
                x = x / s
            
            # Adapt beta
            beta = min(50.0, beta * 1.05)
        
        # Round to signs
        sig_rounded = np.sign(x)
        sig_rounded[sig_rounded == 0] = 1.0
        
        # Local refinement
        refined_sig, refined_score = _local_refinement(sig_rounded, time_limit=15.0)
        
        if refined_score < best_score:
            best_score = refined_score
            best_sig = refined_sig.copy()
    
    # Phase 2: Traditional initializations + local refinement
    if time.time() < deadline - 10.0:
        # Two-block
        sig_tb = np.zeros(N)
        sig_tb[:N//2] = 1.0
        sig_tb[N//2:] = -1.0
        refined_sig, refined_score = _local_refinement(sig_tb, time_limit=10.0)
        if refined_score < best_score:
            best_score = refined_score
            best_sig = refined_sig.copy()
        
        # Three-block
        sig_tb2 = np.zeros(N)
        sig_tb2[:N//3] = 1.0
        sig_tb2[N//3:2*N//3] = -1.0
        sig_tb2[2*N//3:] = 1.0
        refined_sig, refined_score = _local_refinement(sig_tb2, time_limit=10.0)
        if refined_score < best_score:
            best_score = refined_score
            best_sig = refined_sig.copy()
    
    if best_sig is None:
        best_sig = np.zeros(N)
        best_sig[:N//2] = 1.0
        best_sig[N//2:] = -1.0
        best_score = _exact_score(best_sig)
    
    # Phase 3: Final refinement
    if time.time() < deadline - 5.0:
        refined_sig, refined_score = _local_refinement(best_sig, time_limit=20.0)
        if refined_score < best_score:
            best_sig = refined_sig
            best_score = refined_score
    
    return best_sig
