"""Direct optimization of step heights using L-BFGS-B and soft-max peak surrogate.

We optimize the 600 heights directly. The objective is the ratio of the
soft-max approximation of the autoconvolution peak to the sum of heights.
This removes the basis restriction of the previous Fourier approach.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
# interval width
w = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # 0.5/600

# Pre-compute FFT size
CONV_SIZE = 1
while CONV_SIZE < 2 * N - 1:
    CONV_SIZE <<= 1

def _autoconv(f):
    F = np.fft.rfft(f, n=CONV_SIZE)
    conv = np.fft.irfft(F * F, n=CONV_SIZE)[:2 * N - 1]
    return conv

def _soft_max(x, beta):
    # Stable soft max
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _objective_and_grad(f, beta):
    # f is shape (N,)
    # Energy
    energy = np.sum(f)
    if energy < 1e-12:
        return 1e10, np.zeros_like(f)
    
    conv = _autoconv(f)
    
    # Soft max of conv
    # Derivative of soft_max(x, beta) w.r.t x is softmax(x*beta)
    # softmax(z) = exp(z) / sum(exp(z))
    # z = beta * (conv - max_conv)
    m = np.max(conv)
    z = beta * (conv - m)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    softmax_probs = exp_z / sum_exp_z  # shape (2N-1,)
    
    peak_val = m + np.log(sum_exp_z) / beta
    
    # Objective: peak_val / energy
    val = peak_val / energy
    
    # Gradient
    # d(val)/d(conv[i]) = (1/energy) * softmax_probs[i]
    dval_dconv = softmax_probs / energy
    
    # d(conv[i])/d(f[m])
    # conv[i] = sum_k f[k] f[i-k]
    # d(conv[i])/d(f[m]) = f[i-m] + f[m-i] (with bounds check)
    # We need dval/df[m] = sum_i dval_dconv[i] * dconv[i]/df[m]
    
    # This is equivalent to the correlation of dval_dconv with f, but scaled?
    # Let g = dval_dconv (length 2N-1)
    # dval/df[m] = sum_i g[i] (f[i-m] + f[m-i])
    # Note that sum_i g[i] f[i-m] is the cross-correlation of g and f at lag m.
    # Since g is the gradient of the peak w.r.t conv, and conv is quadratic in f,
    # the gradient w.r.t f is related to the correlation of g and f.
    
    # Specifically, let h = g.
    # term1[m] = sum_i h[i] f[i-m]
    # term2[m] = sum_i h[i] f[m-i]
    # Note term2[m] is sum_j h[j] f[m-j] where j=i.
    # If h is symmetric (which it is, because conv is symmetric around N-1 and f is real, 
    # but wait, conv is symmetric only if f is symmetric? No, conv is always symmetric 
    # around its center for auto-convolution? 
    # conv[k] = sum f[i]f[k-i]. conv[2N-2-k] = sum f[i]f[2N-2-k-i]. 
    # Let j = 2N-2-k-i. Then i = 2N-2-k-j.
    # conv[2N-2-k] = sum f[2N-2-k-j] f[j] = sum f[j] f[2N-2-k-j] = conv[k].
    # Yes, auto-convolution is symmetric.
    # Therefore g (softmax of conv) is symmetric.
    # So term1[m] == term2[m].
    # dval/df[m] = 2 * sum_i g[i] f[i-m]
    
    # We can compute this via FFT.
    # cross_corr = ifft(fft(g) * conj(fft(f))) ? No, that's circular.
    # We need linear correlation.
    
    # Let's compute correlation of g (length 2N-1) and f (length N).
    # Result length 2N-1 + N - 1 = 3N-2?
    # Actually, we want sum_i g[i] f[i-m] for m in 0..N-1.
    # This is the cross-correlation of g and f.
    
    # FFT size for correlation
    corr_size = 1
    while corr_size < (2 * N - 1) + N - 1:
        corr_size <<= 1
        
    G = np.fft.rfft(g, n=corr_size)
    Ff = np.fft.rfft(f, n=corr_size)
    
    # Correlation is ifft(conj(G) * Ff) ?
    # sum g[i] f[i-m]
    # Let's check indices.
    # If we compute ifft(conj(fft(g)) * fft(f)), we get sum g[i] f[i+m] (circular).
    # For linear correlation:
    # We want c[m] = sum_i g[i] f[i-m]
    # This is equivalent to convolution of g with reversed f?
    # Let's just use np.correlate if N is small, but N=600 is small enough for direct?
    # 600 * 1200 = 720,000 ops. Very fast.
    
    # g is length 2N-1. f is length N.
    # We want sum_i g[i] f[i-m] for m=0..N-1.
    # Let's reverse f to do convolution?
    # conv(g, rev_f)[k] = sum_i g[i] rev_f[k-i] = sum_i g[i] f[N-1-(k-i)] = sum_i g[i] f[N-1-k+i]
    # Let j = N-1-k+i => i = j - N + 1 + k
    # This is getting messy.
    
    # Direct calculation is safer and fast enough for 600.
    # g: length 1199. f: length 600.
    # We need grad_f[m] for m in 0..599.
    # grad_f[m] = 2 * sum_{i=0}^{1198} g[i] * f[i-m] (where f[x]=0 if x<0 or x>=600)
    
    grad_f = np.zeros(N)
    # Optimization: loop over m
    for m in range(N):
        # i ranges such that 0 <= i-m < N => m <= i < m+N
        i_start = m
        i_end = m + N
        if i_end > len(g):
            i_end = len(g)
        if i_start < 0:
            i_start = 0
        # f[i-m] is f[0..N-1]
        # g[i_start..i_end-1]
        # f[0..N-1]
        # We need sum g[i] * f[i-m]
        # Let k = i-m. i = k+m.
        # k ranges 0..N-1.
        # g[k+m] * f[k]
        
        # Vectorized:
        # g_slice = g[i_start:i_end]
        # f_slice = f[0: i_end - i_start] ? No.
        # If i = k+m, then k = i-m.
        # When i = i_start (m), k=0.
        # When i = i_end-1, k = i_end-1-m.
        
        # Let's just use a simple loop or np.dot on slices.
        # The valid k are 0 .. min(N-1, len(g)-1-m)
        max_k = min(N - 1, len(g) - 1 - m)
        if max_k >= 0:
            # g indices: m .. m + max_k
            # f indices: 0 .. max_k
            grad_f[m] = 2.0 * np.dot(g[m:m + max_k + 1], f[0:max_k + 1])
            
    # Subtract term from energy derivative
    # d(val)/df[m] = dval_dconv part - (val/energy) * d(energy)/df[m]
    # d(energy)/df[m] = 1
    grad_f -= val / energy
    
    return val, grad_f

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin
    
    # Initial guess: Gaussian-like bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + w/2
    f0 = np.exp(-20.0 * x**2)
    
    best_val = np.inf
    best_f = f0.copy()
    
    # Try a few beta values for soft-max
    betas = [5.0, 10.0, 20.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        def obj_func(f):
            val, _ = _objective_and_grad(f, beta)
            return val
        
        def grad_func(f):
            _, grad = _objective_and_grad(f, beta)
            return grad
        
        try:
            res = minimize(
                obj_func, 
                f0, 
                jac=grad_func, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N, 
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_f = res.x
        except Exception:
            pass
            
    # Final check and return
    f = np.clip(best_f, 0.0, None)
    return [float(v) for v in f]
