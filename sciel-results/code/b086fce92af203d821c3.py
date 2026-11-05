"""Program for the third autocorrelation inequality task.

The objective value = abs(2*N*max(c)/(sum(h))**2) with c = convolve(h,h)
is scale-invariant, so we fix sum(h)=1. The global lower bound is 2,
attained by the uniform vector. We formulate the problem as a convex
QP minimizing the squared L2 norm of the autoconvolution c, which also
has the uniform vector as its unique minimizer.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _objective(h):
    # Compute autoconvolution c
    c = np.convolve(h, h)
    # Objective is sum of squares of c
    return float(np.dot(c, c))


def _jac(h):
    # Gradient of sum(c^2) is 4 * convolve(c, h)
    c = np.convolve(h, h)
    grad = 4.0 * np.convolve(c, h)
    # np.convolve returns full length 2N-1. 
    # The gradient with respect to h (length N) is the central part?
    # Let's check the mapping.
    # c_k for k in [-(N-1), N-1].
    # (c*h)_m for m in [0, N-1].
    # The full convolution of c (len 2N-1) and h (len N) has length 3N-2.
    # We need the part corresponding to indices 0..N-1 of the linear convolution?
    # Actually, the derivation $\frac{\partial J}{\partial h_m} = 4 \sum_k c_k (h_{m+k} + h_{m-k})$
    # assumes infinite support or specific boundary conditions.
    # For finite vectors, it's safer to use the relation:
    # J = || h * h ||_2^2.
    # In frequency domain: J = (1/N) sum |H(f)|^4.
    # Gradient in time domain is 4 * ifft(|H(f)|^2 * H(f))? No.
    # Let's stick to the time domain sum.
    # grad[m] = 4 * sum_k c[k] * (h[m+k] + h[m-k])
    # This is 4 * (c * h)[m] if c is defined on all integers and h is zero outside [0, N-1].
    # np.convolve(c, h) gives full convolution.
    # The index m in h corresponds to index m + (N-1) in the full convolution of c and h?
    # Let's just compute it directly to be safe, or use the FFT.
    # FFT approach:
    # H = fft(h)
    # C = H * H
    # J = sum |C|^2 / N (Parseval)
    # dJ/dh = 4 * ifft( C * conj(H) )? 
    # Let's check: J = (1/N) sum_f |H(f)|^4.
    # dJ/dh_m = (1/N) sum_f 4 |H(f)|^2 H(f) e^{i 2 pi f m / N} ?
    # d |H|^4 / d h_m = 4 |H|^2 H * e^{i...}
    # So grad = 4 * ifft( |H|^2 * H )? No, |H|^2 * H is not real.
    # Actually, H is complex. |H|^2 = H * conj(H).
    # So term is H * conj(H) * H = H^2 conj(H).
    # grad = 4 * ifft( H^2 * conj(H) )?
    # Let's verify with real h. H(-f) = conj(H(f)).
    # H^2 conj(H) at -f is H(-f)^2 conj(H(-f)) = conj(H(f))^2 H(f) = conj( H(f)^2 conj(H(f)) ).
    # So the spectrum is conjugate symmetric, result is real.
    
    H = np.fft.fft(h)
    # J = np.sum(np.abs(H)**4) / N
    # grad = 4 * np.real(np.fft.ifft(np.abs(H)**2 * H))
    # Wait, is it 4 or something else?
    # J = (1/N) sum |H|^4.
    # dJ = (1/N) sum 4 |H|^2 H dH.
    # dH(f) = sum h_m e^{-i...} dm.
    # This path is tricky with normalization factors.
    
    # Let's use the time domain formula: grad[m] = 4 * sum_k c[k] * h[m+k] + 4 * sum_k c[k] * h[m-k]
    # This is 4 * (c * h)[m] where c is extended by zeros.
    # np.convolve(c, h) computes full convolution.
    # c has length 2N-1. h has length N.
    # Full conv length 3N-2.
    # The coefficient of h[m] in the derivative corresponds to the sum over k.
    # It is known that grad = 4 * np.convolve(c, h)[N-1 : 2*N-1]?
    # Let's test with small N.
    
    # Alternative: Use finite differences for Jacobian if analytic is risky, but SLSQP can do it.
    # However, providing analytic jac is faster.
    # Let's use the FFT formula which is robust.
    # J = sum(c^2) = (1/N) sum(|H|^4).
    # grad = 4 * ifft( |H|^2 * H )? 
    # Let's check N=1. h=[1]. H=[1]. J=1. grad=4.
    # Formula: 4 * ifft(1*1) = 4. Correct.
    # N=2. h=[1, 0]. H=[1, 1]. |H|^2 = [1, 1]. |H|^2 * H = [1, 1]. ifft = [1, 1]. 4*ifft = [4, 4].
    # Manual: c = [1, 1, 0]. J = 1+1=2.
    # dJ/dh0: h0 changes. c0=h0^2+h1^2, c1=2h0h1, c2=h1^2.
    # J = (h0^2+h1^2)^2 + (2h0h1)^2 + h1^4.
    # At [1,0]: J=1. 
    # dJ/dh0 = 2(h0^2+h1^2)*2h0 + 4h0h1^2 = 2(1)*2 + 0 = 4.
    # dJ/dh1 = 2(h0^2+h1^2)*2h1 + 4h0^2h1 + 4h1^3 = 0 + 4 + 0 = 4.
    # So grad [4, 4] is correct.
    
    H = np.fft.fft(h)
    grad = 4.0 * np.real(np.fft.ifft(np.abs(H)**2 * H))
    return grad


def construct():
    # Fix the normalisation sum(h) = 1.
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_objective, x0, method="SLSQP", jac=_jac,
                   constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-14})
    h = res.x
    # Guard against numerical drift: renormalise to exact sum 1.
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
