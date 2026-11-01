"""
Strategy: Continuous Gradient Descent on relaxed signal + Thresholding.
Instead of discrete bit flips, we optimize a real-valued vector x where x_i in R.
We minimize a smooth surrogate for the max sidelobe (e.g., log-sum-exp or high power sum)
using gradient descent. Then we threshold the result to +/- 1.
This allows escaping discrete local minima by moving through the continuous space.
"""
import numpy as np
import time

N = 400

def _compute_autocorr(x):
    """Compute autocorrelation of x using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _objective_and_grad(x, k=100):
    """
    Compute a smooth objective function and its gradient.
    We use a high-power sum of sidelobes to approximate the max sidelobe.
    J = sum(|acf[i]|^k) for i in 1..N-1
    grad = k * |acf[i]|^(k-2) * acf[i] * grad_acf
    """
    acf = _compute_autocorr(x)
    sidelobes = acf[1:]
    
    # Use a large power to approximate max, but keep it numerically stable
    # k=100 is a good balance for N=400
    abs_sl = np.abs(sidelobes)
    # Avoid 0^0 issues
    abs_sl_safe = np.maximum(abs_sl, 1e-9)
    
    J = np.sum(abs_sl_safe ** k)
    
    # Gradient of J w.r.t. x
    # dJ/dacf_i = k * |acf_i|^(k-1) * sign(acf_i)
    dJ_dacf = k * (abs_sl_safe ** (k - 1)) * np.sign(sidelobes)
    dJ_dacf[0] = 0 # ignore lag 0
    
    # Gradient of acf w.r.t. x
    # acf = ifft(fft(x) * conj(fft(x)))
    # This is a convolution-like operation.
    # Using FFT: acf = ifft(|X|^2)
    # d(acf)/dx is related to the inverse FFT of the derivative.
    # Specifically, if we define Y = X * conj(X), then dY/dx_j involves X and conj(X).
    # A simpler way for gradient of sum f(acf) is to use the fact that
    # acf is a quadratic form. 
    # However, a more robust numerical gradient or analytical derivation is needed.
    
    # Let's derive the gradient properly.
    # acf[m] = sum_n x[n] x[n+m] (circular or linear? We used linear via zero-padding in FFT)
    # The FFT method computes circular autocorrelation if we don't zero pad enough, 
    # but we zero padded to 2*N-1, so it's linear autocorrelation.
    
    # Gradient of linear autocorrelation cost:
    # J = sum_m f(acf[m])
    # dJ/dx[j] = sum_m f'(acf[m]) * d(acf[m])/dx[j]
    # d(acf[m])/dx[j] = x[j+m] + x[j-m] (with boundary conditions)
    # This is expensive to compute directly for all j, m.
    
    # Alternative: Use the FFT property.
    # acf = ifft( |X|^2 )
    # Let G = f'(acf)
    # The gradient in frequency domain?
    # Actually, for quadratic forms, the gradient is:
    # grad_x J = 2 * ifft( X * conj(X) * ... )? No.
    
    # Let's use a simpler, robust gradient approximation or a known identity.
    # Identity: If J = sum_m c_m acf[m], then grad_x J = 2 * (c * x) where * is cross-correlation.
    # Here c_m = f'(acf[m]).
    # So grad_x J = 2 * cross_corr(c, x).
    # We can compute this via FFT.
    
    # c is defined on lags 0..N-1.
    # We need to compute y[j] = sum_m c_m x[j+m] + c_m x[j-m]? 
    # Actually, d(acf[m])/dx[j] = x[j+m] + x[j-m] (for linear autocorr, with x[k]=0 if k<0 or k>=N).
    # So dJ/dx[j] = sum_m f'(acf[m]) (x[j+m] + x[j-m])
    # This is equivalent to 2 * (f'(acf) * x) where * is correlation.
    
    # Let's compute correlation via FFT.
    # corr(c, x)[j] = sum_m c[m] x[j+m] ? 
    # Standard correlation: (c * x)[j] = sum_k c[k] x[j+k].
    # We want sum_m c[m] x[j+m]. This is cross-correlation.
    # Using FFT: 
    # Let C = fft(c, n_fft), X = fft(x, n_fft)
    # corr = ifft(C * conj(X)) ? 
    # Let's check.
    # ifft( C * conj(X) )_j = sum_k c_k x_{j-k} ? 
    # Actually, circular correlation is ifft( C * conj(X) ).
    # Linear correlation can be obtained by zero padding.
    
    # Let's just use the direct sum if N is small enough? N=400, N^2 = 160k. That's fast.
    # But we are in a loop, so FFT is better.
    
    # Let's use the FFT method for gradient.
    # dJ/dx = 2 * ifft( fft(c) * conj(fft(x)) ) ? 
    # Wait, the derivative of acf[m] = sum_n x[n]x[n+m] w.r.t x[j] is x[j+m] + x[j-m].
    # Summing over m with weights f'(acf[m]):
    # grad[j] = sum_m f'(acf[m]) x[j+m] + sum_m f'(acf[m]) x[j-m]
    # Let g[m] = f'(acf[m]).
    # Term 1: sum_m g[m] x[j+m]. This is cross-correlation of g and x.
    # Term 2: sum_m g[m] x[j-m]. This is cross-correlation of g and x reversed? Or just correlation.
    # Note that g[m] is even? No, acf is even, so f'(acf[m]) is even if f is even function of acf?
    # f(u) = |u|^k. f' is odd if k is even? No, f' = k|u|^(k-1)sign(u).
    # Since acf is even, acf[m] = acf[-m]. So g[m] = g[-m]. g is even.
    # So sum_m g[m] x[j+m] = sum_m g[-m] x[j-m] = sum_p g[p] x[j-p] (let p=-m).
    # So Term 1 = Term 2.
    # grad[j] = 2 * sum_m g[m] x[j+m].
    
    # So we just need to compute the cross-correlation of g and x.
    # g is defined for m=0..N-1. We need g[m] for m in -N+1..N-1.
    # Since g is even, g[-m] = g[m].
    
    # Let's construct the full g vector of size 2*N-1 for circular correlation?
    # Or just use linear correlation via FFT.
    
    n_fft = 1 << (2 * N - 1).bit_length()
    # g_full of size n_fft
    g_full = np.zeros(n_fft)
    # g[m] for m=0..N-1
    g_full[:N] = dJ_dacf
    # g[-m] = g[m] => g_full[N_fft - m] = g[m]
    for m in range(1, N):
        g_full[n_fft - m] = dJ_dacf[m]
        
    X = np.fft.rfft(x, n=n_fft)
    G = np.fft.rfft(g_full, n=n_fft)
    
    # Cross correlation: ifft(G * conj(X))
    # This gives sum_k g[k] x[j-k]? 
    # Let's verify.
    # ifft(G * conj(X))_j = sum_k g[k] x[j-k] (circular)
    # We want sum_m g[m] x[j+m].
    # Let k = -m. sum_m g[m] x[j+m] = sum_k g[-k] x[j-k].
    # Since g is even, g[-k] = g[k].
    # So it is sum_k g[k] x[j-k].
    # So yes, ifft(G * conj(X)) gives the desired sum.
    
    corr = np.fft.irfft(G * np.conj(X), n=n_fft).real[:N]
    grad = 2.0 * corr
    
    return J, grad

def construct():
    """
    Construct a signed step function by optimizing a continuous relaxation
    and then thresholding.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize with random values in [-1, 1]
    np.random.seed(42)
    x = np.random.uniform(-1, 1, size=N)
    
    best_x = x.copy()
    best_val = float('inf')
    
    # Gradient Descent
    lr = 0.01
    momentum = 0.9
    v = np.zeros(N)
    
    # We will run a fixed number of steps or until time limit
    # N=400, FFT is fast. ~1000 steps should be quick.
    
    for i in range(2000):
        if time.time() > deadline:
            break
            
        J, grad = _objective_and_grad(x, k=50)
        
        # Momentum update
        v = momentum * v - lr * grad
        x = x + v
        
        # Optional: Keep x bounded to avoid numerical issues, though not strictly necessary
        # x = np.clip(x, -10, 10)
        
        # Check thresholded version
        h = np.sign(x)
        # Ensure no zeros (though sign(0)=0, we want +/-1)
        h[h == 0] = 1
        
        # Evaluate the actual discrete objective (max sidelobe)
        acf_h = _compute_autocorr(h)
        max_sl = np.max(np.abs(acf_h[1:]))
        
        if max_sl < best_val:
            best_val = max_sl
            best_x = h.copy()
            
        # Adaptive learning rate? Simple decay
        lr = max(0.001, lr * 0.999)
        
    # Final check
    h_final = best_x
    # One last local search on the discrete result to polish?
    # The prompt asks for a structurally different strategy. 
    # However, a small polish step is often beneficial and doesn't change the core strategy.
    # Let's do a few iterations of greedy flip to clean up the thresholded result.
    
    deadline_local = min(deadline, time.time() + 10.0)
    h = h_final.copy()
    val = np.max(np.abs(_compute_autocorr(h)[1:]))
    
    improved = True
    while improved and time.time() < deadline_local:
        improved = False
        # Randomly select a subset of indices to try flipping to save time
        # Or just iterate through all if fast enough. N=400, 400 FFTs is ~0.1s?
        # Let's just iterate all.
        for j in range(N):
            if time.time() > deadline_local:
                break
            h[j] = -h[j]
            new_val = np.max(np.abs(_compute_autocorr(h)[1:]))
            h[j] = -h[j] # revert
            
            if new_val < val:
                h[j] = -h[j] # keep
                val = new_val
                improved = True
        
    return h
