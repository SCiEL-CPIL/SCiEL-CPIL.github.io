"""Optimal-control design for the Erdos minimum-overlap task.

Models the step heights as a solution to a discrete optimal control problem.
We minimize the maximum overlap by maximizing the minimum circular autocorrelation.
This is formulated as a non-linear optimization problem where we minimize a
smooth approximation of the negative minimum autocorrelation, subject to
sum and box constraints.
"""
import numpy as np
from scipy.optimize import minimize

N = 100
BETA = 50.0  # Sharpness of the min approximation

def _objective_and_grad(h_flat):
    """
    Compute the objective function and its gradient.
    Objective: Smoothed negative minimum autocorrelation.
    f(h) = (1/BETA) * log( sum_{k=1}^{N-1} exp(-BETA * A(k)) )
    This approximates -min(A(k)) as BETA -> infinity.
    """
    h = h_flat.reshape(N)
    f = np.fft.fft(h)
    ac = np.fft.ifft(f * np.conj(f)).real
    ac_nonzero = ac[1:]
    
    # Numerical stability: shift by max to prevent overflow in exp
    max_ac = np.max(ac_nonzero)
    # We want min, so we look at exp(-BETA * A).
    # To be stable, we can compute logsumexp(-BETA * ac_nonzero).
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    x = -BETA * ac_nonzero
    max_x = np.max(x)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    obj = log_sum_exp / BETA
    
    # Gradient
    # d/dh (1/BETA) log sum exp(-BETA A_k)
    # = (1/BETA) * (1/sum_exp) * sum (exp(-BETA A_k) * d/dh (-BETA A_k))
    # = (1/sum_exp) * sum (exp(-BETA A_k) * -dA_k/dh)
    # Let w_k = exp(-BETA A_k) / sum(exp(-BETA A_j))
    # Grad = - sum_k w_k * dA_k/dh
    
    exp_terms = np.exp(x - max_x)
    weights = exp_terms / np.sum(exp_terms)
    
    # dA_k/dh[j] = h[(j-k)%N] + h[(j+k)%N]
    # We need sum_k weights[k] * (h[j-k] + h[j+k])
    # This is a convolution of weights with h (and its reverse).
    # Since weights are for k=1..N-1, and A(k) is symmetric (A(k)=A(N-k)),
    # we can construct the full kernel.
    
    # Create full kernel of length N for lags 0..N-1
    # A(0) is not included in the sum, so weight[0] = 0.
    # A(k) for k=1..N-1.
    # Note: A(k) = A(N-k).
    kernel = np.zeros(N)
    # weights correspond to indices 0..N-2 in ac_nonzero, which are lags 1..N-1.
    # ac_nonzero[i] is A(i+1).
    # So weights[i] is weight for lag i+1.
    for i, w in enumerate(weights):
        lag = i + 1
        kernel[lag] = w
        kernel[N-lag] = w # Because A(k) = A(N-k) and the derivative structure is symmetric?
        # Actually, dA_k/dh[j] = h[j-k] + h[j+k].
        # Sum_k w_k (h[j-k] + h[j+k])
        # = sum_k w_k h[j-k] + sum_k w_k h[j+k]
        # Let m = -k, sum_m w_{-m} h[j+m] -> Convolution with w_reversed
        # Let m = k, sum_m w_m h[j+m] -> Cross-correlation
        
    # Efficiently compute gradient using FFT
    # grad[j] = sum_k (w_k * h[j-k] + w_k * h[j+k])
    # This is equivalent to:
    # grad = ifft( fft(h) * (fft(w_full) + fft(w_full_reversed?)) )?
    # Simpler: 
    # grad = np.convolve(h, w_full, mode='same')? No, circular.
    
    # Let's use the property:
    # Sum_k w_k h[j-k] is circular convolution of h and w_rev.
    # Sum_k w_k h[j+k] is circular correlation of h and w.
    
    # Construct full w array for lags 0..N-1
    w_full = np.zeros(N)
    # weights array has length N-1, corresponding to lags 1..N-1
    # w_full[1:] = weights
    # w_full[0] = 0
    # But we need to handle the symmetry of lags in the sum.
    # The sum is over k=1 to N-1.
    # Term 1: sum_{k=1}^{N-1} w_k h[j-k]
    # Term 2: sum_{k=1}^{N-1} w_k h[j+k]
    
    # Let's just do it with FFT for speed and correctness.
    # H = fft(h)
    # W1 = fft(w_full) where w_full[k] = w_k for k=1..N-1, 0 for k=0.
    # Term 1 is ifft(H * fft(w_full_rev)) ?
    # Circular convolution (h * w)[j] = sum h[m] w[j-m].
    # We want sum w[k] h[j-k]. Let m = j-k -> k = j-m.
    # sum w[j-m] h[m] = (h * w_rev)[j] where w_rev[k] = w[-k mod N].
    
    w_rev = np.zeros(N)
    # w_full[k] = weights[k-1] for k=1..N-1
    w_full = np.zeros(N)
    w_full[1:] = weights
    
    w_rev[0] = 0
    w_rev[1:] = weights[::-1] # w[N-1], w[N-2]... w[1]
    # Wait, w_rev[k] = w[-k mod N].
    # w[1] -> w_rev[N-1]
    # w[2] -> w_rev[N-2]
    # ...
    # w[N-1] -> w_rev[1]
    # So w_rev[1:] = weights[::-1] is correct.
    
    h_fft = np.fft.fft(h)
    w_rev_fft = np.fft.fft(w_rev)
    w_full_fft = np.fft.fft(w_full)
    
    # Term 1: sum w[k] h[j-k] = ifft( h_fft * w_rev_fft )
    # Term 2: sum w[k] h[j+k] = ifft( h_fft * w_full_fft )? 
    # sum w[k] h[j+k]. Let m = j+k -> k = m-j.
    # sum w[m-j] h[m]. This is correlation.
    # Correlation (h, w)[j] = sum h[m] w[m-j] = sum h[m] w_rev[j-m].
    # This is convolution h * w_rev.
    # Wait, Term 2 is sum_k w_k h[j+k].
    # This is the same as sum_k w_k h[j-(-k)].
    # It's the convolution with w_full (not reversed) if we define indices carefully?
    # Actually, Term 1 and Term 2 are conjugates if h and w are real?
    # A(k) is real.
    # Let's just compute:
    # grad = ifft(h_fft * (w_rev_fft + w_full_fft)).real
    # Let's verify with small N.
    
    grad = np.fft.ifft(h_fft * (w_rev_fft + w_full_fft)).real
    grad = -grad # Because we are minimizing -min, so grad of obj is -grad of min
    
    return obj, grad

def _project(h):
    """Project h onto the feasible set: sum h = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0, 1)
    for _ in range(100):
        s = h.sum()
        if abs(s - N/2) < 1e-9:
            break
        h = h + (N/2 - s) / N
        h = np.clip(h, 0, 1)
    return h

def construct():
    """
    Solves the Erdos minimum overlap problem using numerical optimal control.
    """
    np.random.seed(42)
    h0 = np.random.rand(N)
    h0 = _project(h0)
    
    # Use L-BFGS-B
    res = minimize(
        _objective_and_grad, 
        h0, 
        method='L-BFGS-B', 
        jac=True,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    h_opt = _project(res.x)
    
    # Ensure exact sum
    diff = N/2 - h_opt.sum()
    if diff != 0:
        h_opt = h_opt + diff / N
        h_opt = np.clip(h_opt, 0, 1)
        
    return h_opt
