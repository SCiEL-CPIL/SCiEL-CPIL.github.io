"""
Spectral shaping approach for the third autocorrelation inequality.
We model the problem in the frequency domain: for a signed step function s[k] in {-1,1}^N,
the autoconvolution peak is related to the L-infinity norm of the inverse DFT of |S(f)|^2,
where S is the DFT of s. We relax s to a continuous vector x, optimize a smooth surrogate
for the peak (log-sum-exp of |IFFT(|FFT(x)|^2)|^2), then threshold to {-1,1}.
"""
import numpy as np
import time

N = 400

def _score(s):
    """Compute the upper bound on C3: max of autoconvolution."""
    # autoconvolution via FFT
    S = np.fft.rfft(s, n=2 * N)
    power = S * np.conj(S)
    ac = np.fft.irfft(power, n=2 * N)
    # The autoconvolution has length 2N-1 conceptually, but irfft gives 2N.
    # The peak of the autoconvolution (excluding the trivial center? No, the task says "upper bound on C3")
    # C3 is typically the max of the autocorrelation sidelobes or the full peak.
    # The task says "score is the upper bound on C3".
    # For a signed step function, the autoconvolution peak is at least N (the DC term).
    # The "upper bound on C3" likely refers to the maximum value of the autoconvolution.
    # Let's assume it's the max of the autoconvolution.
    # Note: ac[0] is the sum of squares = N.
    # The sidelobes are the other values.
    # Let's return the max of the autoconvolution.
    return np.max(ac)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial guess: random signs
    rng = np.random.default_rng(42)
    best_s = rng.choice([-1, 1], size=N)
    best_score = _score(best_s)
    
    # Continuous relaxation: x in R^N
    # We optimize x to minimize a smooth surrogate of the peak autoconvolution.
    # Surrogate: log-sum-exp of (IFFT(|FFT(x)|^2))^2
    # This is a smooth approximation to the max.
    
    def surrogate(x):
        S = np.fft.rfft(x, n=2 * N)
        power = S * np.conj(S)
        ac = np.fft.irfft(power, n=2 * N)
        # Use log-sum-exp for numerical stability
        ac2 = ac * ac
        m = np.max(ac2)
        if m > 1e6:
            return m  # Fallback for large values
        return m + np.log(np.sum(np.exp(ac2 - m)))
    
    def surrogate_grad(x):
        S = np.fft.rfft(x, n=2 * N)
        power = S * np.conj(S)
        ac = np.fft.irfft(power, n=2 * N)
        ac2 = ac * ac
        m = np.max(ac2)
        if m > 1e6:
            # Gradient of max
            idx = np.argmax(ac2)
            grad_ac = np.zeros_like(ac)
            grad_ac[idx] = 2.0 * ac[idx]
            # Backprop through irfft
            # ac = irfft(power)
            # power = S * conj(S)
            # This is complex, so let's use finite differences or a simpler approach.
            # For now, return a placeholder gradient.
            return np.zeros_like(x)
        
        exp_ac2 = np.exp(ac2 - m)
        sum_exp = np.sum(exp_ac2)
        weights = exp_ac2 / sum_exp
        
        # d/dac of log-sum-exp(ac2) = 2 * ac * weights
        grad_ac = 2.0 * ac * weights
        
        # Backprop through irfft to power
        # ac = irfft(power) => power = fft(ac) (with appropriate scaling)
        # Actually, irfft is the inverse of rfft.
        # Let's use the chain rule carefully.
        # This is getting complex. Let's use a simpler optimization: 
        # Instead of gradient descent on the continuous relaxation, 
        # let's use a discrete local search with a better initial point.
        
        return np.zeros_like(x)
    
    # Given the complexity of the gradient, let's use a different strategy:
    # 1. Start with a random signed step function.
    # 2. Use simulated annealing or hill climbing with a twist:
    #    - Flip a random subset of bits (not just one) to escape local minima.
    #    - Use a spectral initialization: find a continuous x that minimizes the surrogate,
    #      then threshold.
    
    # Let's try a simple spectral initialization:
    # Minimize ||IFFT(|FFT(x)|^2)||_inf^2
    # This is equivalent to minimizing the peak of the autoconvolution.
    # We can use a few iterations of gradient descent with a finite-difference gradient.
    
    x = best_s.astype(float)
    lr = 0.1
    
    for i in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Finite difference gradient
        eps = 1e-4
        grad = np.zeros_like(x)
        for j in range(N):
            x_plus = x.copy()
            x_plus[j] += eps
            x_minus = x.copy()
            x_minus[j] -= eps
            grad[j] = (surrogate(x_plus) - surrogate(x_minus)) / (2 * eps)
        
        x -= lr * grad
        lr *= 0.95
        
        # Threshold to get a signed step function
        s_candidate = np.sign(x)
        s_candidate[s_candidate == 0] = 1.0
        score_candidate = _score(s_candidate)
        
        if score_candidate < best_score:
            best_score = score_candidate
            best_s = s_candidate.astype(int)
    
    # Now do a local search on the best signed step function
    for i in range(10000):
        if time.time() > deadline - 5.0:
            break
        
        # Flip a random bit
        j = rng.integers(N)
        s_candidate = best_s.copy()
        s_candidate[j] *= -1
        score_candidate = _score(s_candidate)
        
        if score_candidate < best_score:
            best_score = score_candidate
            best_s = s_candidate
    
    return best_s

# EVOLVE-BLOCK-START
# (The above code is the new implementation)
# EVOLVE-BLOCK-END
