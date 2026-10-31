"""
Strategy: Deterministic Kronecker Sequence + Hybrid Optimization
This approach uses the low-discrepancy properties of Kronecker sequences (fractional parts of i*alpha)
to construct signed step functions. It searches over the irrational parameter alpha and a binary sign mask.
The continuous height is optimized analytically for a given mask to balance the DC component (sum) 
against the AC autoconvolution peaks.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # 1. Generate a diverse set of Kronecker sequences (low-discrepancy)
    # x_i = sign( frac(i * alpha) - 0.5 )
    # We will test various alphas.
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Candidate alphas: 
    # - Golden ratio conjugate
    # - Square roots of small primes
    # - Random irrationals
    
    alphas = [
        (1 + np.sqrt(5)) / 2 - 1,  # Golden ratio
        np.sqrt(2) - 1,
        np.sqrt(3) - 1,
        np.sqrt(5) - 2,
        np.sqrt(6) - 2,
        np.sqrt(7) - 2,
        0.38196601125, # 1/phi^2
        0.27639320225, # sqrt(2)-1 approx
        0.31783724519, # sqrt(3)-1 approx
        0.41421356237, # sqrt(2)-1
        0.73205080756, # sqrt(3)-1
        0.61803398875, # 1/phi
    ]
    
    # Add some random irrationals
    rng = np.random.default_rng(42)
    for _ in range(20):
        alphas.append(rng.uniform(0.1, 0.9))
        
    def optimize_sequence(alpha):
        """
        For a given alpha, construct base signs s_i = sign(frac(i*alpha) - 0.5).
        Then, we model the sequence as x_i = c + d * s_i.
        We want to minimize max(|A|) / S^2.
        This is equivalent to finding the best c, d for this specific pattern.
        """
        # Generate signs
        # i from 0 to N-1
        i = np.arange(N)
        frac = (i * alpha) % 1.0
        # Map to [-1, 1] via sign
        # If frac < 0.5, -1; else 1
        s = np.where(frac < 0.5, -1.0, 1.0)
        
        # Compute autoconv of s
        A_s = _autoconv(s)
        
        # x = c + d*s
        # Sum(x) = c*N + d*Sum(s)
        # Autoconv(x) = c^2 * A_ones + 2*c*d * A_s_ones + d^2 * A_s
        # A_ones is the autoconv of ones: [1, 2, ..., N, ..., 1]
        # A_s_ones is the autoconv of s with ones: sum(s) at each lag? No.
        # Conv(s, ones)_k = sum_j s_j * 1_{k-j in [0, N-1]} = sum of s_j over valid j.
        
        # Let's compute these terms efficiently.
        # A_ones:
        A_ones = np.zeros(2*N - 1)
        for k in range(2*N - 1):
            # Overlap of [0, N-1] and [k, k+N-1]
            start = max(0, k)
            end = min(N, k + N)
            A_ones[k] = end - start
            
        # A_s_ones:
        # Conv(s, ones)_k = sum_{j=max(0, k-N+1)}^{min(N-1, k)} s_j
        # This is the sliding window sum of s.
        # Can be computed via cumulative sum.
        cum_s = np.cumsum(s)
        # Prepend 0
        cum_s = np.concatenate(([0], cum_s))
        
        A_s_ones = np.zeros(2*N - 1)
        for k in range(2*N - 1):
            # j ranges from max(0, k-N+1) to min(N-1, k)
            j_start = max(0, k - N + 1)
            j_end = min(N - 1, k)
            if j_start <= j_end:
                A_s_ones[k] = cum_s[j_end + 1] - cum_s[j_start]
            else:
                A_s_ones[k] = 0
                
        # Now, A_x(c, d) = c^2 * A_ones + 2*c*d * A_s_ones + d^2 * A_s
        # S(c, d) = c*N + d*Sum(s)
        # Score(c, d) = max_k |A_x[k]| / S(c, d)^2
        
        # We can scan a small grid of c, d or optimize.
        # Note: Scaling c, d by k scales A by k^2 and S by k, so Score is invariant to global scaling.
        # We can fix d=1 and vary c. Or fix S=1 and vary ratio.
        # Let's vary c in [-2, 2] and d=1.
        
        best_local_score = np.inf
        best_local_c = 0
        
        # Sum(s)
        S_s = np.sum(s)
        
        # If S_s is close to 0, we rely on c for the sum.
        # If S_s is large, d contributes significantly.
        
        # Grid search for c
        # c in [-2, 2] with step 0.1
        for c in np.arange(-2.0, 2.1, 0.1):
            d = 1.0
            S = c * N + d * S_s
            if abs(S) < 1e-9:
                continue
                
            A_x = c*c * A_ones + 2*c*d * A_s_ones + d*d * A_s
            m = np.max(np.abs(A_x))
            sc = m / (S * S)
            
            if sc < best_local_score:
                best_local_score = sc
                best_local_c = c
                
        # Refine around best_local_c
        if best_local_c != 0:
            for c in np.arange(best_local_c - 0.2, best_local_c + 0.2, 0.01):
                d = 1.0
                S = c * N + d * S_s
                if abs(S) < 1e-9:
                    continue
                A_x = c*c * A_ones + 2*c*d * A_s_ones + d*d * A_s
                m = np.max(np.abs(A_x))
                sc = m / (S * S)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local_c = c
        else:
            # If best was at 0, check around 0
            for c in np.arange(-0.2, 0.2, 0.01):
                d = 1.0
                S = c * N + d * S_s
                if abs(S) < 1e-9:
                    continue
                A_x = c*c * A_ones + 2*c*d * A_s_ones + d*d * A_s
                m = np.max(np.abs(A_x))
                sc = m / (S * S)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local_c = c

        if not np.isfinite(best_local_score):
            return None, np.inf

        # Construct the final sequence
        x = best_local_c + 1.0 * s
        return x, best_local_score

    # 2. Iterate through alphas
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
            
        x_candidate, sc_candidate = optimize_sequence(alpha)
        
        if x_candidate is not None and sc_candidate < best_score:
            best_score = sc_candidate
            best_x = x_candidate.copy()
            
    # 3. Local Search on the Best Sequence
    # Try flipping signs of individual elements in the best found sequence (if it was derived from a Kronecker seq)
    # Or just perturb the continuous values?
    # The task allows "signed step heights". 
    # If we found a good (c, d, s), we can try to locally optimize s by flipping signs.
    
    # Let's extract the structure of best_x if it came from a Kronecker seq
    # Actually, best_x is just an array. We can perform a local search on the array itself.
    # But the array is continuous. 
    # If we assume the optimal solution is close to c + d*s, we can refine s.
    
    # Let's try a simple local search on the best_x found:
    # 1. Identify "natural" signs: sign(best_x - median(best_x)) ?
    # 2. Or just try flipping the sign of the element with largest impact?
    
    # A simpler local search:
    # Try flipping the sign of each element in the underlying "sign" vector s that generated best_x.
    # We need to recover s. 
    # If best_x = c + d*s, then s = (best_x - c)/d.
    # We don't know c and d exactly from best_x alone without fitting.
    # Let's fit c and d to best_x.
    
    # Fit c + d*s where s_i in {-1, 1}
    # This is a binary fitting problem. 
    # Approximation: s_i = sign(best_x_i - mean(best_x))?
    # Let's assume the "sign" structure is sign(best_x - c_est).
    
    # Let's just do a generic local search on the continuous vector best_x.
    # Gradient of max(|A|) is hard. 
    # Let's use the L2 proxy gradient again, but starting from the Kronecker solution.
    
    def _loss_smooth(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return np.inf, np.inf, 0
        n = len(x)
        size = 2 * n - 1
        size2 = 1
        while size2 < size:
            size2 *= 2
        F = fft(x, size2)
        sum_f4 = np.sum(np.abs(F)**4)
        val = sum_f4 / (s * s)
        
        H = F * (np.conj(F))**2
        grad_f = 4.0 * size2 * np.real(ifft(H))[:n]
        grad_val = grad_f / (s * s) - 2.0 * val / s
        
        return val, grad_val, s

    x = best_x.copy()
    lr = 0.01
    for step in range(500):
        if time.time() > deadline - 2.0:
            break
            
        val, grad, s = _loss_smooth(x)
        if not np.isfinite(val):
            break
            
        x_new = x - lr * grad
        # Project to [-1, 1]
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Check real score
        real_sc = _score(x_new)
        if real_sc < best_score:
            best_score = real_sc
            best_x = x_new.copy()
            x = x_new
        else:
            # If not improved, try smaller step or break
            lr *= 0.9
            if lr < 1e-5:
                break
            x = x_new

    # Final check: try thresholding to +/- 1 if that helps
    if time.time() < deadline - 1.0:
        x_thresh = np.where(best_x > 0, 1.0, -1.0)
        sc_thresh = _score(x_thresh)
        if sc_thresh < best_score:
            best_score = sc_thresh
            best_x = x_thresh

    return best_x

# EVOLVE-BLOCK-END
