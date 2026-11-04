"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

def _autoconv(x):
    """Full autoconvolution via FFT (circular, zero-padded to 2N-1)."""
    n = len(x)
    size = 2 * n - 1
    # next power of 2 for speed
    size = 1 << (size - 1).bit_length()
    X = fft(x, size)
    ac = ifft(X * X).real[:2 * n - 1]
    return ac

def _score(x):
    """C3' = max |autoconv| / (integral)^2. Lower is better."""
    ac = _autoconv(x)
    integral = x.sum()
    if integral == 0:
        return float('inf')
    return np.max(np.abs(ac)) / (integral * integral)

def _spectral_optimize(x, deadline, max_iter=2000, lr=0.05):
    """
    Optimize the sequence by treating it as a continuous variable in the frequency domain.
    We parameterize x via its Fourier transform X.
    We want to minimize max|ac| / (sum x)^2.
    Note: ac is the inverse FFT of |X|^2.
    sum x = X[0] (DC term).
    
    We use gradient descent on the phases of X (and potentially magnitudes, but we fix magnitudes 
    to 1 initially or allow them to vary slightly? No, for binary ±1, magnitudes are fixed in 
    a certain sense if we project, but in continuous relaxation, we can optimize the complex 
    values directly).
    
    Actually, a better approach for "Spectral Flatness" style:
    Let X be the FFT of x.
    ac = ifft(|X|^2).
    We want to minimize the peaks of ac.
    
    Let's use a smooth approximation of the max:
    Objective J = log( sum_k exp(ac[k] / T) ) / (X[0]^2)
    or simply sum(ac^2) / X[0]^2 (L2 norm minimization) which is easier to differentiate.
    Minimizing L2 norm of autocorrelation is a standard proxy for minimizing peak.
    """
    n = len(x)
    x_cont = x.astype(float)
    
    # Precompute FFT size
    size = 2 * n - 1
    size = 1 << (size - 1).bit_length()
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        X = fft(x_cont, size)
        ac = ifft(X * X).real[:2 * n - 1]
        integral = x_cont.sum()
        
        if abs(integral) < 1e-6:
            break
            
        # Objective: minimize sum(ac^2) / integral^2
        # Let f(x) = sum(ac^2) / integral^2
        # ac = IFFT(|X|^2)
        # This is getting complicated to differentiate directly through FFT.
        
        # Alternative: Gradient Descent on x directly using the gradient of the objective.
        # Objective J(x) = max|ac| / (sum x)^2 is non-smooth.
        # Use smooth proxy: J_smooth(x) = (sum_k ac[k]^2) / (sum x)^2
        # Note: sum_k ac[k]^2 = sum_m |X[m]|^4 (Parseval for autocorrelation? No, Parseval for ac is sum ac^2 = sum |X|^4 / N_fft?)
        # Actually, energy of ac is related to L4 norm of X.
        
        # Let's just do a simple gradient step on x.
        # We want to reduce large values of ac.
        # The gradient of sum(ac^2) w.r.t x[i] can be derived, but it's heavy.
        
        # Instead, let's use a "Peak Reduction" heuristic in the time domain guided by frequency.
        # If ac[k] is high, we want to adjust x to reduce it.
        
        # Let's try a different structural change: 
        # Differential Evolution on a reduced parameterization?
        # No, the prompt suggests "Reformulate ... as a complex optimization problem in the frequency domain ... optimizing the phase ... inverse FFT ... penalty term".
        
        # Let's implement Gradient Descent on the Phases of the FFT.
        # Constraint: x must be real. So X must be Hermitian symmetric.
        # We optimize the phases of the first N/2+1 bins.
        
        pass # Placeholder, see below for actual implementation

def _phase_descent(x, deadline, max_iter=500):
    """
    Optimize the phases of the Fourier transform to minimize the peak of the autocorrelation.
    x is real. X = fft(x).
    We define a parameter vector theta of length N (or N/2+1) representing phases.
    We fix the magnitudes of X to be constant (e.g., 1) or derived from x?
    If we fix magnitudes, we are not searching the space of all ±1 sequences, but a subset.
    However, for a ±1 sequence, the spectrum has specific properties.
    
    Better: Use the continuous relaxation x in [-1, 1] and perform gradient descent on x
    using a smooth objective, then project.
    """
    x = x.copy()
    n = len(x)
    size = 2 * n - 1
    size = 1 << (size - 1).bit_length()
    
    # We will minimize J(x) = sum(ac^2) / (sum x)^2
    # Gradient of sum(ac^2) w.r.t x[i]:
    # ac[k] = sum_j x[j] x[k-j]
    # sum_k ac[k]^2 = sum_k (sum_j x[j] x[k-j])^2
    # This is equivalent to the L4 norm of the spectrum / N.
    # d/dx_i sum_k ac[k]^2 = 4 * (x * ac)[i] ?
    # Let's verify: 
    # E = sum_k ac[k]^2.
    # ac[k] = (x * x)[k].
    # d(ac[k])/dx[i] = x[k-i] + x[i-k] (with bounds).
    # dE/dx[i] = sum_k 2 ac[k] (x[k-i] + x[i-k])
    #          = 2 sum_k ac[k] x[k-i] + 2 sum_k ac[k] x[i-k]
    # Let m = k-i => k=m+i. sum_m ac[m+i] x[m]. This is cross-correlation of ac and x.
    # Since ac is symmetric (for real x), cross-correlation is convolution.
    # So dE/dx[i] = 4 (x * ac)[i] (circular convolution, assuming proper padding).
    
    # Let's use this gradient.
    for iter in range(max_iter):
        if time.time() > deadline:
            break
            
        ac = _autoconv(x)
        integral = x.sum()
        
        if abs(integral) < 1e-6:
            # Add small bias to avoid division by zero
            x += 1e-6 * np.ones(n)
            continue
            
        # Compute gradient of E = sum(ac^2)
        # grad_E = 4 * conv(x, ac)
        # We need to compute conv(x, ac) efficiently.
        # ac has length 2N-1. x has length N.
        # We want the linear convolution part that corresponds to the derivative.
        # The derivative of the linear autocorrelation energy is 4 * (x * ac) centered?
        # Actually, for linear convolution, the gradient of sum(ac^2) w.r.t x[i] is:
        # 2 * sum_k ac[k] * (d ac[k]/dx[i])
        # d ac[k]/dx[i] = x[k-i] (if 0<=k-i<N) + x[i-k] (if 0<=i-k<N)
        # This is effectively the cross-correlation of ac and x.
        
        # Let's compute grad using FFT.
        # grad[i] = 2 * (ac * x)[i] where * denotes cross-correlation?
        # Let's just compute it directly for N=400? O(N^2) = 160k ops. Very fast.
        
        grad_E = np.zeros(n)
        # ac[k] for k in [0, 2N-2]
        # grad[i] = 2 * sum_k ac[k] * (x[k-i] if valid + x[i-k] if valid)
        # This is 2 * (ac * x)_{N-1} ? No.
        # Let's use the property: sum_k ac[k]^2 = ||x||_4^4 / N_fft?
        # Let's stick to the direct sum for gradient if N is small, or use FFT.
        
        # FFT approach for gradient of sum(ac^2):
        # E = sum_k |FFT(x * x)_k|^2 ? No.
        # E = sum_k ac[k]^2.
        # Let's use the convolution theorem.
        # ac = ifft(fft(x)^2).
        # E = sum_k (ifft(fft(x)^2)[k])^2.
        # This is the inner product of ac with itself.
        # By Parseval, sum_k ac[k]^2 = (1/M) sum_m |FFT(ac)_m|^2.
        # FFT(ac) = FFT(ifft(fft(x)^2)) = fft(x)^2 (if M is large enough).
        # So E = (1/M) sum_m |fft(x)_m|^4.
        
        # We want to minimize E / integral^2.
        # Let g(x) = E, h(x) = integral^2.
        # grad (g/h) = (h grad g - g grad h) / h^2.
        
        X = fft(x, size)
        # E = (1/size) * sum(|X|^4)
        # dE/dx[i] ?
        # E = (1/size) sum_m (X[m] X[m]^*)^2 ? No, |X|^4 = X^2 (X^*)^2.
        # This is getting messy. Let's use the time-domain gradient formula:
        # dE/dx[i] = 4 * (x * ac)[i] where * is circular convolution of length N?
        # If we use zero-padding to 2N-1, the autocorrelation is linear.
        # The energy of linear autocorrelation is sum_k ac[k]^2.
        # The gradient w.r.t x[i] is 4 * (x * ac)_i where the convolution is such that it aligns correctly.
        
        # Let's compute the gradient numerically or via direct sum for safety and simplicity given N=400.
        # Direct sum:
        # grad[i] = 2 * sum_k ac[k] * (x[k-i] + x[i-k])
        # We can vectorize this.
        # For each i, we need sum_k ac[k] x[k-i] and sum_k ac[k] x[i-k].
        # These are cross-correlations.
        
        # corr1[i] = sum_k ac[k] x[k-i]
        # This is the convolution of ac and x, reversed?
        # Let's just do it with np.convolve or FFT.
        
        # Using FFT for the gradient term:
        # Term 1: sum_k ac[k] x[k-i]. Let j = k-i => k=j+i. sum_j ac[j+i] x[j].
        # This is the cross-correlation of ac and x at lag i.
        # We can compute the full cross-correlation via FFT.
        
        # ac is length 2N-1. x is length N.
        # We want the part of the correlation that falls within the support of x.
        
        # Let's compute C = cross_correlation(ac, x).
        # C[l] = sum_k ac[k+l] x[k].
        # We need C[i] and C[-i].
        
        # Efficiently:
        # Pad x to length 2N-1.
        # ac is already length 2N-1.
        # FFT both.
        # Correlation is ifft(FFT(ac) * conj(FFT(x))).
        
        x_padded = np.zeros(2 * n - 1)
        x_padded[:n] = x
        ac_padded = ac # already 2N-1
        
        X_pad = fft(x_padded)
        AC_pad = fft(ac_padded)
        
        # Cross correlation
        corr = ifft(AC_pad * np.conj(X_pad)).real
        
        # corr[k] = sum_m ac_padded[m+k] x_padded[m] (circular)
        # Since we padded to 2N-1, the linear correlation is in the first N and last N-1 elements?
        # Actually, for linear correlation of lengths N and 2N-1, the result length is 3N-2.
        # Our FFT size is 2N-1, which is too small for full linear correlation if we consider all lags.
        # But we only need lags i in [0, N-1].
        # The term sum_k ac[k] x[k-i] corresponds to lag i.
        # The term sum_k ac[k] x[i-k] corresponds to lag -i.
        
        # Let's verify the gradient formula again.
        # E = sum_k ac[k]^2.
        # dE/dx[i] = 2 sum_k ac[k] (x[k-i] + x[i-k]).
        
        # Let's compute the sum directly for N=400. It's fast enough.
        # grad = np.zeros(n)
        # for i in range(n):
        #     # sum_k ac[k] x[k-i]
        #     # valid k: 0 <= k < 2N-1, 0 <= k-i < N => i <= k < i+N
        #     # sum_k ac[k] x[i-k]
        #     # valid k: 0 <= k < 2N-1, 0 <= i-k < N => i-N < k <= i
        
        # Vectorized direct sum:
        # We can use np.convolve.
        # conv(ac, x) gives sum_k ac[k] x[n-k].
        # We need sum_k ac[k] x[k-i]. Let j=k-i => k=j+i. sum_j ac[j+i] x[j].
        # This is cross-correlation.
        
        # Let's use the FFT correlation result 'corr' from above.
        # corr is circular correlation of length 2N-1.
        # For lags 0 to N-1, the linear correlation is contained in corr[0:N] and corr[2N-1:N-1:-1]?
        # Let's just trust the direct sum for robustness in this small N.
        
        grad = np.zeros(n)
        for i in range(n):
            # Part 1: sum_k ac[k] x[k-i]
            # k ranges from i to i+N-1 (since x index k-i must be in [0, N-1])
            # Also k must be in [0, 2N-2]
            k_start = max(0, i)
            k_end = min(2 * n - 1, i + n)
            if k_end > k_start:
                # x indices: k - i, so from k_start-i to k_end-1-i
                x_idx_start = k_start - i
                x_idx_end = k_end - i
                grad[i] += np.dot(ac[k_start:k_end], x[x_idx_start:x_idx_end])
            
            # Part 2: sum_k ac[k] x[i-k]
            # x index i-k must be in [0, N-1] => i-N+1 <= k <= i
            k_start = max(0, i - n + 1)
            k_end = min(2 * n - 1, i)
            if k_end >= k_start:
                # x indices: i - k, so from i-k_end to i-k_start
                x_idx_start = i - k_end
                x_idx_end = i - k_start + 1
                grad[i] += np.dot(ac[k_start:k_end+1], x[x_idx_start:x_idx_end])
        
        grad *= 2
        
        # Gradient of integral^2
        # h = (sum x)^2
        # dh/dx[i] = 2 * integral
        integral_sq = integral * integral
        grad_h = 2 * integral * np.ones(n)
        
        # Gradient of J = E / h
        # dJ = (h grad_E - E grad_h) / h^2
        E = np.sum(ac**2)
        
        if integral_sq < 1e-12:
            break
            
        grad_J = (integral_sq * grad - E * grad_h) / (integral_sq * integral_sq)
        
        # Gradient Descent Step
        # Adaptive learning rate
        norm_grad = np.linalg.norm(grad_J)
        if norm_grad < 1e-12:
            break
            
        lr = 0.1 / norm_grad
        x -= lr * grad_J
        
        # Project to [-1, 1]
        x = np.clip(x, -1.0, 1.0)
        
    # Final projection to ±1
    x = np.sign(x)
    # If any zeros, set to 1
    x[x == 0] = 1.0
    
    return x

def _local_search(x, deadline, max_iter=5000):
    """Local search: flip signs to reduce score."""
    n = len(x)
    best_x = x.copy()
    best_score = _score(best_x)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try a random subset of flips to save time, or all if fast enough
        # N=400, checking all is 400 FFTs. ~400 * 0.1ms = 40ms. Doable per iteration?
        # Let's try all flips in one pass.
        
        best_delta = 0
        best_pos = -1
        
        # To speed up, we can precompute ac and update it incrementally, 
        # but for simplicity and robustness, we compute full score for candidates.
        # However, 400 candidates * 400 iterations = 160k FFTs. Too slow.
        # Let's sample 50 random positions per iteration.
        
        indices = np.random.choice(n, size=min(50, n), replace=False)
        
        for i in indices:
            new_x = best_x.copy()
            new_x[i] = -new_x[i]
            new_score = _score(new_x)
            
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_x = new_x
                improved = True
                break # Restart search from new state
        
        if not improved:
            break
            
    return best_x, best_score

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_x = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Initial candidates
    candidates = []
    
    # 1. Random
    candidates.append(np.random.choice([-1, 1], size=N).astype(float))
    
    # 2. Alternating
    x_alt = np.ones(N)
    x_alt[1::2] = -1
    candidates.append(x_alt)
    
    # 3. Block pattern
    x_block = np.ones(N)
    x_block[200:] = -1
    candidates.append(x_block)
    
    # 4. M-sequence like (pseudo-random but structured)
    # Use a simple LFSR or just a different random seed
    np.random.seed(123)
    candidates.append(np.random.choice([-1, 1], size=N).astype(float))
    
    for x0 in candidates:
        if time.time() > deadline - 5:
            break
            
        # Phase 1: Spectral Phase Optimization (Continuous Relaxation)
        x_opt = _phase_descent(x0, deadline - 5, max_iter=100)
        
        # Phase 2: Local Search
        x_final, score = _local_search(x_opt, deadline - 5, max_iter=1000)
        
        if score < best_score:
            best_score = score
            best_x = x_final
        
        # If time remains, do more restarts
        if time.time() < deadline - 10:
             # Generate new random start
             x0 = np.random.choice([-1, 1], size=N).astype(float)
             x_opt = _phase_descent(x0, deadline - 10, max_iter=50)
             x_final, score = _local_search(x_opt, deadline - 10, max_iter=500)
             if score < best_score:
                 best_score = score
                 best_x = x_final

    if best_x is None:
        # Fallback
        best_x = np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
