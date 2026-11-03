"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 115.0 # Safety margin
    
    # 1. Baseline: All ones (Score ~ 1/N = 0.0025)
    h_const = np.ones(N)
    best_h = h_const
    best_score = _score(h_const)
    
    # 2. Spectral Optimization Strategy
    # We want to minimize max|autoconv(h)| subject to sum(h)^2 = 1 (scale invariance).
    # autoconv(h) = ifft(fft(h)^2). 
    # Let H = fft(h). We want to minimize max_k |ifft(|H|^2)_k| ? No, autoconv is ifft(H^2), not ifft(|H|^2).
    # Wait, autoconvolution (h*h) corresponds to ifft(H^2).
    # The magnitude of H^2 is |H|^2. The phase of H^2 is 2*phase(H).
    # The peak of ifft(H^2) is determined by the interference of the frequency components.
    # To minimize the peak of the time-domain signal, we want the frequency components to be "spread out" in phase or amplitude such that they don't constructively interfere at any single time index.
    
    # However, for a real signal h, H[k] = conj(H[N-k]).
    # If we choose h to be a "chirp" or a sequence with random phases in frequency domain, 
    # the time domain autoconvolution will look like noise.
    # The max absolute value of a zero-mean noise sequence of length 2N-1 with variance sigma^2 is typically ~ sqrt(2 * log(2N)) * sigma.
    # The energy is sum(h^2). By Parseval, sum(h^2) = (1/M) sum(|H|^2).
    
    # Let's try a specific structured candidate: A Barker-like sequence or a Zadoff-Chu sequence.
    # Zadoff-Chu sequences have constant modulus in frequency domain (after zero padding/interpolation issues aside) 
    # and low PAPR.
    
    # Alternative: Random Search in Frequency Domain
    # Generate random phases for H, enforce Hermitian symmetry, inverse FFT to get h.
    # Then normalize h so that sum(h) = 1.
    # Then compute score.
    
    # Why this might work:
    # Constant h has H concentrated at DC. autoconv is a triangle. Peak = N. sum(h)=N. Ratio = N/N^2 = 1/N.
    # If we spread H across frequencies, the autoconv becomes "noisier".
    # The peak of the autoconv will be lower relative to the total energy?
    # Let's check the scaling.
    # If h is random +/- 1, sum(h) ~ sqrt(N). autoconv peak ~ sqrt(N) (for large N, max of 2N values of sum of N terms).
    # Ratio ~ sqrt(N) / N = 1/sqrt(N).
    # 1/sqrt(400) = 0.05.
    # 1/400 = 0.0025.
    # So random +/- 1 is WORSE than constant ones.
    
    # Wait, the task asks to MINIMIZE the score.
    # Current best is 1.0 (which corresponds to the seed score? No, the seed returns h1 which has score ~0.0025. 
    # The "score 1.0000" in the prompt text refers to the *search score* seed/value. 
    # If seed value is V_seed, and we return h with value V_new, search score is V_seed / V_new.
    # If V_new < V_seed, score > 1.
    # The current program returns h1 (ones). 
    # Inspiration 1 scored 1.2681. This implies it found a sequence with a LOWER raw value than the seed.
    # How can raw value be lower than 1/N?
    # Theoretical lower bound for max|autoconv| / (sum h)^2:
    # For any h, (sum h)^2 = autoconv(N-1) (the center peak of the triangular support if h is supported on 0..N-1? No, linear autoconv at lag 0 is sum h^2. Lag N-1 is h[0]h[N-1]... 
    # Actually, (sum h)^2 = sum_{i,j} h[i]h[j]. This is exactly the value of the autoconvolution at lag 0? No.
    # Linear autoconv c[k] = sum_i h[i] h[i+k].
    # c[0] = sum h[i]^2.
    # The term (sum h)^2 = (sum h[i]) (sum h[j]) = sum_i sum_j h[i] h[j].
    # This is NOT a single coefficient of the linear autoconvolution.
    # However, note that c[k] for k in -N+1..N-1.
    # The sum of ALL coefficients of the linear autoconvolution is (sum h)^2.
    # So, max_k |c[k]| >= (1 / (2N-1)) * (sum h)^2.
    # Thus, Score >= 1 / (2N-1).
    # For N=400, 1/799 ~ 0.00125.
    # Constant ones: c[k] is a triangle. c[0]=N. Max is N. Sum h = N. Ratio = N/N^2 = 1/N = 0.0025.
    # Lower bound is 1/(2N-1) ~ 0.00125.
    # So there is a factor of 2 improvement possible!
    # To achieve the lower bound, we need the autoconvolution to be as "flat" as possible (constant) across its support.
    # This is related to "Perfect Sequences" or "Golay Complementary Sequences" but for a single sequence, it's about minimizing the peak-to-average power ratio of the autoconvolution.
    
    # If we can make c[k] approximately constant for all k, then max |c[k]| ~ (sum h)^2 / (2N-1).
    # This requires the power spectrum |H|^2 to be flat?
    # If |H|^2 is flat (white noise in freq), then c[k] is an impulse at 0? No, that's correlation.
    # Autoconvolution c[k] = ifft(H^2).
    # If H is flat (white), H^2 is flat (white). ifft(white) is white noise.
    # Then c[k] is white noise. The max of white noise is ~ sqrt(log N) * rms.
    # This is NOT flat.
    
    # We want c[k] to be constant.
    # This implies H^2 must be a delta function in frequency? No, ifft(delta) is constant.
    # So we want H^2 to be concentrated at DC?
    # If H^2 is at DC, H is at DC and Nyquist?
    # If H is only at DC, h is constant. c[k] is triangle. Not flat.
    
    # Let's reconsider the bound.
    # max |c[k]| >= (sum c[k]) / (2N-1) = (sum h)^2 / (2N-1).
    # Equality holds if c[k] is constant.
    # Can c[k] be constant?
    # c[k] = sum_i h[i]h[i+k].
    # This is the autocorrelation (with sign flip if we consider cross-correlation, but autoconv is similar).
    # For a sequence to have constant autocorrelation (except at 0), it's a "Perfect Sequence".
    # Perfect sequences exist only for specific lengths (N=2, 4, 8, 12, 16... powers of 2 and some others).
    # N=400 is not a power of 2.
    # However, we can approximate.
    
    # Strategy: Use a sequence that has "low peak" autoconvolution.
    # Barker sequences have low sidelobes.
    # Let's try to construct a sequence using Gradient Descent on the continuous relaxation.
    # Objective: J(h) = max_k |c[k]| / (sum h)^2.
    # This is non-smooth.
    # Use a soft-max approximation: J_smooth(h) = logsumexp(beta * |c|) / (sum h)^2.
    
    # Let's implement a simple gradient descent step on the continuous domain, then snap to signs?
    # Or just return the continuous result if allowed? 
    # "sequence of signed step heights". Usually implies discrete or specific values.
    # But "signed" might just mean real numbers with signs.
    # If continuous is allowed, we can minimize the max abs value of c[k] subject to sum h = 1.
    # This is a convex problem? max of linear functions is convex. (sum h)^2 is convex (squared norm).
    # Minimizing a convex function over a convex set is convex.
    # But the ratio is not convex.
    # We can fix sum h = 1 and minimize max |c[k]|.
    
    # Let's try this:
    # 1. Start with random h.
    # 2. Normalize sum h = 1.
    # 3. Compute c = autoconv(h).
    # 4. Compute gradient of max |c| wrt h.
    # 5. Update h.
    
    # However, time is limited.
    # Let's try a simpler heuristic: 
    # The "triangle" of the constant sequence has a sharp peak.
    # A sequence that is "two pulses" might have a flatter autoconv?
    # h = [1, 1, ..., 1, -1, -1, ..., -1]
    # Let's test a few structured candidates quickly.
    
    candidates = []
    
    # Candidate: Constant
    candidates.append(np.ones(N))
    
    # Candidate: Square wave (half 1, half -1)
    h_sq = np.ones(N)
    h_sq[N//2:] = -1.0
    candidates.append(h_sq)
    
    # Candidate: Chirp (linear phase in freq)
    freqs = np.fft.rfftfreq(N, d=1.0)
    # Scale freqs to 0..1
    # Phase = 2*pi * t * freq
    # Let's create a chirp in time domain: h[n] = exp(i * pi * n^2 / N)
    # This is a complex chirp. We need real.
    # Real part: cos(pi * n^2 / N)
    h_chirp = np.cos(np.pi * np.arange(N)**2 / N)
    candidates.append(h_chirp)
    
    # Candidate: Random +/- 1 (optimized by local search?)
    # Local search is slow.
    # Let's just evaluate these.
    
    best_h_local = candidates[0]
    best_score_local = _score(candidates[0])
    
    for h_c in candidates[1:]:
        s = _score(h_c)
        if s < best_score_local:
            best_score_local = s
            best_h_local = h_c
    
    if best_score_local < best_score:
        best_h = best_h_local
        best_score = best_score_local
        
    # If we have time, try a few random restarts with local search?
    # No, local search on N=400 is expensive.
    # Let's try a gradient step on the chirp.
    
    if time.time() < deadline - 5.0:
        # Simple gradient descent on continuous h
        # Minimize max |c[k]| subject to sum h = 1.
        # Lagrangian: L = max |c| - lambda (sum h - 1)
        # Grad wrt h: 
        # c[k] = sum_i h[i] h[i+k]
        # dc[k]/dh[j] = h[j-k] + h[j+k] (with bounds)
        
        # We will use a subgradient method.
        h = best_h.copy()
        # Normalize
        s = np.sum(h)
        if s != 0:
            h = h / s # sum h = 1
            
        for step in range(50):
            if time.time() > deadline:
                break
                
            c = np.fft.irfft(np.fft.rfft(h, 2*N-1)**2, 2*N-1)
            # c is length 2N-1.
            # We want to minimize max |c|
            # Find index of max |c|
            idx = np.argmax(np.abs(c))
            sign_c = np.sign(c[idx])
            
            # Gradient of c[idx] wrt h
            # c[idx] = sum_{i} h[i] h[i+idx]
            # d/dh[j] c[idx] = h[j-idx] + h[j+idx] (if indices valid)
            grad = np.zeros(N)
            # Vectorized approach for gradient of a specific lag
            # h[j-idx] is h shifted by idx
            # h[j+idx] is h shifted by -idx
            
            # We need to handle boundaries.
            # Let's use convolution to compute gradients for ALL lags, then pick the one for idx?
            # Or just compute for idx.
            
            # Efficient way:
            # grad[j] = h[j-idx] (if 0 <= j-idx < N) + h[j+idx] (if 0 <= j+idx < N)
            
            if idx >= 0:
                # h[j-idx] -> h shifted right by idx?
                # j-idx = i => i = j-idx.
                # If j-idx is in [0, N-1], then h[j-idx] contributes.
                # This is equivalent to convolving h with a delta at -idx?
                # Let's just loop or use slicing.
                # Slicing is faster.
                
                # Term 1: h[j-idx]
                # j ranges 0..N-1.
                # j-idx ranges -idx .. N-1-idx.
                # Valid j-idx are 0..N-1-idx.
                # So j is idx .. N-1.
                # grad[idx:N] += h[0:N-idx]
                
                # Term 2: h[j+idx]
                # j+idx ranges idx .. N-1+idx.
                # Valid j+idx are idx .. N-1.
                # So j is 0 .. N-1-idx.
                # grad[0:N-idx] += h[idx:N]
                
                grad[idx:N] += h[0:N-idx]
                grad[0:N-idx] += h[idx:N]
                
                # Wait, is this correct?
                # c[idx] = sum_{i=0}^{N-1-idx} h[i] h[i+idx]
                # d/dh[j] sum_{i} h[i] h[i+idx]
                # If j = i, term is h[j] h[j+idx]. Deriv is h[j+idx]. (Contribution to grad[j])
                # If j = i+idx, term is h[j-idx] h[j]. Deriv is h[j-idx]. (Contribution to grad[j])
                
                # So grad[j] = h[j+idx] (if j+idx < N) + h[j-idx] (if j-idx >= 0)
                
                # My slicing:
                # grad[idx:N] += h[0:N-idx]
                # For j in [idx, N-1], grad[j] gets h[j-idx]. Correct.
                # grad[0:N-idx] += h[idx:N]
                # For j in [0, N-1-idx], grad[j] gets h[j+idx]. Correct.
                
                # Multiply by sign_c to minimize |c[idx]|
                grad *= sign_c
                
                # Update h
                # h_new = h - alpha * grad
                # Also enforce sum h = 1?
                # Subtract mean to keep sum 0, then normalize?
                # If we change h, sum h changes.
                # Let's project onto sum h = 1 after update.
                
                alpha = 0.01
                h = h - alpha * grad
                s = np.sum(h)
                if s != 0:
                    h = h / s
                    
                # Check score
                new_score = _score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                # Negative lag?
                # idx from argmax of c. c is computed for lags -N+1 to N-1?
                # np.fft.irfft returns lags 0 to N-1 for real input?
                # No, irfft of length M returns M/2+1 points?
                # We used size 2N-1.
                # The output of irfft is length 2N-1.
                # Indices 0 to 2N-2 correspond to lags 0 to 2N-2?
                # No, linear autoconvolution lags are -N+1 to N-1.
                # FFT autoconv puts lag 0 at index 0.
                # Positive lags 1..N-1 are at indices 1..N-1.
                # Negative lags -1..-(N-1) are at indices N..2N-2.
                # So idx is always >= 0 in the array.
                # But if idx >= N, it represents a negative lag.
                # c[idx] for idx >= N is c[idx - (2N-1)]?
                # Actually, for real sequences, c[-k] = c[k].
                # So we can just use k = idx if idx < N else 2N-1-idx?
                # Wait, c[k] = sum h[i]h[i+k].
                # c[-k] = sum h[i]h[i-k] = sum h[j+k]h[j] = c[k].
                # So symmetry holds.
                # We can map idx to effective lag k.
                if idx >= N:
                    k = 2*N - 1 - idx
                else:
                    k = idx
                    
                # Recompute gradient for lag k
                grad = np.zeros(N)
                grad[k:N] += h[0:N-k]
                grad[0:N-k] += h[k:N]
                grad *= sign_c
                alpha = 0.01
                h = h - alpha * grad
                s = np.sum(h)
                if s != 0:
                    h = h / s
                new_score = _score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()

    return best_h

# EVOLVE-BLOCK-END
