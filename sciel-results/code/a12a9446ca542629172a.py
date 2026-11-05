import numpy as np
import time

N = 400

def _evaluate(h):
    """Compute the objective value."""
    s = np.sum(h)
    if s == 0:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    max_val = np.max(conv)
    return abs(2 * N * max_val / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Strategy: Frequency domain optimization
    # The autoconvolution c[k] is related to the power spectrum S[f] = |H[f]|^2
    # Specifically, c = ifft(fft(h) * conj(fft(h)))
    # We parameterize h by its magnitude spectrum |H[f]| and a random phase.
    # However, to ensure h is real, we must enforce Hermitian symmetry on H.
    # H[0] is real, H[N/2] is real, H[k] and H[N-k] are conjugates.
    # So we can define a vector of magnitudes m of length N//2 + 1.
    # m[0] -> H[0]
    # m[1..N//2-1] -> |H[k]|
    # m[N//2] -> H[N/2]
    
    # We will optimize the log-magnitudes to keep them positive and well-scaled.
    # Let x be the log-magnitudes.
    # h = ifft( exp(x) * exp(1j * phase) )
    # To maximize the chance of finding a good solution, we can fix the phase to zero 
    # initially? No, phase matters for the time-domain shape.
    # Actually, for a given power spectrum, the "worst case" peak of the autocorrelation 
    # depends on the phase. But we can choose the phase.
    # A simpler approach: Optimize the power spectrum P[f] directly (non-negative).
    # Then h is determined up to phase. 
    # If we fix phase to 0, h is real and even (symmetric). 
    # Symmetric h has real, even autoconvolution.
    # Is it beneficial to allow arbitrary phase? 
    # The objective depends on max(conv). 
    # For a fixed power spectrum, the maximum of the autocorrelation is bounded by the DC component (sum h^2).
    # Wait, the DC component of the autoconvolution is sum(h^2).
    # The peak of the autoconvolution is always >= sum(h^2) / N (by Parseval/averaging)?
    # No, the mean of the autoconvolution is (sum h)^2 / (2N-1) approx.
    # The peak is at least the average.
    # If we make the spectrum flat (white noise), the autoconvolution is a spike at 0.
    # The peak is sum(h^2).
    # The denominator is (sum h)^2.
    # For white noise, sum h ~ 0, so denominator is small? No, we can shift DC.
    
    # Let's try optimizing the power spectrum P = |H|^2.
    # We will use Simulated Annealing on the log-power spectrum.
    # To keep h real, we construct H with Hermitian symmetry.
    # We will fix the phase to be random but fixed during the run? 
    # Or optimize phase too? That doubles the dimension.
    # Let's start with random phase fixed, and optimize magnitudes.
    # If that doesn't work, we can try optimizing both.
    # Given the time limit, let's try optimizing magnitudes only with a fixed random phase.
    # Actually, a better fixed phase is 0 (symmetric h).
    # Let's try symmetric h first (phase = 0).
    
    # Dimensions: N//2 + 1 = 201.
    # This is a high dimension for SA, but SA scales okay.
    # However, 200 evaluations per second is fast in numpy.
    # 150s * 1000 evals/s = 150,000 evals. Plenty for SA.
    
    # Let's use a smarter initialization.
    # Start with flat spectrum (white noise) + DC bias.
    # Flat spectrum -> h is white noise (with phase 0, it's a cosine sum, which is not white noise in time domain? 
    # If phase is 0, H is real and even. ifft of real even is real even.
    # A flat magnitude spectrum with zero phase corresponds to a delta function in time? No.
    # H[k] = 1 for all k. h[0] = 1, h[others] = 0.
    # That's a spike. Autoconvolution is a spike.
    # Value: 2 * 400 * 1 / 1^2 = 800. Bad.
    
    # We want to minimize max(conv) / (sum h)^2.
    # This is related to the "peak-to-average power ratio" of the autoconvolution.
    # We want a "flat" autoconvolution.
    # A flat autoconvolution corresponds to a "spiky" spectrum?
    # No, flat time domain -> spiky frequency domain.
    # Spiky time domain -> flat frequency domain.
    # We want flat autoconvolution (low peak).
    # This implies we want a "spiky" power spectrum?
    # If P[f] is spiky (concentrated in few bins), h is a sum of few sinusoids.
    # This is what the current program does (low freq sinusoids).
    # Current score 1.26.
    
    # What if we use a "Golay-like" or "CAZAC" (Constant Amplitude Zero Autocorrelation) sequence?
    # Zadoff-Chu sequences have constant amplitude and zero periodic autocorrelation (except at lag 0).
    # For aperiodic autocorrelation, they are very good.
    # Let's try generating a Zadoff-Chu sequence and scaling it.
    
    # Zadoff-Chu sequence: x[n] = exp(-j * pi * u * n * (n+1) / N)
    # It has constant amplitude 1.
    # sum(x) = 0 for N > 1? 
    # If sum(x) = 0, denominator is 0. We must add a DC offset.
    # Let h = A * x + B.
    # We can optimize A and B.
    
    # Let's try this specific structural candidate first.
    # If it's good, we can use it as a starting point for local search.
    
    best_candidate = best_h
    best_candidate_val = best_val
    
    # Try Zadoff-Chu with different roots u
    for u in range(1, N, 2):
        if np.gcd(u, N) != 1:
            continue
        n = np.arange(N)
        # Zadoff-Chu
        zc = np.exp(-1j * np.pi * u * n * (n + 1) / N)
        # Real part? Or use complex? 
        # h must be real. 
        # We can take real part, or use a pair of conjugate frequencies.
        # Let's use the real part of the Zadoff-Chu sequence? 
        # Or better, construct a real sequence from two conjugate Zadoff-Chu sequences?
        # Actually, simply taking the real part of a complex Zadoff-Chu sequence 
        # gives a real sequence with similar autocorrelation properties (approximately).
        # Let's try h = cos(pi * u * n * (n+1) / N)
        h_zc = np.cos(np.pi * u * n * (n + 1) / N)
        
        # Add DC offset to make sum non-zero and potentially improve ratio.
        # Optimize DC offset b for fixed amplitude a=1.
        # h = h_zc + b
        # sum(h) = sum(h_zc) + N*b. sum(h_zc) is small.
        # Let's just try a few b values.
        for b in np.linspace(-0.5, 0.5, 11):
            h_test = h_zc + b
            val = _evaluate(h_test)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = h_test.copy()
    
    if best_candidate_val < best_val:
        best_h = best_candidate
        best_val = best_candidate_val
    else:
        # Fallback to SA on log-magnitudes if ZC didn't beat seed
        # We will run SA on the log-magnitude spectrum.
        # Dimension 201.
        # Start from current best_h spectrum.
        H_best = np.fft.fft(best_h)
        mags = np.abs(H_best)
        # Log magnitudes, handle zeros
        log_mags = np.log(mags + 1e-10)
        
        # Symmetric log mags:
        # We only need first N//2 + 1.
        # Let's define a function to reconstruct h from log_mags (first half) and fixed phase 0.
        # Phase 0 means H is real and non-negative? 
        # If H is real and non-negative, h is real and even.
        # This restricts us to even functions.
        # Is that a bad restriction? 
        # The optimal h might not be even.
        # But let's try it. It's a structural change from sinusoidal basis to spectral shaping.
        
        def _h_from_log_mags(lm):
            # lm is length N//2 + 1
            # Construct H
            H = np.zeros(N, dtype=complex)
            H[0] = np.exp(lm[0])
            if N % 2 == 0:
                H[N//2] = np.exp(lm[N//2])
            for k in range(1, N//2):
                H[k] = np.exp(lm[k])
                H[N-k] = np.exp(lm[k])
            # Phase is 0, so H is real and positive.
            h = np.fft.ifft(H).real
            return h
        
        # SA Parameters
        T = 1.0
        T_min = 1e-6
        alpha = 0.999
        max_iters = 50000
        current_lm = log_mags[:N//2 + 1].copy()
        current_h = _h_from_log_mags(current_lm)
        current_val = _evaluate(current_h)
        
        if current_val < best_val:
            best_h = current_h
            best_val = current_val
            
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if i % 100 == 0 and T < T_min:
                break
                
            # Perturb one or a few log magnitudes
            idx = np.random.randint(0, N//2 + 1)
            delta = np.random.normal(0, 0.5)
            new_lm = current_lm.copy()
            new_lm[idx] += delta
            new_h = _h_from_log_mags(new_lm)
            new_val = _evaluate(new_h)
            
            if new_val < current_val:
                current_lm = new_lm
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                # Accept with probability exp(-(new_val - current_val) / T)
                prob = np.exp(-(new_val - current_val) / T)
                if np.random.rand() < prob:
                    current_lm = new_lm
                    current_h = new_h
                    current_val = new_val
                    
            T *= alpha
            
        # Local polish on the best found h using Nelder-Mead on time domain?
        # No, time domain is 400D, too high for NM.
        # Polish on log_mags?
        from scipy.optimize import minimize
        def _obj_lm(lm):
            h = _h_from_log_mags(lm)
            return _evaluate(h)
        
        try:
            res = minimize(_obj_lm, current_lm, method='Nelder-Mead', options={'maxiter': 500})
            if res.fun < best_val:
                best_h = _h_from_log_mags(res.x)
                best_val = res.fun
        except:
            pass

    return best_h

if __name__ == "__main__":
    h = construct()
    print("Value:", _evaluate(h))
    print("Sum:", np.sum(h))
