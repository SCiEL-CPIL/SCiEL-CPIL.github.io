"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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

def _fourier_papr(h):
    """
    Compute a Fourier-domain proxy for the objective.
    Based on Parseval's identity, the energy of the autoconvolution is related to the 4th moment of the spectrum.
    Specifically, sum(ac^2) = (1/2pi) * integral(|H|^4).
    Minimizing the max|ac| is correlated with flattening the spectrum (minimizing PAPR of |H|^2).
    This proxy is much cheaper to evaluate in the frequency domain directly.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    # Power spectrum
    P = np.abs(H)**2
    # Mean power
    mean_p = np.mean(P)
    if mean_p < 1e-15:
        return 1e10
    # Peak to average power ratio
    papr = np.max(P) / mean_p
    # Also consider the 4th moment which relates to energy of ac
    # sum(ac^2) is proportional to sum(P^2)
    # So minimizing sum(P^2) / (sum(h))^4 is a good proxy for minimizing max|ac| / (sum(h))^2
    # Let's use a combination: minimize sum(P^2) normalized by energy^2
    energy = np.sum(P)
    if energy < 1e-15:
        return 1e10
    # Normalized 4th moment
    return np.sum(P**2) / (energy**2)

def _spectral_flatness_optimization(N, max_time=140.0):
    """
    Strategy:
    1. Initialize h with random +/- 1.
    2. Compute FFT H.
    3. Identify the frequency bin with the highest magnitude |H[k]|.
    4. This high peak in H contributes quadratically to the autoconvolution peak (via inverse FFT).
    5. To reduce this peak, we should flip bits in h that are constructively adding to this frequency component.
    6. The contribution of bit i to frequency k is exp(-j 2 pi k i / N).
    7. If h[i] * exp(-j 2 pi k i / N) has a positive real part, flipping h[i] will reduce |H[k]|.
    8. We iterate this process, potentially using a "greedy" flip that gives the largest reduction in the spectral peak.
    
    This is a structural change from time-domain local search to frequency-domain guided local search.
    """
    start_time = time.time()
    
    # Initialize with random +/- 1
    h = np.random.choice([-1, 1], size=N)
    
    # Normalize sum to be close to 1? No, the score is scale invariant.
    # But we need to ensure sum is not zero.
    # Random +/- 1 usually has small sum.
    # Let's start with a structure that has a known non-zero sum, e.g., mostly 1s with some -1s.
    # Or just random.
    
    best_score = _score(h)
    best_h = h.copy()
    
    # Precompute the complex exponentials for the top few frequency bins?
    # No, we just need the arg of H[k] to decide the flip.
    
    # We will perform a local search where the "move" is determined by the spectral peak.
    # Instead of trying all N flips, we only try the flip that best reduces the dominant spectral peak.
    
    # To make it more robust, we can look at the top K frequency bins.
    K = 10
    
    while time.time() - start_time < max_time:
        # Compute FFT
        fsize = 1
        while fsize < 2*N - 1:
            fsize *= 2
        H = np.fft.rfft(h, fsize)
        P = np.abs(H)**2
        
        # Find top K frequency bins (excluding DC if possible, as DC corresponds to sum)
        # Actually, DC is important.
        # Let's just take the top K indices.
        top_k_indices = np.argsort(P)[-K:]
        
        improved = False
        # For each top frequency bin, find the bit flip that most reduces its magnitude
        # |H[k]|^2 = |sum h[i] exp(-j 2 pi k i / N)|^2
        # Let S = sum h[i] e^{-j \theta_{k,i}}. H[k] = S.
        # If we flip h[i] to -h[i], the new sum is S' = S - 2 h[i] e^{-j \theta_{k,i}}.
        # We want to minimize |S'|^2.
        # |S - 2 h[i] e^{-j \theta}|^2 = |S|^2 - 4 h[i] Re(S e^{j \theta}) + 4.
        # To minimize this, we want to maximize h[i] Re(S e^{j \theta}).
        # Note: e^{j \theta_{k,i}} = e^{j 2 pi k i / N}.
        # S e^{j \theta_{k,i}} = H[k] e^{j 2 pi k i / N}.
        # So we want to maximize h[i] Re(H[k] e^{j 2 pi k i / N}).
        
        for k in top_k_indices:
            # Calculate the phase term for all i
            # theta_i = 2 pi k i / N
            # We need Re(H[k] * exp(j theta_i))
            # H[k] is complex.
            # Let H[k] = A exp(j phi).
            # Re(A exp(j (phi + theta_i))) = A cos(phi + theta_i).
            # We want to find i that maximizes h[i] * A cos(phi + theta_i).
            # This is equivalent to finding i that maximizes h[i] * cos(phi + 2 pi k i / N).
            
            # To do this efficiently, we can compute the values for all i.
            # i = 0..N-1
            i_vals = np.arange(N)
            phases = 2 * np.pi * k * i_vals / N
            # H[k] is H[k]
            # Re(H[k] * exp(j phases))
            # H[k] * exp(j phases) = H[k].real * cos(phases) - H[k].imag * sin(phases) + j(...)
            # Real part is H[k].real * cos(phases) - H[k].imag * sin(phases)
            real_parts = H[k].real * np.cos(phases) - H[k].imag * np.sin(phases)
            
            # We want to maximize h[i] * real_parts[i]
            scores = h * real_parts
            best_i = np.argmax(scores)
            
            # Check if flipping best_i actually improves the score
            # We should check the actual objective function, not just the proxy, to be safe.
            # But evaluating the full objective is O(N log N).
            # We can do a quick check on the proxy _fourier_papr or just the peak P.
            
            # Let's try the flip
            old_val = h[best_i]
            h[best_i] = -old_val
            
            # Evaluate new score (expensive, but we only do it once per iteration)
            new_score = _score(h)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                # Revert
                h[best_i] = old_val
                
        if not improved:
            # If no improvement in the top K bins, try random flips to escape local minima
            # Or break?
            # Let's try a few random flips.
            random_improved = False
            for _ in range(10):
                i = np.random.randint(N)
                old_val = h[i]
                h[i] = -old_val
                new_score = _score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                    random_improved = True
                else:
                    h[i] = old_val
            if not random_improved:
                # Stuck in local minimum. Restart with new random initialization?
                # Or perturb the current solution?
                # Let's restart with a new random sequence.
                h = np.random.choice([-1, 1], size=N)
                # Ensure sum is not zero
                s = np.sum(h)
                if abs(s) < 1e-6:
                    h[0] = 1 if s >= 0 else -1
                best_score = _score(h)
                best_h = h.copy()
                
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0 # Safety margin
    
    # Run spectral flatness optimization
    try:
        h_best, score_best = _spectral_flatness_optimization(N, max_time=deadline - t_start - 5)
    except Exception:
        # Fallback to simple random
        h_best = np.random.choice([-1, 1], size=N)
        score_best = _score(h_best)
    
    # Final local refinement: flip individual bits
    if time.time() < deadline - 10:
        h_refined = h_best.copy()
        current_val = score_best
        improved = True
        passes = 0
        while improved and passes < 5 and time.time() < deadline - 5:
            improved = False
            passes += 1
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                h_refined[i] = -h_refined[i]
                new_val = _score(h_refined)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    h_refined[i] = -h_refined[i] # revert
            if current_val < score_best:
                score_best = current_val
                h_best = h_refined.copy()
            else:
                if not improved:
                    break
                
    return h_best

# EVOLVE-BLOCK-END
