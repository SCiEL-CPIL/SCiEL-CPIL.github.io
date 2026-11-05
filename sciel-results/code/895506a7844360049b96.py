import numpy as np
import time
import math

def evaluate(coeffs, K=2048):
    """
    Evaluate the sup norm of P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden-section refinement on top peaks.
    """
    n = len(coeffs)
    # Pad coefficients to length K for FFT
    c_pad = np.zeros(K, dtype=complex)
    c_pad[:n] = coeffs
    
    # Compute FFT
    # P(e^{-i 2 pi k / K}) = sum c_j e^{-i 2 pi j k / K}
    # Magnitude is same for P(e^{i theta})
    P_vals = np.fft.fft(c_pad)
    mags = np.abs(P_vals)
    
    # Find top 5 peaks for refinement
    top_indices = np.argsort(mags)[-5:]
    
    max_val = 0.0
    
    # Precompute indices for vectorized evaluation in refinement
    js = np.arange(n)
    
    def P_mag(theta):
        # P(e^{i theta}) = sum c_j e^{i j theta}
        # Using coeffs directly
        return np.abs(np.sum(coeffs * np.exp(1j * js * theta)))
        
    for idx in top_indices:
        # Initial theta from FFT bin
        # Note: FFT bin k corresponds to angle -2*pi*k/K.
        # Due to symmetry, we can just use 2*pi*idx/K for search center.
        theta_center = 2 * np.pi * idx / K
        
        # Search interval: half the bin width on each side
        delta = 2 * np.pi / K
        a = theta_center - delta/2
        b = theta_center + delta/2
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = P_mag(c_pt)
        fd = P_mag(d_pt)
        
        for _ in range(40):
            if fc > fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = P_mag(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = P_mag(d_pt)
                
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def construct():
    N = 24
    # Time budget
    max_time = 100.0
    start_time = time.time()
    
    candidates = []
    
    # 1. Spectral Synthesis Initialization
    # We want to find a sequence of +/-1 that minimizes the sup norm.
    # This is equivalent to making the spectrum as flat as possible.
    # We generate candidates by IFFT-ing various "flat" targets and rounding.
    
    K_spec = 4096 # High resolution for synthesis
    
    # Target 1: Pure flat (will result in delta function, so we need to modify)
    # Actually, IFFT of ones is a delta. We want the *inverse* property.
    # If we want a time-domain signal with low peaks, we want a frequency-domain signal that is "spread out" but constrained?
    # No, the standard heuristic is:
    # If we take a random phase spectrum and IFFT, we get a Gaussian-like time domain.
    # If we take a flat magnitude spectrum with random phases, IFFT gives a "white noise" like time domain.
    # We want +/-1 time domain.
    
    # Strategy: 
    # Generate a target spectrum H(f) with flat magnitude |H(f)| = 1 (or tapered).
    # Apply random phases.
    # IFFT to get time domain x(t).
    # Threshold x(t) to +/-1.
    # This creates a sequence that is "coherent" with the flat spectrum goal.
    
    for _ in range(20):
        # Create a flat spectrum with random phases
        # Symmetric spectrum to ensure real time domain
        mag = np.ones(K_spec)
        # Apply a taper to reduce Gibbs artifacts? 
        # A Hann window in frequency domain corresponds to a sinc in time?
        # Let's try a simple cosine taper in frequency
        freq_idx = np.arange(K_spec)
        # Center the spectrum? No, standard FFT binning.
        # Taper: 0.5 + 0.5 * cos(2*pi*freq/K)
        taper = 0.5 + 0.5 * np.cos(2 * np.pi * freq_idx / K_spec)
        mag *= taper
        
        # Random phases
        phases = np.random.uniform(0, 2*np.pi, K_spec)
        # Ensure Hermitian symmetry for real output
        # H[k] = H[K-k]^*
        # H[0] and H[K/2] must be real
        phases[0] = 0
        if K_spec % 2 == 0:
            phases[K_spec//2] = 0
            for k in range(1, K_spec//2):
                phases[K_spec-k] = -phases[k]
        
        spectrum = mag * np.exp(1j * phases)
        
        # IFFT
        x = np.fft.ifft(spectrum).real
        
        # Normalize and threshold to +/-1
        # x is roughly zero-mean.
        # Threshold at 0?
        # If x > 0, coeff = 1, else -1.
        # This might not give exactly 12 ones and 12 minuses, but it doesn't strictly require it.
        # The task says coeffs in {+1, -1}.
        
        coeffs = np.sign(x)
        # Ensure no zeros (replace 0 with 1 or -1 randomly)
        coeffs[coeffs == 0] = 1
        coeffs = coeffs[:N].astype(int)
        
        # Ensure we have valid +/-1
        candidates.append(coeffs.tolist())

    # 2. Add Golay-Rudin-Shapiro as a strong baseline candidate
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(p[:N])
    
    # 3. Add a few random candidates
    for _ in range(5):
        candidates.append([1 if np.random.random() < 0.5 else -1 for _ in range(N)])

    # 4. Local Search (Bit Flip) from best candidates
    # We will run a simple hill climbing / simulated annealing hybrid
    # on the best few spectral candidates.
    
    best_coeffs = None
    best_score = float('inf')
    
    # Sort candidates by initial score (using fast evaluation)
    def fast_score(coeffs):
        return evaluate(coeffs, K=512)
        
    scored_candidates = [(fast_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 candidates for refinement
    top_candidates = [c for s, c in scored_candidates[:5]]
    
    for init_coeffs in top_candidates:
        current_coeffs = init_coeffs[:]
        current_score = fast_score(current_coeffs)
        
        T = 0.01 # Start with low temperature, we are in a good basin
        cooling_rate = 0.9999
        max_iter = 50000
        
        # Simulated Annealing
        for i in range(max_iter):
            if time.time() - start_time > max_time - 10: # Leave time for final check
                break
                
            # Pick random flip
            idx = np.random.randint(N)
            new_coeffs = current_coeffs[:]
            new_coeffs[idx] = -new_coeffs[idx]
            
            new_score = fast_score(new_coeffs)
            delta = new_score - current_score
            
            if delta < 0:
                current_coeffs = new_coeffs
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = new_coeffs[:]
            else:
                # Accept with probability exp(-delta/T)
                if np.random.random() < np.exp(-delta / T):
                    current_coeffs = new_coeffs
                    current_score = new_score
            
            T *= cooling_rate
            if T < 1e-6:
                break

    # Final refinement with high accuracy evaluation
    if best_coeffs is not None:
        # Do a few rounds of exact local search on the best found
        # Try flipping each bit, keep if it improves the HIGH-ACC score
        improved = True
        while improved and (time.time() - start_time < max_time - 2):
            improved = False
            base_score = evaluate(best_coeffs, K=4096)
            for i in range(N):
                trial = best_coeffs[:]
                trial[i] = -trial[i]
                trial_score = evaluate(trial, K=4096)
                if trial_score < base_score:
                    best_coeffs = trial
                    base_score = trial_score
                    improved = True
                    break # Restart scan as coeffs changed

    return best_coeffs
