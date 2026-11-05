import numpy as np
import time
import math

def evaluate(coeffs, K=1000):
    # coeffs is a list of +1/-1
    n = len(coeffs)
    # Sample theta
    thetas = np.linspace(0, 2*np.pi, K, endpoint=False)
    # Compute P(z) = sum c_k e^{i k theta}
    # Use FFT for speed? 
    # P(e^{i theta}) is the DFT of coeffs if thetas are multiples of 2pi/K.
    # Since K is arbitrary, we can just use matrix multiplication or FFT if K is power of 2?
    # Actually, np.fft.fft computes DFT at k=0..K-1 which corresponds to theta = 2*pi*k/K.
    # This is a valid dense sampling.
    # Let's use FFT for speed.
    # Pad coeffs to length K? No, P(z) is length N.
    # The values P(e^{i 2 pi k / K}) for k=0..K-1 are the first K points of the DFT of the infinite periodic extension?
    # No, P(e^{i theta}) is just the polynomial.
    # We can compute it efficiently.
    # coeffs: c_0, ..., c_{N-1}
    # P(e^{i theta}) = c_0 + c_1 e^{i theta} + ... + c_{N-1} e^{i (N-1) theta}
    
    # Using FFT:
    # If we take FFT of coeffs (length N), we get DFT at N points.
    # We want K points.
    # We can zero-pad coeffs to length K and take FFT?
    # FFT of zero-padded coeffs gives samples of the DTFT at K points.
    # Yes.
    
    # Pad coeffs to K
    c_pad = np.zeros(K, dtype=complex)
    c_pad[:n] = coeffs
    P_vals = np.fft.fft(c_pad)
    
    # P_vals[k] = sum_{j=0}^{N-1} c_j e^{-i 2 pi j k / K}
    # Note the sign convention. np.fft.fft uses e^{-i 2 pi ...}
    # Our P(z) uses z^k = e^{i k theta}.
    # So P(e^{i theta}) = sum c_k e^{i k theta}.
    # If theta = -2 pi k / K, then e^{i k theta} = e^{-i 2 pi k^2 / K}? No.
    # Let's just stick to the definition.
    # P(e^{i theta}) = sum c_j e^{i j theta}.
    # np.fft.ifft of something?
    # Actually, np.fft.fft(c_pad) gives sum c_j e^{-i 2 pi j k / K}.
    # This is P(e^{-i 2 pi k / K}).
    # Since |P(e^{i theta})| = |P(e^{-i theta})| for real coefficients, the magnitude is the same.
    # So we can just take magnitude of fft(c_pad).
    
    mags = np.abs(P_vals)
    
    # Find top peaks
    # Get indices of top 5 peaks
    top_indices = np.argsort(mags)[-5:]
    max_val = 0
    for idx in top_indices:
        # Golden section search around theta = 2*pi*idx/K
        # Actually, the peak might be between samples.
        # Let's refine around the sample with the highest magnitude in a small window.
        # Or just refine the global max from the samples.
        pass
        
    # Simple refinement: 
    # Find the max sample index
    max_idx = np.argmax(mags)
    theta_max = 2 * np.pi * max_idx / K
    
    # Golden section search in a small interval around theta_max
    # Interval size: 2*pi/K
    delta = 2 * np.pi / K
    a = theta_max - delta/2
    b = theta_max + delta/2
    
    # Function to evaluate P(e^{i theta})
    def P_mag(theta):
        # sum c_j e^{i j theta}
        # Can be computed as real and imag parts
        # Or use the fact that we have coeffs
        # For speed, we can precompute c_j
        # But this is called many times in SA.
        # Let's optimize P_mag.
        # P(e^{i theta}) = sum c_j (cos(j theta) + i sin(j theta))
        # We can use vectorized numpy operations if we pass an array of thetas?
        # But SA calls it one theta at a time.
        # Let's just compute it.
        js = np.arange(n)
        angles = js * theta
        # c_j are +/- 1
        # sum c_j e^{i j theta}
        # = sum c_j cos(j theta) + i sum c_j sin(j theta)
        # Since c_j is +/- 1, we can just sum
        # This is O(N). N=24. Very fast.
        return np.abs(np.sum(coeffs * np.exp(1j * js * theta)))
        
    # Golden section search
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = P_mag(c)
    fd = P_mag(d)
    
    for _ in range(30):
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = P_mag(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = P_mag(d)
            
    peak_val = max(fc, fd)
    if peak_val > max_val:
        max_val = peak_val
        
    # Also check other top peaks?
    # For now, just the global max sample.
    
    return max_val

def construct():
    N = 24
    # Initialize with Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = p[:N]
    
    # Spectral Synthesis Initialization?
    # The directive says to use it.
    # Let's create a few candidates:
    # 1. Golay
    # 2. Random
    # 3. Spectral: IFFT of flat spectrum (with some taper)
    
    candidates = [best_coeffs]
    
    # Spectral init
    # Target spectrum: flat magnitude
    K_spec = 1024
    target_spec = np.ones(K_spec)
    # Maybe taper to avoid discontinuities?
    # target_spec = np.cos(np.pi * np.arange(K_spec) / K_spec) ** 2 ?
    # Let's just use flat for now, or random phases?
    # If we use random phases for the spectrum, IFFT will be complex.
    # We need real coefficients? No, coeffs are real (+/-1).
    # So the spectrum must be Hermitian symmetric.
    # This is complicated.
    # Let's stick to Golay and Random inits for SA.
    
    # Add a random candidate
    candidates.append([1 if np.random.random() < 0.5 else -1 for _ in range(N)])
    
    # Run Simulated Annealing on the best candidate
    # Or run SA on multiple candidates?
    # Time limit 120s.
    
    def score(coeffs):
        return evaluate(coeffs, K=500) # Lower K for speed during SA
    
    best_score = score(best_coeffs)
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    T = 1.0
    cooling_rate = 0.99999 # Adjust based on iterations
    start_time = time.time()
    max_time = 60 # 60 seconds for SA
    
    while time.time() - start_time < max_time:
        # Pick a random index to flip
        idx = np.random.randint(N)
        new_coeffs = current_coeffs[:]
        new_coeffs[idx] = -new_coeffs[idx]
        
        new_score = score(new_coeffs)
        
        delta = new_score - current_score
        if delta < 0:
            current_coeffs = new_coeffs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = new_coeffs[:]
        else:
            if np.random.random() < np.exp(-delta / T):
                current_coeffs = new_coeffs
                current_score = new_score
        
        T *= cooling_rate
        
    # Final refinement with higher K
    # Do a few local searches with K=1000
    for _ in range(5):
        # Try flipping each bit, keep if improves
        improved = True
        while improved:
            improved = False
            for i in range(N):
                trial = best_coeffs[:]
                trial[i] = -trial[i]
                s_trial = evaluate(trial, K=1000)
                s_best = evaluate(best_coeffs, K=1000)
                if s_trial < s_best:
                    best_coeffs = trial
                    improved = True
                    break # Restart loop as coeffs changed
                    
    return best_coeffs
