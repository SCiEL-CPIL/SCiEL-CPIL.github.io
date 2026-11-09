import numpy as np
import time
import scipy.optimize as opt

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3 (interpreted as peak third-order autocorrelation).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)

    def compute_triple_conv_peak(h):
        """Compute peak of triple convolution (h*h*h)."""
        n = len(h)
        nfft = 1 << (3 * n - 2).bit_length()
        H = np.fft.rfft(h, n=nfft)
        triple = np.fft.irfft(H ** 3, n=nfft)
        # Valid part: indices n-1 to 3n-3 (inclusive), so length 2n-2
        valid = triple[n-1:3*n-2]
        return np.max(np.abs(valid))

    def compute_autocorr_sidelobe_energy(h):
        """
        Computes the energy of the autocorrelation sidelobes.
        This is a smooth proxy for peak minimization.
        """
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        # R[0] is the main lobe (power). We want to minimize the rest.
        # Due to circular convolution in FFT, the sidelobes wrap around.
        # For linear autocorrelation, we take R[1:n] and R[nfft-n+1:nfft].
        # Actually, simpler: just sum squares of R[1:n] and R[nfft-n+1:nfft]
        # Or just sum squares of all R except R[0] if we assume periodic?
        # Let's use linear autocorrelation energy.
        # R is circular. 
        # Linear autocorr r[t] for t>0 is R[t]. For t<0 is R[nfft-t].
        # Energy = sum_{t=1}^{n-1} (r[t]^2 + r[-t]^2) = 2 * sum_{t=1}^{n-1} R[t]^2
        # Wait, R from irfft is real.
        # Let's just compute linear autocorrelation directly for safety in the proxy, 
        # or use the FFT trick carefully.
        # For a proxy, minimizing sum of squares of non-zero lags is good.
        # Using FFT:
        # Power = R[0]
        # Sidelobe energy = sum(R[1:n])^2 + sum(R[nfft-n+1:nfft])^2
        # Note: R[nfft-n+1] corresponds to lag -(n-1).
        # Let's verify indices.
        # Lag t corresponds to index t for t >= 0.
        # Lag -t corresponds to index nfft - t.
        # So lags 1..n-1 are indices 1..n-1.
        # Lags -(n-1)..-1 are indices nfft-(n-1)..nfft-1.
        energy = np.sum(R[1:n]**2) + np.sum(R[nfft-n+1:nfft]**2)
        return energy

    def generate_waveform(params, N):
        """
        Generate a continuous waveform from parameters.
        params: [A1, f1, phi1, A2, f2, phi2, ...]
        """
        t = np.arange(N)
        wave = np.zeros(N)
        num_harmonics = len(params) // 3
        for i in range(num_harmonics):
            A = params[i*3]
            f = params[i*3+1]
            phi = params[i*3+2]
            wave += A * np.sin(2 * np.pi * f * t + phi)
        return wave

    def objective(params):
        """
        Objective function for L-BFGS.
        Minimizes the sidelobe energy of the sign-quantized waveform?
        No, sign is not differentiable.
        We minimize the sidelobe energy of the *continuous* waveform,
        hoping that the sign-preserving structure yields a good binary sequence.
        """
        wave = generate_waveform(params, N)
        # To make the objective differentiable w.r.t params, 
        # we should use the continuous wave's autocorrelation.
        # But we care about the binary sequence.
        # A common trick: minimize the variance of the continuous wave's autocorrelation sidelobes.
        # Or, since we will binarize, maybe we should penalize values close to 0 
        # to ensure robust sign determination?
        # Let's stick to minimizing the continuous sidelobe energy.
        
        # Compute autocorrelation of the continuous wave
        n = len(wave)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(wave, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        
        # Main lobe is R[0]
        # Sidelobes are the rest
        # We want to minimize the sum of squares of sidelobes
        # To avoid scaling issues, normalize by main lobe?
        # Let's just minimize the raw energy of sidelobes.
        energy = np.sum(R[1:n]**2) + np.sum(R[nfft-n+1:nfft]**2)
        
        # Add a penalty for small amplitudes to encourage clear signs
        # If wave is close to 0, the sign is unstable.
        # Penalize 1/sqrt(1 + wave^2) ? No, just add a small regularization.
        # Actually, if we minimize sidelobe energy, the wave might collapse to 0.
        # We need to constrain the main lobe energy.
        # Let's normalize the waveform to have unit power before computing autocorrelation?
        # No, that makes it non-linear in params.
        
        # Better: Minimize (Sidelobe Energy / Main Lobe Energy)
        # This is scale-invariant.
        main_lobe = R[0]
        if main_lobe < 1e-6:
            return 1e6
        return energy / main_lobe

    def discrete_local_search(h, max_iterations=500):
        """Quick greedy local search on the discrete sequence."""
        current_score = compute_triple_conv_peak(h)
        improved = True
        iterations = 0
        n = len(h)
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break
            
            # Try a few random flips
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                i = np.random.randint(n)
                h[i] *= -1
                new_score = compute_triple_conv_peak(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i] *= -1
        return h, current_score

    best_h = None
    best_score = float('inf')
    
    # Number of harmonics to use
    # Each harmonic has 3 params: Amplitude, Frequency, Phase
    # Let's try 3 harmonics (9 params) and 5 harmonics (15 params)
    num_harmonics_list = [3, 5]
    
    np.random.seed(42)
    
    for K in num_harmonics_list:
        if time.time() > deadline - 30:
            break
            
        # Initial parameters
        # Frequencies should be distinct and not too close to avoid interference
        # Amplitudes positive
        # Phases random
        
        for restart in range(5):
            if time.time() > deadline - 15:
                break
                
            # Initialize params
            # Frequencies: spread out in [0.01, 0.5]
            freqs = np.sort(np.random.uniform(0.01, 0.5, K))
            amps = np.random.uniform(1.0, 2.0, K)
            phis = np.random.uniform(0, 2*np.pi, K)
            
            params = np.concatenate([amps, freqs, phis])
            
            # Bounds
            # Amps: [0.1, 5.0]
            # Freqs: [0.01, 0.5]
            # Phis: [0, 2*pi]
            bounds = []
            for i in range(K):
                bounds.append((0.1, 5.0)) # Amp
                bounds.append((0.01, 0.5)) # Freq
                bounds.append((0, 2*np.pi)) # Phi
            
            try:
                result = opt.minimize(objective, params, method='L-BFGS-B', bounds=bounds, 
                                      options={'maxiter': 100, 'ftol': 1e-6})
                
                if result.success or result.fun < 1e6:
                    # Generate waveform
                    wave = generate_waveform(result.x, N)
                    # Binarize
                    h = np.sign(wave).astype(np.int8)
                    # Handle zeros (rare but possible)
                    h[h == 0] = 1
                    
                    # Refine with discrete search
                    h, score = discrete_local_search(h, max_iterations=200)
                    
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
                        
            except Exception as e:
                pass

    # If no good result from continuous optimization, fallback to a simple random search
    if best_h is None:
        # Try a few random sequences
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            h = np.random.choice([-1, 1], N)
            h, score = discrete_local_search(h, max_iterations=100)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if best_h is None:
            best_h = np.random.choice([-1, 1], N)

    return best_h
