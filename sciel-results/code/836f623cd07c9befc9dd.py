import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill
    
    # Pre-compute FFT size
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def max_autocorr(h):
        """Compute the maximum autocorrelation value (excluding lag 0)."""
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        # The autocorrelation is symmetric. 
        # acf[0] is the energy (sum of squares = N).
        # We want the max of the sidelobes.
        # Due to circular convolution in FFT, we need to be careful about indexing.
        # For linear autocorrelation of length N, the FFT length should be >= 2N-1.
        # The resulting acf from ifft has length n_fft.
        # The linear autocorrelation is contained in acf[0:N-1] and acf[n_fft-(N-1):n_fft].
        # Actually, for a real sequence, the autocorrelation is real and even.
        # acf[k] corresponds to lag k.
        # We want max over k=1..N-1.
        # In the FFT output, lags 1..N-1 are at indices 1..N-1.
        # Lags -1..-(N-1) are at indices n_fft-1 .. n_fft-(N-1).
        # So we can just take max of acf[1:N] and acf[n_fft-N+1:n_fft].
        # Or simply max(acf[1:N-1]) is often sufficient if we assume symmetry and just check positive lags.
        # Let's check both to be safe, though for real signals they are equal.
        return np.max(np.concatenate([acf[1:N], acf[n_fft-N+1:n_fft]]))

    def grosch_pratt(h, max_iters=1000):
        """
        Iterative optimization in the frequency domain.
        1. Compute FFT of current h.
        2. Modify magnitude to reduce peak sidelobe.
           A common heuristic is to reduce the magnitude at frequencies where the 
           autocorrelation peak is high, or more simply, enforce a "flat" or 
           "low-pass" magnitude profile.
           However, a more direct approach for minimizing peak sidelobe is:
           Compute the current autocorrelation A(k).
           Identify the peak sidelobe A_peak.
           The relationship is A(k) = IFFT(|H|^2).
           We want to minimize max(A(k)).
           
           Alternative: "Spectral Shaping"
           Let H = FFT(h).
           Let A = IFFT(|H|^2).
           We want to modify H such that the new A has lower peaks.
           One method: 
           H_new = H * exp(-alpha * A) ? No, A is in time domain.
           
           Let's use the standard "Grosch-Pratt" style iteration:
           1. Compute H = FFT(h).
           2. Compute A = IFFT(|H|^2).
           3. Identify the "bad" lags (high autocorrelation).
           4. Construct a target magnitude |H_target|.
              A simple effective heuristic: 
              Reduce the magnitude of H at frequencies that contribute to the peak sidelobe.
              This is complex.
              
           Simpler effective heuristic: 
              1. Compute H.
              2. Compute A.
              3. If max(A[1:N]) > threshold, 
                 Modify H by multiplying by a window that suppresses high energy?
                 
           Let's try a different spectral approach: 
           Minimize the peak of the power spectrum? No, that's not autocorrelation.
           
           Let's use the "Iterative Amplitude Correction" (IAC) or similar.
           Actually, a very effective method for low autocorrelation sequences is:
           1. Start with random h.
           2. Compute H.
           3. Compute A.
           4. Find the lag k_peak with max A.
           5. This is hard to map back to H directly.
           
           Let's stick to a robust hybrid: 
           Use the spectral initialization to get a good start, then refine with local search.
           But the prompt asks for a MATERIAL change. 
           
           Let's try: **Spectral Phase Retrieval with Peak Sidelobe Constraint**
           Target: Minimize max(A(k), k!=0).
           We can approximate this by minimizing the variance of A(k) or by pushing down the peaks.
           
           Iteration:
           1. h = current sequence.
           2. H = FFT(h).
           3. A = IFFT(|H|^2).
           4. Create a "target" autocorrelation A_target which is 0 everywhere except lag 0.
              But we can't achieve that. 
              Instead, we can penalize high values in A.
              
           Let's try a simpler spectral heuristic that often works:
           "Flatten the spectrum" tends to increase sidelobes.
           "Low-pass" the spectrum?
           
           Actually, the most effective spectral method for this specific problem (minimizing autocorrelation peaks) is often:
           1. Compute H.
           2. Compute A.
           3. Identify the peak sidelobe value P.
           4. Construct a new magnitude |H_new| = |H| * exp(-beta * A) is not possible directly as A is time domain.
           
           Let's use **Local Search in the Frequency Domain**? No.
           
           Let's go with **Spectral Initialization + Fast Local Search**.
           The key insight from inspiration 2 (score 0.09) was spectral initialization.
           The current program (score 0.0006) uses random/block initialization.
           The difference is likely the quality of the start.
           
           Strategy:
           1. Generate a sequence with a "good" spectral profile.
              A sequence with a flat power spectrum has high sidelobes (like a boxcar).
              A sequence with a "bell-shaped" or "windowed" spectrum might have lower sidelobes?
              Actually, for a binary sequence, the spectrum is constrained.
              
           Let's try **Simulated Annealing on the Spectrum**?
           Or **Genetic Algorithm**?
           
           Let's try a different local search: **Greedy Hill Climbing with Look-ahead**.
           The current SA is slow. Greedy is fast.
           If we can find a good start, Greedy can polish it quickly.
           
           How to find a good start?
           Use the **Grosch-Pratt** method properly:
           1. Start with random h.
           2. Compute H.
           3. Compute A.
           4. The goal is to minimize the peak of A.
           5. We can approximate the gradient of the peak with respect to the phases of H.
           
           Actually, there is a known technique: **Iterative Phase Retrieval with a "Peak Sidelobe" mask**.
           1. Define a target autocorrelation A_target.
              Let A_target[0] = N.
              Let A_target[k] = -epsilon for k != 0? No, that's not feasible.
              
           Let's try a simpler structural change: **Use a different representation**.
           Instead of ±1, use the fact that we can flip signs.
           
           Let's implement **Fast Greedy Local Search** with **Spectral Initialization**.
           
           Initialization:
           Use the inverse FFT of a random magnitude spectrum with a specific shape?
           Or, use the **Mian-Chowla** sequence properties? No, N=400 is too large.
           
           Let's try:
           1. Initialize h with random ±1.
           2. Compute H.
           3. Compute A.
           4. While time:
              - Find the lag k with max A.
              - This lag corresponds to a correlation between h and shifted h.
              - Flipping a bit at position i changes A by a specific amount.
              - Compute the change in A for all possible flips? That's O(N^2) per step if done naively, but can be optimized.
              
           Optimized Greedy:
           Let h be the current sequence.
           Let A be the autocorrelation.
           For each bit i, if we flip h[i], how does A change?
           A_new[k] = A[k] - 2*h[i]*sum(h[i+k]*h[i+k])? No.
           
           Let h' = h with h[i] flipped.
           h'[j] = h[j] if j!=i, -h[i] if j=i.
           A'[k] = sum_j h'[j] h'[j+k].
           A'[k] = sum_{j!=i, j+k!=i} h[j]h[j+k] + h'[i]h'[i+k] + h'[i+k]h'[i] (if terms exist)
           
           This is complex. 
           
           Let's stick to the **Spectral Initialization** hypothesis.
           Inspiration 2 scored 0.09 with spectral init. Current is 0.0006.
           The gap is huge. Spectral init must be better.
           
           How to do spectral init for low autocorrelation?
           We want |H|^2 to be as "flat" as possible? No, flat |H|^2 leads to high sidelobes in A.
           We want A to be low.
           A = IFFT(|H|^2).
           We want IFFT(|H|^2) to have low peaks.
           This is equivalent to making |H|^2 have a "smooth" or "low-pass" character?
           Actually, if |H|^2 is constant, A is a delta function (ideal).
           But for binary sequences, |H|^2 cannot be constant.
           
           Let's try to initialize H with a magnitude that is **as flat as possible** given the constraint that the inverse transform must be binary.
           This is a phase retrieval problem.
           
           Algorithm:
           1. Set |H_target| = constant (or slightly windowed).
           2. Set phase(H) = random.
           3. h = IFFT(H).
           4. h = sign(h) (project to binary).
           5. H = FFT(h).
           6. Replace magnitude of H with |H_target|, keep phase.
           7. Repeat 3-6.
           
           This is the **Grosch-Pratt** method.
           
           Let's implement this.
        """
        # Initialize with random phase and flat magnitude
        rng = np.random.RandomState(42)
        H_mag = np.ones(n_fft)
        # Apply a slight window to avoid high-frequency noise? 
        # Or just flat.
        H_phase = rng.uniform(0, 2*np.pi, n_fft)
        # Ensure Hermitian symmetry for real h
        H_phase[0] = 0
        H_phase[n_fft//2] = 0 if n_fft % 2 == 0 else 0
        for i in range(1, n_fft//2 + 1):
            j = n_fft - i
            if j < n_fft:
                H_phase[j] = -H_phase[i]
                
        H = H_mag * np.exp(1j * H_phase)
        
        best_h = None
        best_val = np.inf
        
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            h = np.fft.ifft(H).real
            h = np.sign(h)
            if np.all(h == 0):
                h = 1.0
            h[h==0] = 1.0 # Handle zeros
            
            # Evaluate
            val = max_autocorr(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
            # Update H
            H_new = np.fft.fft(h, n=n_fft)
            # Keep magnitude of H_mag, take phase from H_new
            H = H_mag * (H_new / np.abs(H_new))
            
        return best_h

    # Run Grosch-Pratt
    h_gp = grosch_pratt(1000)
    best_h = h_gp
    best_val = max_autocorr(h_gp)
    
    # Refine with Fast Greedy Local Search
    # Greedy: Flip the bit that gives the best improvement.
    # To make it fast, we can use a random subset of bits or just iterate.
    # O(N) per step if we can compute delta efficiently.
    # For now, just do simple hill climbing with random restarts if stuck.
    
    def greedy_refine(h, max_iters=5000):
        current_h = h.copy()
        current_val = max_autocorr(current_h)
        best_h = current_h.copy()
        best_val = current_val
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try flipping each bit? O(N) evaluations per step. N=400. 
            # 400 * 5000 = 2M evaluations. 
            # Each eval is FFT (O(N log N)). 2M * 400*log(400) is too slow.
            # We need to evaluate max_autocorr faster or use a heuristic.
            
            # Heuristic: Pick a random bit, flip it, if better, keep.
            # This is stochastic hill climbing.
            idx = np.random.randint(N)
            trial = current_h.copy()
            trial[idx] = -trial[idx]
            trial_val = max_autocorr(trial)
            
            if trial_val < current_val:
                current_h = trial
                current_val = trial_val
                if current_val < best_val:
                    best_val = trial_val
                    best_h = trial.copy()
                improved = True
            
            if not improved and i % 100 == 0:
                # Small random perturbation to escape local minima
                for _ in range(5):
                    idx = np.random.randint(N)
                    current_h[idx] = -current_h[idx]
                current_val = max_autocorr(current_h)
                
        return best_h, best_val

    h_final, val_final = greedy_refine(best_h, max_iters=10000)
    
    return h_final
# EVOLVE-BLOCK-END
