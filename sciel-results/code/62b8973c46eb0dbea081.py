import numpy as np
import time

N = 100

def construct():
    """
    Finds a sequence of step heights h of length N, h_i in [0, 1], sum(h) = N/2,
    minimizing the maximum overlap with shifted 1-h.
    
    Strategy:
    1. Exploit symmetry: The optimal solution likely has a periodic structure.
    We restrict the search to patterns of length P that divide N, repeated N/P times.
    2. Optimization: Use Simulated Annealing on the pattern variables.
       This allows uphill moves to escape local minima, unlike the previous 
       greedy local search.
    3. Efficiency: Compute overlaps using FFT for O(N log N) per evaluation 
       instead of O(N^2).
    """
    
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    def max_overlap_fft(h):
        """
        Computes the maximum overlap C(s) = sum h_i * (1 - h_{i+s}) over all shifts s.
        Note: The problem defines overlap with shifted 1-h.
        Let g = 1 - h.
        Overlap for shift s is sum h_i * g_{i+s}.
        This is the cross-correlation of h and g at lag s.
        Using FFT:
        corr = ifft(fft(h) * conj(fft(g)))
        We need to handle circular shifts correctly.
        """
        # h and g are circular.
        # We want max over s of sum_i h_i * g_{(i+s) % N}
        # This is the circular cross-correlation.
        
        # FFT approach:
        # C(s) = sum_i h_i g_{i+s}
        # Let's compute the full circular correlation.
        
        # Zero-padding is not needed for circular correlation if we use ifft directly 
        # with appropriate scaling, but standard numpy fft/ifft computes circular correlation 
        # if we multiply FFTs.
        
        # C(s) = ifft(FFT(h) * conj(FFT(g))) ? 
        # Let's verify indices.
        # X = FFT(x), Y = FFT(y).
        # ifft(X * conj(Y)) gives sum x_i y_{i-s} (or similar depending on convention).
        # Let's just compute the values directly for small N in the inner loop if FFT overhead is high, 
        # but N=100 is small. FFT is still faster than N^2=10000 ops if we do it once.
        # Actually, for N=100, a direct loop is ~100*100 = 10k ops. FFT is ~100*5 = 500 ops.
        
        # Let's use the direct method for simplicity and correctness first, 
        # but vectorized.
        # shifts = np.arange(1, N)
        # ov = np.array([np.sum(h * np.roll(1-h, s)) for s in shifts])
        # This is N * N operations. 100 * 100 = 10,000.
        # 10,000 float ops is very fast in numpy.
        
        g = 1.0 - h
        max_ov = 0.0
        # We can compute all shifts at once using matrix multiplication or similar?
        # Or just loop. The loop is 99 iterations.
        # Each iteration is a dot product of two length-100 vectors.
        # This is very fast.
        
        # Optimization: 
        # C(s) = sum h_i (1 - h_{i+s}) = sum h_i - sum h_i h_{i+s} = N/2 - sum h_i h_{i+s}
        # Minimizing max C(s) is equivalent to Maximizing min(sum h_i h_{i+s}).
        # Let's compute the autocorrelation of h.
        # A(s) = sum h_i h_{i+s}
        # We want to maximize the minimum A(s) for s=1..N-1.
        
        # Using FFT for Autocorrelation:
        # A(s) = ifft(FFT(h) * conj(FFT(h)))
        # This gives circular autocorrelation.
        
        Fh = np.fft.fft(h)
        # Power spectrum
        S = Fh * np.conj(Fh)
        # Inverse FFT gives circular autocorrelation
        # Note: numpy fft normalization.
        # ifft(fft(x)*conj(fft(x))) gives sum x_i x_{i-s} (lag -s) or similar.
        # Let's check the output.
        acf = np.fft.ifft(S).real
        # acf[0] is sum h^2.
        # acf[k] corresponds to shift k?
        # In numpy, ifft(X * conj(Y)) gives correlation with lag.
        # Specifically, c[k] = sum_j x[j] y[j-k] (circular).
        # So acf[s] = sum h[i] h[i-s] = sum h[i+s] h[i] (by reindexing).
        # So acf[s] is the sum of products for shift s.
        
        # We are interested in s = 1 to N-1.
        # The overlap for shift s is N/2 - acf[s].
        # We want to minimize max(N/2 - acf[s]) => Maximize min(acf[s]).
        
        # The values of interest are acf[1:N].
        min_acf = np.min(acf[1:N])
        # The corresponding max overlap is N/2 - min_acf
        return N / 2.0 - min_acf

    def construct_from_pattern(pattern):
        """
        Takes a pattern of length P and repeats it to length N.
        Ensures sum is N/2.
        """
        P = len(pattern)
        # Repeat pattern
        h = np.tile(pattern, N // P)
        # Adjust for sum constraint
        # Target sum is N/2.
        current_sum = np.sum(h)
        target_sum = N / 2.0
        
        if np.isclose(current_sum, target_sum):
            return h
            
        # Simple adjustment: distribute the difference evenly?
        # This might violate [0,1] bounds.
        # Better: treat it as a projection.
        # For small N, we can just adjust and clip, then fix sum again?
        # Or, during SA, we maintain the sum constraint by doing moves that preserve sum.
        
        # If we are just evaluating a candidate from SA, we assume SA maintains sum.
        # But if we start from a pattern, we need to normalize.
        # Let's normalize by scaling, then clip, then adjust?
        # Actually, it's better to work with the full vector h in SA, 
        # but constrained to be periodic?
        
        # Alternative: Run SA on the full vector h, but penalize non-periodicity?
        # Or: Run SA on the pattern, and ensure the pattern sum is P/2.
        # If sum(pattern) = P/2, then sum(h) = (N/P) * (P/2) = N/2.
        
        # So, we should constrain the pattern to have sum P/2.
        # And values in [0,1].
        
        # The input 'pattern' is already assumed to satisfy these if generated correctly.
        # If not, we fix it.
        if not np.isclose(np.sum(pattern), P/2.0):
            # Scale
            scale = (P/2.0) / np.sum(pattern)
            pattern = pattern * scale
            # Clip
            pattern = np.clip(pattern, 0, 1)
            # Re-normalize (this might not be exact)
            # For the purpose of this function, we assume the caller passes valid patterns.
            # If we are using this for initialization, we need a valid start.
            
        return h

    # --- Simulated Annealing Setup ---
    
    # We will search over patterns of length P.
    # Possible P: 1, 2, 4, 5, 10, 20, 25, 50, 100.
    # P=100 is the full space. P=50 is half, etc.
    # Smaller P means lower dimensionality and potentially better structure.
    # Let's try a few P values.
    
    best_h_global = None
    best_score_global = float('inf')
    
    rng = np.random.default_rng(42)
    
    candidate_periods = [100, 50, 25, 20, 10, 5, 4, 2, 1]
    
    for P in candidate_periods:
        if time.time() > deadline:
            break
            
        # Initialize pattern
        # Start with uniform 0.5
        pattern = np.full(P, 0.5)
        
        # Ensure sum is P/2
        # 0.5 * P = P/2. Good.
        
        # SA Parameters
        T = 0.1  # Initial temperature
        T_min = 1e-6
        alpha = 0.999  # Cooling rate
        # Steps per temperature:
        # We have time budget. Let's estimate steps.
        # Each step: 1 eval (FFT ~ fast) + 1 move.
        # Let's run until time limit or convergence.
        
        current_score = max_overlap_fft(construct_from_pattern(pattern))
        
        # If P is small, the search space is small.
        # If P is large, it's larger.
        
        # To make SA work on the pattern while maintaining sum constraint:
        # Move: Pick i, j. delta = rng.uniform(-eps, eps).
        # pattern[i] += delta, pattern[j] -= delta.
        # Check bounds.
        
        eps = 0.05
        
        while T > T_min:
            if time.time() > deadline:
                break
                
            # Perform several steps per temperature to allow mixing
            steps = 50
            
            for _ in range(steps):
                if time.time() > deadline:
                    break
                    
                i = rng.integers(0, P)
                j = rng.integers(0, P)
                if i == j:
                    continue
                    
                delta = rng.uniform(-eps, eps)
                
                new_pattern = pattern.copy()
                new_pattern[i] += delta
                new_pattern[j] -= delta
                
                # Check bounds
                if new_pattern[i] < 0 or new_pattern[i] > 1 or new_pattern[j] < 0 or new_pattern[j] > 1:
                    continue
                    
                # Evaluate
                new_h = construct_from_pattern(new_pattern)
                new_score = max_overlap_fft(new_h)
                
                delta_score = new_score - current_score
                
                # Accept if better, or with probability exp(-delta_score / T)
                if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
                    pattern = new_pattern
                    current_score = new_score
                    
                    # Update global best
                    if current_score < best_score_global:
                        best_score_global = current_score
                        best_h_global = new_h.copy()
                        
            T *= alpha
            
    # If no valid solution found (shouldn't happen), return uniform
    if best_h_global is None:
        return np.full(N, 0.5)
        
    return best_h_global
