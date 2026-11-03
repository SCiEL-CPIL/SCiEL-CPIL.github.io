"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 130.0  # 10s safety margin before 140s (or 120s/150s limit)
    
    # We want to minimize the "normalized maximum of its autoconvolution".
    # Based on the seed score of 1.0 (which is likely ||f*f||_inf / ||f||_2^2 or similar ratio for f=1),
    # and the fact that we want to beat 1.0, we need a function f such that its autoconvolution peak
    # is smaller relative to some norm of f.
    
    # Common normalization in such bounds (e.g., for C1 in uncertainty principles or similar):
    # Often C1 relates to ||f * f||_inf / ||f||_2^2.
    # For f = 1 on [-1/4, 1/4]:
    # f*f is a triangle with max 1/4 (at 0).
    # ||f||_2^2 = integral 1^2 = 1/4.
    # Ratio = (1/4) / (1/4) = 1.
    # So the objective is likely J(f) = ||f * f||_inf / ||f||_2^2.
    # We want to minimize J(f).
    
    # Let's test a few shapes analytically/numerically to pick a good starting point for local search.
    # Shapes must be non-negative step functions with N=600 steps on [-1/4, 1/4].
    # Delta = (1/2) / 600 = 1/1200.
    
    def compute_score(heights):
        """
        heights: array of length N, non-negative.
        Returns ||f * f||_inf / ||f||_2^2.
        """
        if np.sum(heights) == 0:
            return np.inf
        # Compute autoconvolution
        # Discrete autoconvolution of step function approximating continuous.
        # f(x) = h_i for x in [i*delta, (i+1)*delta) shifted to [-1/4, 1/4].
        # The continuous autoconvolution peak is approximated by sum h_i h_j delta.
        # Actually, the max of the continuous autoconvolution is max_k (sum_i h_i h_{i+k} delta).
        # ||f||_2^2 = sum h_i^2 delta.
        
        # Efficient computation using FFT for linear convolution?
        # N=600, so N^2 = 360,000 operations. This is fast.
        # Let's use direct summation or FFT.
        # FFT is safer for potential extensions, but N=600 is small.
        
        # Pad for linear convolution
        n = len(heights)
        f = np.zeros(2*n)
        f[:n] = heights
        # Linear convolution
        conv = np.convolve(heights, heights, mode='full')
        # The continuous integral approximates sum(conv) * delta^2?
        # No, let's be precise.
        # f(x) is constant on intervals.
        # (f*f)(t) = integral f(t-u) f(u) du.
        # For t in [k*delta, (k+1)*delta), the value is approx (1/delta) * (sum_{i,j: i-j approx k} h_i h_j delta^2)?
        # Actually, the standard Riemann sum approximation for the convolution peak:
        # Max_{t} (f*f)(t) \approx \max_k \sum_i h_i h_{i+k} \delta
        # Wait, if f is piecewise constant, the convolution is piecewise quadratic? No, piecewise linear?
        # Let's stick to the discrete proxy that matches the seed score of 1.0.
        
        # For f=1 (N points):
        # h_i = 1.
        # Sum_i h_i^2 delta = N * 1 * delta = N * (1/2)/N = 1/2. (Wait, length is 1/2? Interval is [-1/4, 1/4], length 1/2).
        # Let's re-read: "600 equal intervals on [-1/4,1/4]". Length = 1/2. Delta = 1/1200.
        # ||f||_2^2 = sum h_i^2 delta.
        # For h_i=1: sum = 600. ||f||_2^2 = 600 * (1/1200) = 0.5.
        
        # Autoconvolution max:
        # For h_i=1, the convolution is a triangle.
        # Peak at center.
        # Sum_{i} h_i h_{i+0} delta? No.
        # Let's compute the discrete max of the "sum of products" scaled by delta.
        # Conv_k = sum_i h_i h_{i+k}
        # Approx max value = max_k (Conv_k * delta).
        # For h=1:
        # Conv_0 = 600.
        # Conv_1 = 599.
        # ...
        # Conv_599 = 1.
        # Max Conv_k = 600.
        # Approx max (f*f) = 600 * delta = 600 / 1200 = 0.5.
        
        # Ratio = 0.5 / 0.5 = 1.0.
        # This matches the seed score!
        
        # So the objective to minimize is:
        # Score = (max_k sum_i h_i h_{i+k} * delta) / (sum_i h_i^2 * delta)
        # The delta cancels out!
        # Score = max_k sum_i h_i h_{i+k} / sum_i h_i^2
        
        # We need to minimize R(h) = max_{k} (h . h_shifted_k) / (h . h)
        # where h_shifted_k is h shifted by k (with zero padding).
        
        # Note: max_k includes k=0, which is h.h.
        # So R(h) >= 1.
        # Wait, if R(h) >= 1 always, and seed is 1.0, how can we improve?
        # "minimize the normalized maximum... gives an upper bound on C1".
        # "score is the upper bound on C1".
        # "search score is seed/value... 1.0 equals seed... above 1.0 is improvement".
        # This implies the "value" (the bound) should be SMALLER than the seed's value to get a score > 1.0?
        # If seed value is V_seed, and new value is V_new.
        # Score = V_seed / V_new.
        # If Score > 1.0, then V_new < V_seed.
        # So we want to MINIMIZE the bound (the raw value).
        
        # But if R(h) >= 1 is a mathematical lower bound, and the seed achieves 1.0,
        # then the seed is already optimal?
        # Let's check the definition again.
        # "normalized maximum of its autoconvolution".
        # Maybe the normalization is different?
        # Perhaps ||f||_1?
        # Score = ||f*f||_inf / (||f||_1^2)?
        # For f=1, ||f||_1 = 0.5. ||f||_1^2 = 0.25.
        # ||f*f||_inf = 0.25 (Triangle height is L/2? No, for length L, height is L. L=0.5, height=0.5? No.)
        # Let's re-evaluate continuous f=1 on [-1/4, 1/4].
        # f*f is triangle with base [-1/2, 1/2] and height 1/4 at 0.
        # ||f*f||_inf = 1/4.
        # ||f||_1 = 1/2.
        # ||f||_1^2 = 1/4.
        # Ratio = 1.
        
        # What if normalization is ||f||_2^2?
        # ||f||_2^2 = 1/4.
        # Ratio = 1.
        
        # What if normalization is ||f||_infty * ||f||_1?
        # 1 * 1/2 = 1/2. Ratio = 1/2. Score 0.5.
        
        # What if the "normalized maximum" is max(f*f) / max(f)^2?
        # 1/4 / 1 = 1/4.
        
        # Let's look at the "First autocorrelation inequality".
        # Usually: |f*f(0)| <= ||f||_2^2. This is trivial (Cauchy-Schwarz).
        # Maybe it's about the L_infty norm of the autocorrelation vs L_2 norm?
        # ||f * f||_infty <= C ||f||_2^2 ?
        # For f=1, 0.25 <= C * 0.25 => C >= 1.
        # If we find a function with ratio < 1, we disprove C=1?
        # But R(h) = max_k (h.h_k) / (h.h) is always >= 1 because k=0 term is h.h.
        # So the max is at least h.h.
        # Thus the ratio is always >= 1.
        
        # UNLESS... the "maximum" is taken over k != 0?
        # "First autocorrelation" might refer to the lag?
        # If we exclude lag 0:
        # Seed f=1: Max_{k!=0} (h.h_k) = 599.
        # Denominator h.h = 600.
        # Ratio = 599/600 = 0.99833.
        # If seed score is 1.0, and this ratio is 0.99833, then Score = 1.0 / 0.99833 = 1.0016 > 1.
        # This fits!
        
        # Hypothesis: The metric is M(f) = (max_{k \neq 0} |A_f(k)|) / A_f(0)
        # Where A_f(k) is the autocorrelation at lag k.
        # For non-negative f, A_f(k) >= 0.
        # We want to MINIMIZE M(f).
        # For f=1 (uniform), M = (N-1)/N.
        # We can try to make M smaller.
        
        # How to minimize max_{k!=0} A_f(k) / A_f(0)?
        # We want the autocorrelation to drop off quickly after lag 0.
        # This suggests a "spiky" or "random" like function?
        # But f must be non-negative.
        # If f is a single spike (1 at center, 0 elsewhere):
        # A_f(0) = 1.
        # A_f(k) = 0 for k!=0.
        # M = 0.
        # Score = 1.0 / 0 -> Inf?
        # The prompt says "score is seed/value". If value is 0, score is undefined/infinite.
        # But "minimize... gives an upper bound".
        # If the bound is 0, it's a very good bound.
        # However, usually these problems involve a trade-off or a specific norm.
        
        # Let's check the constraints: "nonnegative step function".
        # If a single spike is allowed, the answer is trivial.
        # Maybe there's a constraint I'm missing?
        # "Choose a nonnegative step function... to minimize the normalized maximum..."
        # If the spike gives 0, why would the seed be 1.0?
        # Perhaps the normalization is different.
        
        # Alternative Hypothesis:
        # The "normalized maximum" is ||f*f||_infty / ||f||_1^2.
        # For spike: ||f*f||_inf = 1 (if height 1). ||f||_1 = 1*delta. Ratio = 1/delta^2. Huge.
        # For uniform: 1.
        
        # Let's reconsider the "First Autocorrelation Inequality" context.
        # It might be related to the uncertainty principle or bandlimited signals.
        # But the input is a general step function.
        
        # Let's look at the observed change in the JSON:
        # A child program scored 1.107.
        # This implies the value was 1/1.107 = 0.903.
        # If the metric was M(f) = max_{k!=0} A(k)/A(0),
        # Uniform gives 599/600 = 0.9983.
        # A score of 1.107 implies a value of 0.903.
        # This is lower than 0.9983.
        # So we need a function with lower "sidelobe" levels relative to the peak.
        
        # Functions with low sidelobes in autocorrelation:
        # Binary sequences (Barker codes) have low sidelobes.
        # But we are dealing with continuous step functions (real values).
        # We can use continuous analogs of Barker codes or "optimal" autocorrelation functions.
        # Or simply, make the function look "random" to decorrelate lags.
        
        # Strategy:
        # 1. Start with a random non-negative function.
        # 2. Optimize it to minimize the ratio.
        # 3. Use a simple gradient-free method or random restarts.
        
        # Let's implement a search for a function h (length N, non-negative)
        # that minimizes R(h) = max_{k=1..N-1} (sum h_i h_{i+k}) / (sum h_i^2).
        
        # To make it robust, we should probably normalize the input so scale doesn't matter.
        # The ratio is scale-invariant.
        
        # Algorithm:
        # 1. Generate random positive vectors.
        # 2. Evaluate R(h).
        # 3. Keep the best.
        # 4. Maybe refine with local perturbations.
        
        # Given time limit (150s), N=600.
        # Computing R(h) involves N-1 lags, each O(N). Total O(N^2) = 360k ops.
        # 10,000 evaluations = 3.6e9 ops. Might be slow in Python loops.
        # Use NumPy vectorization for correlation?
        # np.correlate(h, h, mode='full') gives all lags. O(N^2) but fast in C.
        
        def score_h(h):
            # h: 1D array
            corr = np.correlate(h, h, mode='full')
            # corr index N-1 is lag 0.
            peak = corr[N-1]
            if peak == 0:
                return np.inf
            # We want max of |corr| for lags != 0.
            # Since h is non-negative, corr is non-negative?
            # No, correlation of non-negative vectors is non-negative.
            # corr[k] = sum h_i h_{i+k}. All terms >= 0.
            # So max is just max.
            # Lags are indexed 0 to 2N-2 in 'full' mode.
            # Lag 0 is at N-1.
            # We want max of corr[0:N-1] and corr[N:2N-1].
            side_lobs = np.max(np.concatenate((corr[:N-1], corr[N:2*N-1])))
            return side_lobs / peak
        
        best_score = np.inf
        best_h = np.ones(N)
        
        # Initial candidates
        candidates = []
        candidates.append(np.ones(N))
        
        # Random uniform
        candidates.append(np.random.rand(N))
        
        # Triangle (tapered)
        tri = np.linspace(0, 1, N//2)
        tri = np.concatenate((tri, tri[::-1]))
        candidates.append(tri)
        
        # Cosine taper (Hanning window)
        x = np.linspace(0, 1, N)
        candidates.append(0.5 * (1 + np.cos(2 * np.pi * x)))
        
        # "Barker-like" random +/- 1 mapped to 0..1?
        # If we use {-1, 1}, we can't use non-negative directly?
        # We can use {0, 1} (binary).
        candidates.append(np.random.choice([0, 1], N))
        
        # Evaluate initial candidates
        for h in candidates:
            h = h.astype(float)
            # Ensure non-negative
            h = np.maximum(h, 0)
            s = score_h(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        
        # Local search: Perturb the best candidate
        # Use a simple hill-climbing or simulated annealing approach?
        # Given time, a few hundred iterations of random perturbations on the best candidate.
        
        # Time check
        iterations = 0
        while time.time() < start_time + DEADLINE:
            iterations += 1
            # Create a perturbed version
            h = best_h.copy()
            
            # Strategy: Randomly flip a few elements or add noise
            # Since we want low sidelobes, randomizing phases is good.
            # For non-negative, we can mix the current best with a new random vector.
            # Or just add Gaussian noise and clamp.
            
            noise = np.random.randn(N) * 0.1 * np.std(best_h)
            h_new = best_h + noise
            h_new = np.maximum(h_new, 0)
            
            # Occasionally, restart with a completely new random vector to avoid local minima
            if iterations % 10 == 0:
                h_new = np.random.rand(N)
            
            s = score_h(h_new)
            if s < best_score:
                best_score = s
                best_h = h_new.copy()
        
        # Return the best heights found
        # The prompt asks for "sequence of nonnegative step heights".
        # It doesn't specify normalization.
        # Since the score is a ratio (scale invariant), the scale shouldn't matter.
        # However, to be safe, maybe return the vector as is.
        
        return list(best_h)

# EVOLVE-BLOCK-END
