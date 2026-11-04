import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        # Use FFT for faster convolution if needed, but np.convolve is fine for N=400
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    best_h = np.ones(N)
    best_val = score(best_h)
    
    # Strategy: Search over parametric "step" functions.
    # Key insight: The objective is scale-invariant. 
    # We can restrict to sum(h) = 1 for optimization, but the formula handles scaling.
    # Actually, let's just generate candidates and score them.
    
    # Family 1: Single Boxcar (positive)
    # h[i] = 1 if i in [start, end) else 0
    # This is equivalent to a window. The autocorrelation is a triangle.
    # Max autocorr is at lag 0, which is sum(h^2).
    # Value = 2 * N * sum(h^2) / (sum(h))^2.
    # For a boxcar of length L: sum(h^2) = L, sum(h) = L. Value = 2*N/L.
    # To minimize value, maximize L. Max L = N -> Value = 2.0 (Seed).
    # So pure positive boxcars are bad. We need negative values to reduce max(z).
    
    # Family 2: Two-level step function
    # h[i] = a if i < k, else b
    # Let's parameterize by k (split point) and ratio r = b/a.
    # We can fix a=1 and vary b.
    # Or better, fix sum(h) = 1 and vary shape.
    # Let's just iterate over k and a few b values.
    
    # More powerful Family: Symmetric "dip" or "peak"
    # h[i] = 1 - delta * window(i)
    # where window is 1 in the middle and 0 at edges.
    # This creates negative lobes in the center.
    
    # Let's try a grid search over "bump" parameters.
    # h(x) = A + B * rect((x - c) / w)
    # Since domain is discrete indices 0..N-1.
    
    # Candidate Generator 1: Rectangular window with DC offset
    # h[i] = dc + amp * (1 if start <= i < end else 0)
    # We want sum(h) != 0.
    # Let's fix sum(h) = 1 by scaling? No, the formula is homogeneous of degree 0.
    # value(h) = value(c*h). So we can normalize h to sum=1.
    
    # Let's generate shapes, normalize them, and score.
    
    candidates = []
    
    # 1. Seed
    candidates.append(np.ones(N))
    
    # 2. "Dip" functions: 1 - amp * rect
    # rect is 1 in [start, end), 0 elsewhere.
    # This makes the center lower (or negative).
    # If amp > 1, the center becomes negative.
    # Autocorrelation of (1 - amp*rect) = 1*1 - 1*amp*rect - amp*rect*1 + amp^2*rect*rect
    # The cross terms might reduce the peak at lag 0?
    # Let's just test it.
    
    # Grid over width w and amplitude amp
    # w from 10% to 90% of N
    # amp from -2 to 2
    
    # To save time, we will evaluate on the fly and keep track of best.
    
    # Let's define a helper to create a rect function
    def make_rect(start, end):
        r = np.zeros(N)
        r[start:end] = 1.0
        return r
    
    # We will search over:
    # 1. Centered rects
    # 2. Off-center rects (less likely to help due to symmetry of max, but possible)
    
    # Optimization: 
    # The max of autocorrelation is likely at lag 0 (sum of squares) or near it.
    # For symmetric positive-definite kernels, max is at 0.
    # If we introduce negative values, the autocorrelation at lag 0 is sum(h^2).
    # The value at other lags might be larger if the negative parts align.
    # However, typically sum(h^2) is the largest term.
    # Minimizing sum(h^2) / (sum(h))^2 is equivalent to minimizing "energy" for fixed "mass".
    # The minimum of sum(h^2) for fixed sum(h)=1 and h real is achieved when h is constant (1/N).
    # Value = 2*N * (N*(1/N)^2) / 1^2 = 2*N * (1/N) = 2.
    # So the constant vector is the GLOBAL MINIMUM for the positive case?
    # Wait. If h can be negative, can we get lower?
    # Let h = 1 + epsilon * v, where sum(v)=0.
    # sum(h) = N.
    # sum(h^2) = sum((1+eps v)^2) = N + 2eps sum(v) + eps^2 sum(v^2) = N + eps^2 sum(v^2).
    # Value = 2*N * (N + eps^2 sum(v^2)) / N^2 = 2 + 2*eps^2*sum(v^2)/N.
    # This is > 2.
    # So any perturbation of the constant vector increases the value IF the max autocorrelation stays at lag 0.
    
    # BUT, the max might NOT be at lag 0!
    # If we create a pattern where the autocorrelation at lag 0 is small, but some other lag is even smaller?
    # No, we want to MINIMIZE the max.
    # If max is at lag 0, we are stuck at 2.0 minimum for sum=const?
    # Let's check the seed: ones(400). sum=400. sum(h^2)=400. max(z)=400.
    # Value = 2*400*400 / 160000 = 2.0.
    
    # Is it possible to get < 2.0?
    # We need max(z) / (sum(h))^2 < 1/N.
    # Let sum(h) = 1. We need max(z) < 1/N.
    # z[0] = sum(h^2). By Cauchy-Schwarz, sum(h^2) >= (sum(h))^2 / N = 1/N.
    # So z[0] >= 1/N.
    # Therefore, max(z) >= z[0] >= 1/N.
    # Therefore, Value >= 2*N * (1/N) / 1 = 2.
    
    # WAIT. The inequality holds for ANY real h.
    # Is the lower bound 2.0?
    # Let's re-read carefully.
    # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # If the lower bound is 2.0, then the seed is optimal?
    # But the task says "seed ... gives 2.0" and "lower is better".
    # And inspirations have scores > 1.0 (meaning value < 2.0).
    # Score = seed/value. If score > 1, value < 2.
    # So it IS possible to get < 2.0.
    
    # Where did my derivation fail?
    # "max(z) >= z[0]"
    # z[0] is the first element of the FULL convolution?
    # np.convolve(h, h) returns length 2N-1.
    # Index 0 corresponds to lag N-1? Or lag 0?
    # np.convolve([1,0], [1,0]) -> [1, 0, 0].
    # np.convolve([0,1], [0,1]) -> [0, 0, 1].
    # The peak is in the MIDDLE.
    # For h = ones(N), z is a triangle peak at index N-1.
    # z[N-1] = N.
    # z[0] = 1 (if h[0]=1, rest 0? No, h is ones).
    # If h = ones(N), z[k] is the number of overlapping 1s.
    # Max is at center.
    
    # My Cauchy-Schwarz argument applied to z[0] (which is h[0]^2 if we define convolution differently? No).
    # In standard autocorrelation R[0] = sum h[i]^2.
    # In np.convolve, the "zero lag" (max overlap) is at index N-1.
    # So max(z) >= z[N-1] = sum(h^2).
    # And sum(h^2) >= (sum(h))^2 / N.
    # So max(z) >= (sum(h))^2 / N.
    # Value >= 2*N * (sum(h)^2 / N) / sum(h)^2 = 2.
    
    # So the lower bound IS 2.0?
    # Why do inspirations have score > 1.0?
    # Score = 2.0 / value.
    # If score = 1.18, value = 2.0 / 1.18 = 1.69.
    # This contradicts the bound.
    
    # Let's check the sign.
    # "maximum is taken over its signed entries, with no absolute value inside np.max"
    # If h has negative values, z entries can be negative.
    # max(z) is the LARGEST signed entry.
    # If we can make the peak (which is positive) SMALLER, while keeping sum(h) constant?
    # No, sum(h^2) is always positive and is the max overlap.
    # Unless... the max overlap is NOT the maximum signed entry?
    # If all entries are positive, max is the peak.
    # If we introduce negative values, the peak sum(h^2) is still positive.
    # Can sum(h^2) be smaller than (sum(h))^2/N? No.
    
    # Is it possible that np.max(z) is NOT at the center?
    # If we design h such that the autocorrelation has a HUGE negative dip, and the positive peak is small?
    # No, sum(h^2) is fixed by the energy.
    
    # Let's re-read the seed score.
    # Seed: ones(400). Value 2.0.
    # Inspiration 1: Score 1.1892. Value = 2.0 / 1.1892 = 1.682.
    
    # How is this possible?
    # Maybe I am misinterpreting "np.convolve(h, h)".
    # For h = [1, 1], np.convolve(h, h) = [1, 2, 1]. Max is 2.
    # Sum(h) = 2. Sum^2 = 4.
    # Value = 2 * 2 * 2 / 4 = 2.
    
    # Let's try h = [1, -1].
    # np.convolve([1, -1], [1, -1]) = [1, 0, -1, 0, 1]?
    # No, length is 2*2-1 = 3.
    # [1*1, 1*(-1)+(-1)*1, (-1)*(-1)] = [1, -2, 1].
    # Max is 1.
    # Sum(h) = 0. Invalid.
    
    # Let's try h = [1, -0.5, 1]. N=3.
    # Sum = 1.5. Sum^2 = 2.25.
    # Conv:
    # 1*1 = 1
    # 1*(-0.5) + (-0.5)*1 = -1
    # 1*1 + (-0.5)*(-0.5) + 1*1 = 1 + 0.25 + 1 = 2.25
    # (-0.5)*1 + 1*(-0.5) = -1
    # 1*1 = 1
    # z = [1, -1, 2.25, -1, 1].
    # Max = 2.25.
    # Value = 2 * 3 * 2.25 / 2.25 = 6.0. Worse.
    
    # Let's try to make the peak smaller.
    # We need sum(h^2) to be small relative to sum(h)^2.
    # This happens when h is "spread out" and positive.
    # But we can't go below 2.0.
    
    # IS THE BOUND ACTUALLY 2.0?
    # If so, why are there better scores?
    # Perhaps `np.max` picks a negative value if the positive peak is small?
    # No, max picks the largest. If there is any positive value, it's larger than negative.
    # Sum(h^2) is always positive.
    
    # UNLESS... the sum is negative?
    # If sum(h) is negative, sum(h)^2 is positive.
    # The bound holds.
    
    # Let's check the inspiration code snippet if available? No.
    # But wait. "Third autocorrelation inequality".
    # Maybe the constant C3 is related to this.
    # If the theoretical lower bound is 2.0, then 2.0 is the best possible.
    # But the seed IS 2.0.
    # Why would the task exist if the seed is optimal?
    # "Choose real, possibly negative step heights".
    
    # Let's look at the formula again.
    # value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    # Maybe `np.convolve` is not what I think?
    # `np.convolve([1, 2], [1, 2])` -> `[1, 4, 4]`? No.
    # `[1*1, 1*2+2*1, 2*2]` -> `[1, 4, 4]`. Max 4.
    # Sum 3. Sum^2 9.
    # Value = 2*2*4/9 = 16/9 = 1.77.
    # Wait. N=2.
    # Value = 1.77 < 2.0.
    
    # Let's re-evaluate the bound for N=2, h=[1,2].
    # sum(h) = 3.
    # sum(h^2) = 1 + 4 = 5.
    # z = [1, 4, 4]. Max = 4.
    # My previous bound said max(z) >= sum(h^2).
    # Here max(z) = 4, sum(h^2) = 5.
    # So max(z) < sum(h^2).
    # Why? Because the "peak" of the convolution is NOT sum(h^2) for non-symmetric h?
    # No, autocorrelation is symmetric.
    # `np.convolve(h, h)` is NOT the standard autocorrelation `sum h[i] h[i+k]`.
    # `np.convolve` does `sum h[i] h[j-i]`.
    # For real h, `np.convolve(h, h)` is the same as `np.convolve(h, h[::-1])`?
    # No.
    # Standard autocorrelation R[k] = sum h[i] h[i+k].
    # `np.correlate(h, h, 'full')` gives this.
    # `np.convolve(h, h, 'full')` gives the convolution.
    # For real h, Conv(h, h) is symmetric?
    # h * h. Yes, convolution is commutative.
    # So z is symmetric.
    # The center term (lag 0 in correlation) corresponds to sum h[i]^2.
    # In `np.convolve`, the index of the center term is N-1.
    # z[N-1] = sum h[i]^2.
    # So max(z) MUST be >= sum(h^2).
    
    # Let's re-calculate h=[1,2].
    # N=2.
    # h = [1, 2].
    # np.convolve([1,2], [1,2])
    # i=0: 1*1 = 1
    # i=1: 1*2 + 2*1 = 4
    # i=2: 2*2 = 4
    # z = [1, 4, 4].
    # Center index N-1 = 1. z[1] = 4.
    # sum(h^2) = 1^2 + 2^2 = 5.
    # z[1] is NOT sum(h^2).
    # Why?
    # Because `np.convolve` shifts.
    # The term at index k is sum_{i+j=k} h[i]h[j].
    # For k=N-1=1: h[0]h[1] + h[1]h[0] = 1*2 + 2*1 = 4.
    # The sum of squares is sum h[i]^2, which corresponds to i=j.
    # i=j implies 2i = k.
    # So the sum of squares is distributed?
    # No.
    # In correlation, R[0] = sum h[i]^2.
    # In convolution, the "peak" is NOT sum of squares.
    # The sum of squares is the sum of the DIAGONAL elements of the outer product.
    # In the convolution sum, the diagonal elements (i=j) only contribute to specific k values.
    # Specifically, h[i]h[i] contributes to k=2i.
    # So sum(h^2) is the sum of z[2i] for all i.
    # It is NOT a single entry in z.
    
    # AH. This is the key.
    # `np.convolve(h, h)` does NOT contain `sum(h^2)` as a single entry.
    # It contains cross terms.
    # The maximum entry of the convolution can be much smaller than sum(h^2) if the energy is spread across many lags.
    # To minimize max(z), we want the autocorrelation to be "flat" or "spread out".
    # This is the opposite of a peak!
    # We want a "flat-top" autocorrelation.
    # This is achieved by "Chirp" sequences or "Golay" sequences or similar.
    # Specifically, we want h to be such that its autocorrelation is close to a delta? No, we want the MAX to be small.
    # If we make the autocorrelation flat, the max is small.
    # Ideally, z[k] = constant for all k.
    # Then max(z) = C.
    # Sum(z) = sum(h)^2.
    # Sum(z) = (2N-1) * C.
    # So C = sum(h)^2 / (2N-1).
    # Value = 2 * N * C / sum(h)^2 = 2 * N / (2N-1).
    # For N=400, Value = 800 / 799 = 1.00125.
    # This is < 2.0.
    # So we need a sequence with "flat" autocorrelation.
    # This is known as a "Low Autocorrelation Sequence" or "Barker Sequence" (though Barker is for low side lobes, here we want low PEAK relative to DC? No, we want low PEAK relative to Sum^2).
    # Actually, we want the max coefficient of the autocorrelation to be small.
    # This is related to "Sidon sets" or "Golay sequences" or "Zadoff-Chu".
    # Zadoff-Chu sequences have ideal autocorrelation (impulse).
    # If z is an impulse, max(z) = sum(h^2).
    # But we established sum(h^2) >= sum(h)^2 / N.
    # So if z is impulse, Value >= 2.
    # We want z to be FLAT.
    # Flat autocorrelation is achieved by random-like sequences?
    # For random h, z[k] is small for k!=0, but z[0] is large?
    # No, in convolution, there is no single "0 lag" peak of sum of squares.
    # For random h, z[k] behaves like a random walk?
    # Let's test a random sequence.
    
    # Strategy: Generate random sequences, or sequences with specific spectral properties.
    # A sequence with flat power spectrum (white noise) has flat autocorrelation?
    # Wiener-Khinchin: Autocorrelation is IFFT of Power Spectrum.
    # If Power Spectrum is flat (white), Autocorrelation is Delta.
    # If Autocorrelation is Flat (rect), Power Spectrum is Sinc.
    # We want Autocorrelation to be Flat.
    # So we want Power Spectrum to be Sinc-like?
    # Or we can just search for random sequences that have low max autocorrelation.
    
    # Let's implement a random search + local refinement.
    # 1. Generate many random real vectors.
    # 2. Score them.
    # 3. Keep the best.
    # 4. Try to refine the best one using gradient descent on the objective?
    # The objective is not differentiable everywhere (abs, max), but we can smooth it or use subgradients.
    # Or just use random restarts and local search (e.g., Nelder-Mead on a subset of parameters).
    
    # Given the time limit, a simple random search with many iterations might work.
    # Or better: Search over "Chirp" signals.
    # h[i] = exp(1j * pi * i^2 / N). Real part?
    # Or just random.
    
    # Let's try random search first.
    
    best_h = np.ones(N)
    best_val = score(best_h)
    
    rng = np.random.RandomState(123)
    
    # Try 1000 random vectors
    for i in range(2000):
        if time.time() > deadline:
            break
        # Random normal
        h = rng.randn(N)
        # Ensure sum is not 0
        if abs(np.sum(h)) < 1e-6:
            continue
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Try structured "Chirp" / "Sine" modulated
    # h[i] = cos(2*pi*i*i/N)
    # h[i] = sin(2*pi*i*i/N)
    # These have good autocorrelation properties.
    
    for freq in [0.1, 0.5, 1.0, 2.0, 5.0]:
        if time.time() > deadline:
            break
        t = np.arange(N)
        # Linear Chirp
        h = np.cos(2 * np.pi * freq * t * t / N)
        if abs(np.sum(h)) > 1e-6:
            val = score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
        h = np.sin(2 * np.pi * freq * t * t / N)
        if abs(np.sum(h)) > 1e-6:
            val = score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()

    # Try "Golay" like sequences (binary +/- 1)
    # Binary sequences have high sum of squares.
    # But maybe the max autocorr is low?
    # For binary, sum(h^2) = N.
    # If max(z) is small, value is small.
    # Random binary:
    for i in range(500):
        if time.time() > deadline:
            break
        h = rng.choice([-1, 1], N).astype(float)
        if abs(np.sum(h)) < 1e-6:
            continue
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    return best_h
# EVOLVE-BLOCK-END
