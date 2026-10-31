import time
import math
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Precompute powers of omega = exp(2*pi*i/3)
# omega^0 = 1
# omega^1 = -0.5 + i*sqrt(3)/2
# omega^2 = -0.5 - i*sqrt(3)/2
# omega^3 = 1
# We need omega^(k * v) where k is frequency index, v is vector component.
# Since v in {0,1,2}, k*v mod 3 determines the power.
# Powers: 1, omega, omega^2
# We can precompute a lookup table for omega^(k*v) for k in range(n), v in {0,1,2}
# But n is variable. We can compute on the fly or use complex exponentials.
# For speed, we can use the property that the sum is linear.

# Actually, a faster way to compute the DFT magnitude squared:
# |X[k]|^2 = |sum_j omega^(k*v_j)|^2
# = sum_j sum_l omega^(k*(v_j - v_l))
# This is O(n^2).
# Alternative: Compute the DFT for all k? No, we only need the sum of |X[k]|^2 or similar.
# Parseval's theorem: sum_k |X[k]|^2 = 3^n * sum_j |x_j|^2 = 3^n * n. This is constant!
# So we can't use the total energy.
# We need to penalize low frequencies.
# Low frequencies k=0 is the DC component. |X[0]| = n.
# k=1, 2, ..., n-1.
# The "structure" of a vector is related to how much of its spectral energy is in low frequencies.
# A random vector has |X[k]| ~ sqrt(n) for k != 0.
# A structured vector (like a line) has large |X[k]| for some k.
# We want to maximize the "randomness" or minimize the "structure".
# Let's try to minimize the sum of |X[k]|^2 for small k, or maximize the entropy of the spectrum?
# Or simply: Score = - sum_{k=1}^{n} |X[k]|^2 / (n^2) ?
# If the vector is all zeros, X[0]=n, X[k!=0]=0. Sum = 0. Score = 0.
# If the vector is random, X[k] ~ sqrt(n). Sum ~ n * n = n^2. Score ~ -1.
# We want high score for random. So maybe Score = 1 - (Sum_{k=1}^n |X[k]|^2 / n^2)?
# Wait, for random, Sum_{k=1}^n |X[k]|^2 ~ n^2. So Score ~ 0.
# For all zeros, Sum = 0. Score = 1.
# This is backwards. All zeros is highly structured (a line).
# We want to AVOID lines. So we want to AVOID structured vectors.
# So we want LOW score for structured vectors.
# Let's try Score = (Sum_{k=1}^n |X[k]|^2) / n^2.
# Random: ~1.
# All zeros: 0.
# This looks like a good "randomness" metric.

# However, computing the full DFT is O(n^2).
# We can approximate or use a cheaper proxy.
# The sum of |X[k]|^2 for k=1..n-1 is related to the autocorrelation.
# sum_k |X[k]|^2 = n * 3^n.
# sum_{k=0}^{n-1} |X[k]|^2 = n * 3^n.
# |X[0]|^2 = n^2.
# So sum_{k=1}^{n-1} |X[k]|^2 = n * 3^n - n^2. This is constant for a given n!
# Wait, Parseval says sum_{k=0}^{3^n-1} |X[k]|^2 = 3^n * sum_j |x_j|^2 = 3^n * n.
# The DFT is defined over the group Z_3^n. The frequencies k are also in Z_3^n.
# So there are 3^n frequency components.
# My previous reasoning about k=1..n-1 was for the 1D DFT.
# For the n-dimensional DFT over Z_3^n, the frequencies are vectors k in Z_3^n.
# X[k] = sum_{v in Z_3^n} omega^{<k, v>} x_v.
# Here x_v is 1 if v is the vector we are scoring, 0 otherwise?
# No, the vector IS the signal. The signal is a single point in the space?
# No, the "vector" is an element of {0,1,2}^n.
# The DFT of a single point delta_v is a constant function 1.
# This interpretation is wrong.
# The "signal" is the vector v itself, viewed as a function f: {0,1,2}^n -> C?
# No, v is a single point.
# The spectral representation of a SINGLE point is trivial.
# The inspiration suggests "scoring vectors based on the magnitude of their Discrete Fourier Transform".
# This implies the vector is treated as a signal in 1D? Or is it the frequency content of the SET?
# "scoring vectors ... based on ... DFT over Z_3^n"
# If we treat the vector v as a 1D signal of length n, we can compute its 1D DFT.
# Let's assume the inspiration meant the 1D DFT of the sequence of values.
# v = (v_0, ..., v_{n-1}).
# X[k] = sum_{j=0}^{n-1} v_j * omega^(k*j).
# This is O(n^2) to compute for all k.
# Or we can just compute the sum of squared magnitudes of the DFT coefficients.
# sum_{k=0}^{n-1} |X[k]|^2 = n * sum_{j=0}^{n-1} |v_j|^2 = n * sum v_j^2.
# This is also constant for a given set of values!
# So the 1D DFT energy is constant.
# What varies is the SPECTRAL DISTRIBUTION.
# A "smooth" or "structured" sequence has energy concentrated in low frequencies.
# A "random" sequence has energy spread out.
# So we want to MAXIMIZE the entropy of the spectral distribution?
# Or MINIMIZE the low-frequency energy?
# Let's compute the energy in the first few frequency bins (k=1, 2, ..., K).
# Low frequency energy = sum_{k=1}^{K} |X[k]|^2.
# We want this to be LOW relative to the total energy.
# Score = 1 - (LowFreqEnergy / TotalEnergy).

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # Convert to numpy array for speed
    v = np.array(vector, dtype=np.float64)
    
    # Total energy is n * sum(v^2). Since v in {0,1,2}, sum(v^2) is integer.
    # We want to measure the "randomness".
    # Let's compute the DFT for the first few frequencies.
    # K = min(n, 10) ?
    K = min(n, 10)
    
    # Compute X[k] for k=1..K
    # X[k] = sum_{j=0}^{n-1} v_j * exp(2*pi*i*k*j/n)
    # This is a matrix multiplication.
    # Frequencies: k=1..K
    # Time indices: j=0..n-1
    # Matrix M[k, j] = exp(2*pi*i*k*j/n)
    # X = M @ v
    
    # For speed, we can precompute the exponentials if n is fixed, but n varies.
    # n is usually small (<= 10 or so for cap sets? No, cap sets in Z_3^n, n can be larger).
    # If n is large, O(n*K) is fine.
    
    # Let's just compute the low frequency energy.
    # We can use numpy.fft if we pad, but n is small.
    
    # Actually, a simpler proxy for "structure" in Z_3^n is the number of "lines" or "affine subspaces" it belongs to.
    # But we can't check that easily.
    
    # Let's stick to the 1D spectral flatness.
    # Compute DFT of v.
    # X = np.fft.fft(v)
    # LowFreqEnergy = sum(|X[1:K+1]|^2)
    # TotalEnergy = sum(|X|^2)
    # Score = 1 - LowFreqEnergy/TotalEnergy
    
    # However, fft is O(n log n).
    
    # Let's try a different "global" metric: 
    # The variance of the values? No, that's local.
    # The "autocorrelation" at lag 1?
    
    # Let's go with the DFT approach as per inspiration 1.
    # We will compute the sum of squared magnitudes of the DFT coefficients for k=1 to K.
    # To avoid complex numbers, we can compute the real and imaginary parts separately.
    
    # X[k] = sum_j v_j cos(2*pi*k*j/n) + i * sum_j v_j sin(2*pi*k*j/n)
    # |X[k]|^2 = (sum v_j cos)^2 + (sum v_j sin)^2
    
    # Precompute cos and sin tables for k=1..K, j=0..n-1
    # This is O(n*K).
    
    # Optimization: If n is small, just compute it.
    
    # Edge case: n=1
    if n == 1:
        return 0.5 # Neutral
        
    # K = min(n, 5) to keep it fast
    K = min(n, 5)
    
    # Generate angles
    # theta_kj = 2*pi*k*j/n
    # We can use numpy vectorization.
    
    # j = 0..n-1
    j = np.arange(n)
    # k = 1..K
    k = np.arange(1, K+1)
    
    # Matrix of angles: shape (K, n)
    # theta[k-1, j] = 2*pi*k[j]*j / n
    # k is (K,1), j is (1,n)
    theta = 2 * np.pi * np.outer(k, j) / n
    
    C = np.cos(theta) # (K, n)
    S = np.sin(theta) # (K, n)
    
    # v is (n,)
    # Real parts: C @ v -> (K,)
    # Imag parts: S @ v -> (K,)
    
    real_parts = C @ v
    imag_parts = S @ v
    
    # |X[k]|^2 = real^2 + imag^2
    low_freq_energy = np.sum(real_parts**2 + imag_parts**2)
    
    # Total energy = n * sum(v^2)
    total_energy = n * np.sum(v**2)
    
    if total_energy == 0:
        return 0.0 # All zeros, highly structured
        
    ratio = low_freq_energy / total_energy
    
    # We want LOW ratio (random) -> HIGH score.
    # Score = 1 - ratio
    
    # Add a small hash tie-breaker
    h = 0
    for val in vector:
        h = (h * 37 + val) & 0xFFFFFFFF
        
    score = (1.0 - ratio) * 10.0 + (h / 4294967296.0)
    
    return float(score)
