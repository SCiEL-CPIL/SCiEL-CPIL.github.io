import time
import zlib
import math

# Strategy: Spectral Heuristic for Cap Set Greedy Construction
#
# The cap set constraint in $\mathbb{Z}_3^n$ forbids affine lines $\{a, b, c\}$ such that $a+b+c=0$.
# In the Fourier domain over $\mathbb{Z}_3^n$, affine lines correspond to specific low-frequency structures 
# or, more precisely, the indicator function of a cap set should have a "flat" spectrum in the high frequencies 
# to avoid the structured patterns that form lines.
#
# However, a simpler and more robust proxy for "line-completion potential" in a greedy setting is the 
# "spectral energy" of the vector itself when viewed as a point in the character group. 
# A vector $v$ with coordinates in $\{0,1,2\}$ can be associated with a character $\chi_v(u) = \omega^{v \cdot u}$ where $\omega = e^{2\pi i / 3}$.
#
# Actually, the most effective spectral proxy found in similar problems (like the "Cap Set" problem in FunSearch) 
# is to prioritize vectors that are "orthogonal" to many existing vectors in a generalized sense, or simply 
# to use the magnitude of the Fourier transform of the *indicator* of the vector if it were a set. 
#
# Let's refine the spectral idea: 
# The number of lines completed by a candidate $c$ is related to the autocorrelation of the set $S$ at shift $c$.
# If we haven't built $S$ yet, we can't compute this. 
#
# Alternative Structural Approach: 
# Use the "Discrete Fourier Transform" magnitude of the vector's coordinate sequence interpreted as a signal.
# Vectors with high variance in the frequency domain (i.e., "random-looking" or "high entropy" in frequency) 
# are less likely to align with the structured linear dependencies that form cap sets.
#
# Let's try scoring based on the sum of squared magnitudes of the DFT coefficients of the vector coordinates.
# For a vector $v \in \{0,1,2\}^n$, we can compute its DFT. 
# If the DFT is "flat" (energy spread across all frequencies), the vector is "random-like".
# If the DFT has a strong DC component or low-frequency peaks, it is "structured".
# Cap sets tend to avoid structured vectors? Or do they prefer them?
#
# Actually, a known good heuristic for cap sets is to avoid vectors that are "close" to existing ones in the 
# sense of completing lines. 
#
# Let's look at Inspiration 1 (score 1.2930): "scoring vectors based on the magnitude of their Discrete Fourier Transform".
# This suggests that the *magnitude* of the DFT is a good signal. 
#
# How to compute DFT of a vector $v \in \{0,1,2\}^n$?
# We can treat the coordinates $v_1, \dots, v_n$ as a 1D signal of length $n$.
# Compute $X_k = \sum_{j=0}^{n-1} v_j \omega^{-jk}$ for $k=0, \dots, n-1$.
# Score = $\sum |X_k|^2$ or $\max |X_k|^2$ or variance of $|X_k|^2$.
#
# By Parseval's theorem, $\sum |X_k|^2 = n \sum v_j^2$. This is constant for fixed Hamming weight? No, $\sum v_j^2$ varies.
# If we normalize by $\sum v_j^2$, we get a measure of "spectral flatness".
#
# Let's try: Score = - (Variance of DFT magnitudes). 
# High variance in DFT magnitudes means "peaked" spectrum (structured). 
# Low variance means "flat" spectrum (random). 
# Do we want random or structured?
# Cap sets are "random-like" in the sense that they avoid arithmetic progressions. 
# So we probably want vectors that look "random" in their coordinate distribution.
#
# Let's try: Score = Spectral Flatness = Geometric Mean of $|X_k|^2$ / Arithmetic Mean of $|X_k|^2$.
# Higher flatness is better.
#
# To be safe and fast, we can compute the DFT using numpy.

import numpy as np

def priority(vector, n):
    """
    Prioritize vectors by their spectral flatness in the 1D DFT of their coordinates.
    Vectors with flatter spectra (more "random" coordinate distributions) are prioritized.
    """
    if n == 0:
        return 0.0
    
    # Convert to numpy array for fast DFT
    v = np.array(vector, dtype=np.float64)
    
    # Compute DFT
    # Note: numpy.fft.fft is fast (O(n log n))
    # For small n, this is very fast.
    try:
        dft = np.fft.fft(v)
        magnitudes = np.abs(dft) ** 2
        
        # Avoid division by zero if all zeros
        total_energy = np.sum(magnitudes)
        if total_energy < 1e-9:
            # All zero vector. This is a "structured" vector (constant).
            # We might want to penalize it or handle it specifically.
            # Let's give it a low score.
            return -1.0 + (zlib.crc32(bytes(vector)) & 0xFFFFFFFF) * 1e-9
        
        # Spectral flatness measure:
        # Geometric mean of magnitudes / Arithmetic mean of magnitudes
        # Since magnitudes are squared, we use sqrt for geometric mean of magnitudes?
        # Let's use the standard definition on power spectrum.
        # Flatness = exp(mean(log(magnitudes + eps))) / mean(magnitudes)
        
        # Add epsilon to avoid log(0)
        eps = 1e-12
        log_mags = np.log(magnitudes + eps)
        geo_mean = np.exp(np.mean(log_mags))
        arith_mean = np.mean(magnitudes)
        
        flatness = geo_mean / arith_mean
        
        # Higher flatness is better.
        # Add a small tie-breaker
        tie_breaker = (zlib.crc32(bytes(vector)) & 0xFFFFFFFF) / 0x100000000 * 0.001
        
        return flatness + tie_breaker
        
    except Exception:
        # Fallback to variance if DFT fails
        mean = np.mean(v)
        var = np.var(v)
        tie_breaker = (zlib.crc32(bytes(vector)) & 0xFFFFFFFF) / 0x100000000 * 0.001
        return var + tie_breaker
