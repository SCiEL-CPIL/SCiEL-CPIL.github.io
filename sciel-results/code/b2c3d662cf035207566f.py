"""Cap set greedy construction with balanced-digit priority and deadline guard."""
import time
import itertools
from functools import lru_cache
import numpy as np

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 320.0  # safety margin before 330s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.monotonic()
    return time.monotonic() - _start_time

def _remaining():
    return _DEADLINE_SECONDS - _elapsed()

def priority(vector, n):
    """
    Structural change: Use Spectral Peakiness to prioritize 'random-like' vectors.
    Vectors with flat spectra are less likely to be part of small structured subspaces
    that form APs.
    """
    # Convert to numpy array
    v = np.array(vector, dtype=float)
    
    # Compute FFT
    # For n=1, fft is just the value itself.
    fft_v = np.fft.fft(v)
    magnitudes = np.abs(fft_v)
    
    # Compute Spectral Peakiness (Kurtosis-like)
    energy = np.sum(magnitudes ** 2)
    if energy == 0:
        peakiness = 0.0
    else:
        peakiness = np.sum(magnitudes ** 4) / (energy ** 2)
        
    # Hamming Weight Balance
    # Prefer vectors with ~2n/3 non-zero digits (balanced in F_3)
    w_count = np.sum(v != 0)
    expected_nonzero = 2.0 * n / 3.0
    w_penalty = (w_count - expected_nonzero) ** 2
    
    # Combine
    # We want to MINIMIZE peakiness and w_penalty.
    # Priority = - (Peakiness + Weight Penalty)
    val = -(peakiness + 0.1 * w_penalty)
    
    # Tie breaker: Lexicographic
    rank = 0
    for val_in in vector:
        rank = rank * 3 + val_in
    val += -rank / (3 ** n)
        
    return val

# EVOLVE-BLOCK-END

# The evaluator will call priority(vector, n) to get priorities, then sort all vectors
# in descending priority and greedily add them if they keep the set a cap set.
# The score is the size of the resulting cap set.

# We need to make sure the function is fast since it will be called for all 3^n vectors.
# For large n, 3^n can be huge, so we need to be careful.
# However, the problem says vectors of {0,1,2}^n, and the evaluator will call priority
# for each vector. We just need to return a float.

# Let's optimize the priority function for speed.
# Precompute nothing since n varies.

def priority(vector, n):
    """Optimized priority function with balanced counts and positional spread."""
    # Count digits
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    expected = n / 3.0
    var = (c0 - expected) ** 2 + (c1 - expected) ** 2 + (c2 - expected) ** 2
    
    # Positional spread: for each digit value, measure how evenly its positions are distributed
    # A vector with digits spread across coordinates is less likely to form APs
    # We compute the variance of position indices for each digit group
    pos0 = []
    pos1 = []
    pos2 = []
    for i, v in enumerate(vector):
        if v == 0:
            pos0.append(i)
        elif v == 1:
            pos1.append(i)
        else:
            pos2.append(i)
    
    def _pos_spread_penalty(positions, n):
        """Penalty for clustered positions. Returns 0 if positions are evenly spread."""
        if len(positions) <= 1:
            return 0.0
        # Expected spacing if evenly distributed
        expected_spacing = n / len(positions)
        # Compute actual spacings between consecutive positions
        spacings = []
        for k in range(len(positions) - 1):
            spacings.append(positions[k + 1] - positions[k])
        # Variance of spacings relative to expected
        mean_spacing = sum(spacings) / len(spacings)
        if mean_spacing == 0:
            return 0.0
        var_spacing = sum((s - mean_spacing) ** 2 for s in spacings) / len(spacings)
        # Normalize by expected spacing squared
        return var_spacing / (expected_spacing ** 2)
    
    spread_penalty = (_pos_spread_penalty(pos0, n) + 
                      _pos_spread_penalty(pos1, n) + 
                      _pos_spread_penalty(pos2, n)) / 3.0
    
    # Lex rank for tie-breaking
    rank = 0
    for v in vector:
        rank = rank * 3 + v
    
    # Combine: balanced count priority dominates, spread is secondary, lex breaks ties
    # Weight: count balance (1.0), positional spread (0.3), lex (tiny)
    return -var - 0.3 * spread_penalty - rank / (3 ** n)
