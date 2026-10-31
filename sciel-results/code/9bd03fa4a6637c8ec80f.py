import time
import math
import random

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Precompute random characters for spectral sparsity estimation
# We use a fixed seed for determinism
_rng = random.Random(42)
_NUM_PROJECTIONS = 20
_MAX_N = 20

# We will generate random 'frequency' vectors k in {0, 1, 2}^n
# Since n varies, we generate them on the fly or use a fixed pool for max_n
# and truncate.
# Actually, generating 20 vectors of length 20 is cheap.
# But n is passed to priority.
# We can precompute a pool of random vectors for n up to 20.
# If n > 20, we just use the first n components? No, that's not a valid character for Z_3^n if we truncate.
# Better: Generate random k vectors inside the function if n is small?
# Or, use a deterministic pseudo-random generator based on the vector itself?
# No, we need fixed projections to compare vectors consistently?
# Actually, for spectral sparsity, we want to see if the vector correlates with ANY character.
# If we use random projections, the score is "Max Correlation with Random Characters".
# This approximates the L-infinity norm of the spectrum (scaled).
# High L-infinity norm = Peaked Spectrum = Sparse Spectrum.

# To avoid regenerating random numbers for every call (which might be slow if called millions of times),
# we can precompute a set of random "frequency" vectors for a maximum N.
# If n <= MAX_N, we use the first n components.
# If n > MAX_N, we need to handle it. Assuming n is small (<= 15-20) for these tasks.

_MAX_N_SUPPORTED = 30
_PROJ_VECS = []
for _ in range(_NUM_PROJECTIONS):
    _PROJ_VECS.append(tuple(_rng.randint(0, 2) for _ in range(_MAX_N_SUPPORTED)))

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    if n < 2:
        return 1.0

    # Calculate "Spectral Sparsity" Proxy
    # We approximate the DFT magnitude at a few random frequencies.
    # chi_k(x) = exp(2*pi*i * (k dot x) / 3)
    # Since we only care about the magnitude of the correlation (or real part for speed),
    # and the values are 0,1,2, we can compute the sum of omega^(k_i * x_i).
    
    # To be fast, we precompute the powers of omega for each component?
    # omega = exp(2*pi*i/3)
    # omega^0 = 1
    # omega^1 = -0.5 + 0.866i
    # omega^2 = -0.5 - 0.866i
    
    # We want to maximize the MAXIMUM magnitude of the projection.
    # |sum omega^(k_i x_i)|
    
    # For speed, we can just compute the real part? 
    # The magnitude is sqrt(Re^2 + Im^2).
    # Let's compute Re and Im.
    
    # Precomputed cos and sin values for exponents 0, 1, 2
    # exp(0) = 1 + 0i
    # exp(1) = -0.5 + 0.866025i
    # exp(2) = -0.5 - 0.866025i
    
    c0, s0 = 1.0, 0.0
    c1, s1 = -0.5, 0.8660254037844386
    c2, s2 = -0.5, -0.8660254037844386
    
    # We need to be careful with n > _MAX_N_SUPPORTED
    if n > _MAX_N_SUPPORTED:
        # Fallback to simple balance if n is too large for precomputed projections
        # This shouldn't happen often, but safety first.
        c0_count = sum(1 for v in vector if v == 0)
        c1_count = sum(1 for v in vector if v == 1)
        c2_count = sum(1 for v in vector if v == 2)
        ideal = n / 3.0
        var = (c0_count - ideal)**2 + (c1_count - ideal)**2 + (c2_count - ideal)**2
        return 1.0 - (var / (n * n)) if n > 0 else 1.0

    max_mag_sq = 0.0
    
    # Iterate over random projections
    for k_vec in _PROJ_VECS:
        re_sum = 0.0
        im_sum = 0.0
        
        # Unroll loop for speed? n is small.
        for i in range(n):
            x_i = vector[i]
            k_i = k_vec[i]
            prod = (k_i * x_i) % 3
            
            if prod == 0:
                re_sum += c0
                im_sum += s0
            elif prod == 1:
                re_sum += c1
                im_sum += s1
            else:
                re_sum += c2
                im_sum += s2
        
        mag_sq = re_sum * re_sum + im_sum * im_sum
        if mag_sq > max_mag_sq:
            max_mag_sq = mag_sq
            
    # Normalize max_mag_sq
    # The maximum possible magnitude is n (if all terms align).
    # So max_mag_sq_max = n^2.
    # A random vector will have magnitude ~ sqrt(n).
    # A structured vector (like all zeros) will have magnitude n.
    # We want to MAXIMIZE max_mag_sq (prefer structured/sparse spectrum).
    
    spectral_score = max_mag_sq / (n * n) if n > 0 else 1.0
    
    # Add a small balance term to break ties and ensure global distribution?
    # The directive says "minimizing non-zero coefficients". 
    # High spectral_score = Few non-zero coeffs (Peaked).
    # So we prioritize High spectral_score.
    
    # Let's also check if the vector is "too simple" (e.g., all 0s). 
    # All 0s is a valid vector.
    
    # Return the spectral score.
    # To make it a float in a reasonable range, we just return it.
    # The evaluator uses descending priority.
    # Higher score = Higher Priority.
    
    return float(spectral_score)
