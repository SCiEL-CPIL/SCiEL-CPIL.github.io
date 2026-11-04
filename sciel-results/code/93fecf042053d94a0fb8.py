"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 400
SEED = 42
np.random.seed(SEED)

# Precompute lag indices for autocorrelation computation
# For a step function on [-1/4, 1/4] with N equal intervals,
# the autocorrelation at lag k is sum_i f[i] * f[i+k] (with appropriate scaling)
# We need to compute max |autocorr(k)| for k = 1..N-1 (or similar)

# The task says "signed step function with 400 equal intervals on [-1/4,1/4]"
# and we want to minimize the autoconvolution upper bound on C3.
# The "upper bound on C3" likely relates to the maximum autocorrelation magnitude.
# For a step function f with N bins, autocorr[k] = (1/N) * sum_{i=0}^{N-1-k} f[i]*f[i+k]
# We want to minimize max_k |autocorr[k]| for k >= 1 (excluding lag 0 which is just ||f||^2).

# Actually, let me think about this more carefully.
# The "third autocorrelation inequality" and "autoconvolution upper bound" suggest:
# C3 = sup |f * f * f| or something similar, and we want to bound it.
# 
# For a step function, the autoconvolution at a point depends on the sum of products.
# The key quantity is likely max_k |autocorr[k]| where autocorr[k] = sum_i f[i]*f[i+k].
#
# Let's define: given f = [f_0, ..., f_{N-1}], 
# autocorr[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k] for k = 0, ..., N-1
# We want to minimize max_{k=1}^{N-1} |autocorr[k]|
# Subject to some constraint on f (e.g., f[i] in [-1, 1] or sum f[i] = 0, etc.)

# The seed program returns np.ones(N), which gives autocorr[k] = N-k for all k.
# max autocorr = N-1 = 399. That's the baseline.
# If we use alternating signs: f = [1,-1,1,-1,...], autocorr[k] = 0 for odd k, 
# and for even k: sum_{i} (-1)^i * (-1)^{i+k} = sum_{i} 1 = N-k. Still large for small k.
# Actually for even k, f[i]*f[i+k] = (-1)^i * (-1)^{i+k} = (-1)^{2i+k} = (-1)^k = 1.
# So autocorr[k] = N-k for even k. Still bad.

# A better approach: use random signs and optimize.
# Or use a structured approach like Barker sequences or m-sequences.

# Let's implement a local search / simulated annealing approach to minimize max autocorrelation.

def compute_autocorrs(f):
    """Compute autocorrelations for lags 1..N-1."""
    N = len(f)
    acs = np.zeros(N)
    for k in range(1, N):
        acs[k] = np.dot(f[:N-k], f[k:])
    return acs

def objective(f):
    """Objective: max absolute autocorrelation for lags 1..N-1."""
    acs = compute_autocorrs(f)
    return np.max(np.abs(acs[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Start with a random signed step function
    # Values in {-1, 1} or continuous? Let's try continuous first, then project.
    # Actually, "signed step function" suggests values can be any real, but signs matter.
    # Let's try values in [-1, 1].
    
    best_f = np.random.uniform(-1, 1, N)
    best_val = objective(best_f)
    
    # Try a few initial structures
    # 1. Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    val_alt = objective(alt)
    if val_alt < best_val:
        best_f = alt.copy()
        best_val = val_alt
    
    # 2. Random signs
    for _ in range(5):
        rs = np.random.choice([-1, 1], N)
        val_rs = objective(rs)
        if val_rs < best_val:
            best_f = rs.astype(float)
            best_val = val_rs
    
    # Local search: flip individual elements or small groups
    # Use simulated annealing style
    T = 1.0
    cooling = 0.9995
    
    elapsed = time.time() - start_time
    while elapsed < deadline:
        # Pick a random position to flip/modify
        i = np.random.randint(N)
        f_new = best_f.copy()
        
        # Try flipping the sign or adjusting the value
        if np.random.rand() < 0.5:
            # Flip sign
            f_new[i] = -f_new[i]
        else:
            # Small random perturbation
            f_new[i] = np.clip(f_new[i] + np.random.normal(0, 0.1), -1, 1)
        
        val_new = objective(f_new)
        
        # Accept if better or with probability exp(-(delta)/T)
        if val_new < best_val:
            best_f = f_new
            best_val = val_new
        else:
            delta = val_new - best_val
            if np.random.rand() < np.exp(-delta / T):
                best_f = f_new
                best_val = val_new
        
        T *= cooling
        elapsed = time.time() - start_time
    
    return best_f

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
