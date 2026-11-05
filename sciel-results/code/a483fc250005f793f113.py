"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    using frequency-aware spectral filtering in the Chebyshev basis.
    
    Strategy:
    1. Map x to Chebyshev domain [-1, 1].
    2. Fit a degree-5 Chebyshev polynomial using least squares on the full training data.
    3. Compute the energy (squared norm) of each Chebyshev coefficient.
    4. Apply a threshold filter: attenuate coefficients whose energy is a small
       fraction of the total energy, treating them as noise.
    5. Convert the filtered Chebyshev coefficients to the monomial basis using
       exact recurrence relations.
    
    This approach avoids the randomness and computational overhead of bootstrapping,
    instead directly addressing the overfitting of high-frequency noise in the
    subgrid tendency, which is critical for generalization to unseen regimes.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)

    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        c0 = np.mean(y)
        return np.array([c0, 0, 0, 0, 0, 0])
    
    A = x_range / 2.0
    B = (x_max + x_min) / 2.0
    
    N = len(x)
    
    # Map to Chebyshev domain [-1, 1]
    t_all = (x - B) / A
    t_all = np.clip(t_all, -1.0, 1.0)
    
    # Pre-compute Chebyshev design matrix
    T_all = np.zeros((N, 6))
    T_all[:, 0] = 1.0
    if N > 0:
        T_all[:, 1] = t_all
        for k in range(2, 6):
            T_all[:, k] = 2.0 * t_all * T_all[:, k-1] - T_all[:, k-2]
    
    # Fit Chebyshev coefficients
    try:
        c_cheb, _, _, _ = np.linalg.lstsq(T_all, y, rcond=None)
    except Exception:
        return np.zeros(6)
        
    if not np.all(np.isfinite(c_cheb)):
        return np.zeros(6)

    # Frequency-aware spectral filtering
    # Compute energy of each coefficient
    energy = np.abs(c_cheb) ** 2
    total_energy = np.sum(energy)
    
    if total_energy > 1e-15:
        # Normalize energy to get relative importance
        rel_energy = energy / total_energy
        
        # Threshold: coefficients with less than 0.1% of total energy are considered noise
        # We use a soft threshold to avoid hard cutoffs
        threshold = 1e-3
        # Attenuation factor: 1 if above threshold, scaled down if below
        # Using a smoothstep-like function for smoothness
        # Actually, a simple hard threshold with a small epsilon is often more robust
        # Let's try a soft threshold: if rel_energy < threshold, scale by rel_energy/threshold
        # This preserves some signal but heavily penalizes noise
        attenuation = np.ones(6)
        for i in range(6):
            if rel_energy[i] < threshold:
                attenuation[i] = rel_energy[i] / threshold
            
        c_cheb_filtered = c_cheb * attenuation
    else:
        c_cheb_filtered = c_cheb

    # Convert Chebyshev coefficients to monomial basis
    # t = (x - B) / A = alpha * x + beta
    alpha = 1.0 / A
    beta = -B / A
    
    # Chebyshev polynomials in monomial form (t^j)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    D = np.zeros((6, 6))
    D[0, 0] = 1.0
    D[1, 1] = 1.0
    D[2, 2] = 2.0; D[2, 0] = -1.0
    D[3, 3] = 4.0; D[3, 1] = -3.0
    D[4, 4] = 8.0; D[4, 2] = -8.0; D[4, 0] = 1.0
    D[5, 5] = 16.0; D[5, 3] = -20.0; D[5, 1] = 5.0
    
    # Build conversion matrix M such that c_monomial = M @ c_cheb
    # c_monomial[m] = sum_{k=m}^5 c_cheb[k] * sum_{j=m}^k D[k,j] * Binom(j,m) * alpha^m * beta^(j-m)
    
    def binom(n, k):
        if k < 0 or k > n:
            return 0
        res = 1
        for i in range(k):
            res *= (n - i) / (i + 1)
        return res
    
    M = np.zeros((6, 6))
    for k in range(6):
        for j in range(k + 1):
            if D[k, j] == 0:
                continue
            for m in range(j + 1):
                term = D[k, j] * binom(j, m) * (alpha ** m) * (beta ** (j - m))
                M[m, k] += term
    
    c_monomial = M @ c_cheb_filtered
    
    if not np.all(np.isfinite(c_monomial)):
        return np.zeros(6)
        
    return c_monomial
# EVOLVE-BLOCK-END
