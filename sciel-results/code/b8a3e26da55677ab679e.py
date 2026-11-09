import numpy as np
import time

def _evaluate_peak(coeffs):
    """
    Calculate the sup norm of P(z) for |z|=1.
    Uses dense sampling followed by golden-section refinement on the top few peaks.
    """
    n = len(coeffs)
    # Dense sampling
    # 2048 points is usually sufficient for initial detection
    angles = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    z = np.exp(1j * angles)
    
    # Evaluate P(z) = sum c_k z^k
    # We can use np.polynomial.polynomial.polyval or manual dot product
    # coeffs are c_0 ... c_{n-1}
    # P(z) = c_0 + c_1 z + ... + c_{n-1} z^{n-1}
    # Using Horner's rule or FFT?
    # For N=24, direct sum is fine.
    # z_k = exp(i * theta_k)
    # P(z_k) = sum_{j=0}^{23} c_j (exp(i theta_k))^j
    
    # Vectorized:
    # z_powers[k, j] = z_k^j
    # This is 2048 x 24 matrix.
    # coeffs is 24 vector.
    # values = z_powers @ coeffs
    
    # Optimization:
    # z^j = exp(i * j * theta)
    # We can compute this as a matrix multiplication.
    
    # Let's try a simpler loop if numpy overhead is high, but 2048x24 is small.
    # theta = angles
    # j = np.arange(n)
    # exponents = np.outer(theta, j) # 2048 x 24
    # z_powers = np.exp(1j * exponents)
    # values = z_powers @ coeffs
    
    # Alternative: Recurrence
    # P(z) = c_0 + z(c_1 + z(c_2 + ...))
    # This is O(N * M) where M is number of points.
    # 24 * 2048 = 49k operations. Very fast.
    
    vals = np.zeros(2048, dtype=complex)
    # Initialize with c_{n-1}
    # Actually, let's just use polyval if available or manual.
    # np.polynomial.polynomial.polyval expects coefficients in increasing order of power.
    # Our coeffs are c_0, c_1, ... c_{n-1}. This matches.
    vals = np.polynomial.polynomial.polyval(z, coeffs)
    
    mags = np.abs(vals)
    
    # Find top peaks
    # Get indices of top 10 peaks
    top_indices = np.argsort(mags)[-10:]
    
    best_peak = 0.0
    for idx in top_indices:
        theta0 = angles[idx]
        # Golden section search around theta0
        # Range: +/- one sample step
        step = 2 * np.pi / 2048
        a = theta0 - step
        b = theta0 + step
        
        # Golden ratio
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(theta):
            z_val = np.exp(1j * theta)
            # Evaluate P(z_val)
            # Using Horner or polyval
            # polyval is probably slower for single point due to overhead?
            # Let's use a manual loop for single point speed
            val = 0.0 + 0.0j
            # P(z) = sum c_k z^k
            # z^k
            z_power = 1.0 + 0.0j
            for k in range(n):
                val += coeffs[k] * z_power
                z_power *= z_val
            return np.abs(val)
            
        fc = f(c)
        fd = f(d)
        
        for _ in range(50): # 50 iterations for high precision
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
                
        best_peak = max(best_peak, max(fc, fd))
        
    return best_peak

def construct():
    # Initial known good polynomial
    coeffs = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1]
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    current_peak = _evaluate_peak(coeffs)
    best_coeffs = coeffs[:]
    best_peak = current_peak
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(24):
            # Try flipping coefficient i
            new_coeffs = best_coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            
            new_peak = _evaluate_peak(new_coeffs)
            
            if new_peak < best_peak:
                best_peak = new_peak
                best_coeffs = new_coeffs
                improved = True
                # Optional: break and restart if significant improvement? 
                # For now, continue to find local optimum in this neighborhood.
                
    return best_coeffs
