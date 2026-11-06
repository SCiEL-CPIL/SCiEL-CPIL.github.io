import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Generate Golay-Rudin-Shapiro sequence of length 24
    # g[0] = 1
    # g[2n] = g[n]
    # g[2n+1] = -g[n]
    n = 24
    g = [0] * n
    g[0] = 1
    for i in range(1, n):
        if i % 2 == 0:
            g[i] = g[i // 2]
        else:
            g[i] = -g[(i - 1) // 2]
    
    # Convert to float array for NumPy ops
    coeffs = np.array(g, dtype=np.float64)
    
    # Precompute points on unit circle for evaluation
    # Use 2048 points for good resolution
    num_points = 2048
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Function to calculate max |P(z)|
    def calc_max(coeffs):
        # P(z) = sum c_k z^k
        # coeffs[k] is c_k
        # z is vector of points
        # We can compute P(z) for all z efficiently using matrix mult or loop
        # z**k is (z[:, None] ** np.arange(n))
        # But for N=24, simple loop is fast enough or use np.vdot
        # Let's use np.polynomial or just matrix multiplication
        # Z_matrix[i, k] = z[i] ** k
        # P_vals = Z_matrix @ coeffs
        # This creates a 2048x24 matrix. 2048*24*8 bytes ~ 400KB. Very fast.
        powers = np.ones((num_points, n), dtype=np.complex128)
        powers[:, 0] = 1
        for k in range(1, n):
            powers[:, k] = powers[:, k-1] * z
        p_vals = powers @ coeffs
        max_val = np.max(np.abs(p_vals))
        return max_val

    # Refine peak using golden section search on a small interval
    def refine_peak(coeffs, theta_center):
        # Define function to evaluate |P(e^{i theta})|
        def f(theta):
            z_point = np.exp(1j * theta)
            # Compute P(z_point)
            # coeffs[k] * z_point**k
            # Use Horner's method or sum
            p_val = 0.0
            z_power = 1.0 + 0.0j
            for c in coeffs:
                p_val += c * z_power
                z_power *= z_point
            return np.abs(p_val)
        
        # Golden section search in [theta_center - delta, theta_center + delta]
        # delta = 2*pi/num_points
        delta = 2 * np.pi / num_points
        a = theta_center - delta
        b = theta_center + delta
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f(c)
        fd = f(d)
        
        # 20 iterations is plenty for 1e-9 accuracy on small interval
        for _ in range(20):
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
        
        return f((a + b) / 2)

    def calc_max_refined(coeffs):
        max_val = calc_max(coeffs)
        # Find index of max
        # Recalculate to find index
        powers = np.ones((num_points, n), dtype=np.complex128)
        powers[:, 0] = 1
        for k in range(1, n):
            powers[:, k] = powers[:, k-1] * z
        p_vals = powers @ coeffs
        abs_vals = np.abs(p_vals)
        idx = np.argmax(abs_vals)
        theta_max = thetas[idx]
        refined_val = refine_peak(coeffs, theta_max)
        return refined_val

    current_max = calc_max_refined(coeffs)
    best_coeffs = coeffs.copy()
    best_max = current_max
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for k in range(n):
            # Flip bit k
            old_val = coeffs[k]
            coeffs[k] = -coeffs[k]
            
            # Quick check with coarse grid first
            coarse_max = calc_max(coeffs)
            
            # If coarse max is already worse than best_max, revert
            if coarse_max >= best_max * 1.0001: # Allow small tolerance
                coeffs[k] = old_val
                continue
            
            # If coarse max is better or close, refine
            new_max = calc_max_refined(coeffs)
            
            if new_max < best_max:
                best_max = new_max
                best_coeffs = coeffs.copy()
                improved = True
            else:
                coeffs[k] = old_val
        
        # If no improvement in a full pass, try a random restart or simulated annealing step?
        # For N=24, hill climbing from GR-S usually finds the optimum.
        # If stuck, break.
        if not improved:
            # Try a few random restarts if time permits
            if time.time() < deadline - 5.0:
                for _ in range(10):
                    if time.time() >= deadline - 5.0:
                        break
                    random_coeffs = np.random.choice([-1, 1], size=n)
                    # Evaluate random
                    r_max = calc_max_refined(random_coeffs)
                    if r_max < best_max:
                        best_max = r_max
                        best_coeffs = random_coeffs.copy()
                        # Continue local search from this new point
                        coeffs = random_coeffs.copy()
                        improved = True
                        break

    return best_coeffs.astype(int).tolist()
