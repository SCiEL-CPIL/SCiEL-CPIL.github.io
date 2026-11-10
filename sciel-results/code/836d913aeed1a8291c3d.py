import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Generate Golay-Rudin-Shapiro coefficients as initial guess
    def grs_coeffs(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    c = np.array(grs_coeffs(24), dtype=np.float64)
    
    # 2. Setup grid for evaluation
    # Use a dense grid to approximate the sup-norm
    M = 4096
    thetas = 2 * np.pi * np.arange(M) / M
    z = np.exp(1j * thetas)
    
    # Precompute powers of z: z^0, z^1, ..., z^23
    # V[i, k] = z_i^k
    # This is a Vandermonde matrix
    # Using np.vander is convenient but might be slow for complex? 
    # Better to use outer product or broadcasting.
    # z[:, None] ** np.arange(24)[None, :]
    powers = np.arange(24)
    V = z[:, None] ** powers[None, :]
    
    # Current values on the grid
    y = V @ c
    current_max = np.max(np.abs(y))
    
    # Helper to calculate max magnitude
    def calc_max(y_vals):
        return np.max(np.abs(y_vals))
        
    # Local Search
    # We try to flip coefficients to reduce the max magnitude.
    # To be more robust, we can use a simple Simulated Annealing or just Hill Climbing with restarts.
    # Given the small size, Hill Climbing from GRS is a strong baseline.
    # If it gets stuck, we can try random flips.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try flipping each coefficient
        for j in range(24):
            # Calculate change if we flip c[j]
            # c[j] is +/- 1. New c[j] is -c[j].
            # Delta y = -2 * c[j] * V[:, j]
            delta_y = -2 * c[j] * V[:, j]
            new_y = y + delta_y
            new_max = calc_max(new_y)
            
            if new_max < current_max - 1e-12:
                # Accept move
                c[j] = -c[j]
                y = new_y
                current_max = new_max
                improved = True
    
    # If we have time left, try some random restarts or perturbations to escape local minima
    # The optimal value is 1.2583758 * sqrt(24) ~ 6.144
    # GRS max is around 6.6 or so?
    # Let's check if we are close to the theoretical lower bound.
    # Theoretical lower bound for flat polynomials is often related to sqrt(N) or slightly higher.
    # For N=24, the optimal is 6.144.
    
    # If we haven't reached the target, we might be in a local minimum.
    # Let's try a few random perturbations.
    best_c = c.copy()
    best_max = current_max
    
    # Simple random restarts
    restarts = 0
    while (time.time() - start_time) < deadline - 5.0 and restarts < 50:
        # Perturb the best solution found so far
        c_try = best_c.copy()
        # Flip 5-10 random coefficients
        n_flips = np.random.randint(5, 11)
        indices = np.random.choice(24, n_flips, replace=False)
        c_try[indices] *= -1
        
        # Run local search from this new point
        y_try = V @ c_try
        max_try = calc_max(y_try)
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 5.0:
            improved = False
            for j in range(24):
                delta_y = -2 * c_try[j] * V[:, j]
                new_y = y_try + delta_y
                new_max = calc_max(new_y)
                if new_max < max_try - 1e-12:
                    c_try[j] = -c_try[j]
                    y_try = new_y
                    max_try = new_max
                    improved = True
        
        if max_try < best_max:
            best_max = max_try
            best_c = c_try.copy()
        
        restarts += 1

    # Convert back to integers
    return [int(x) for x in best_c]
