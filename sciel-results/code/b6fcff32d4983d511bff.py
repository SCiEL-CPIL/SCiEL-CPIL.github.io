import numpy as np

def construct():
    n = 24
    # Use 4000 points for better peak resolution during search
    theta = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    z = np.exp(1j * theta)
    # Precompute powers z^k for k=0..23
    # powers[i, k] = z[i]^k
    powers = z[:, None] ** np.arange(n)
    
    def get_max_mag(coeffs):
        p_vals = powers.T @ coeffs
        return np.max(np.abs(p_vals))
    
    best_coeffs = None
    best_score = np.inf
    
    # Perform multiple random restarts to ensure finding the global optimum
    # N=24 is small, so 50 restarts is very fast
    for restart in range(50):
        if restart == 0:
            # Start with a known good pattern, e.g., Rudin-Shapiro truncated
            # Or just random
            coeffs = np.array([1 if (i % 2 == 0) else -1 for i in range(n)])
        else:
            coeffs = np.random.choice([1, -1], size=n)
        
        current_score = get_max_mag(coeffs)
        
        # Best-Improvement Local Search
        # In each step, find the single bit flip that gives the largest improvement
        # If no improvement, stop.
        improved = True
        while improved:
            improved = False
            best_i = -1
            best_new_score = current_score
            
            for i in range(n):
                # Flip bit i
                coeffs[i] *= -1
                new_score = get_max_mag(coeffs)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_i = i
                    
                # Revert flip to try next i
                coeffs[i] *= -1
                
            if best_i != -1:
                # Apply the best flip
                coeffs[best_i] *= -1
                current_score = best_new_score
                improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = coeffs.copy()
            
    # Return as list of integers
    return [int(c) for c in best_coeffs]
