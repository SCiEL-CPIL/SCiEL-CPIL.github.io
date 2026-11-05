import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Initial guess: Rudin-Shapiro sequence
    # P_0 = 1, Q_0 = 1
    # P_{n+1} = P_n Q_n, Q_{n+1} = P_n (-Q_n)
    # Actually, the standard recursion is:
    # P_0(z) = 1
    # Q_0(z) = 1
    # P_{n+1}(z) = P_n(z) Q_n(z)
    # Q_{n+1}(z) = P_n(z) (-Q_n(z))
    # This gives length 2^{n+1}.
    # We need length 24.
    # Let's just generate a long RS sequence and truncate.
    def rudin_shapiro(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    coeffs = rudin_shapiro(24)
    
    # Precompute DFT basis for fast evaluation
    # M sample points
    M = 4096
    thetas = 2 * np.pi * np.arange(M) / M
    # Z[k, j] = exp(i * k * theta_j)
    # Z = np.exp(1j * np.outer(np.arange(24), thetas))
    # This is 24 x 4096 complex matrix.
    Z = np.exp(1j * np.outer(np.arange(24), thetas))
    
    def get_sup_norm(c):
        # c is a numpy array of +1/-1
        # P_j = sum c_k Z[k, j]
        P = c @ Z
        return np.max(np.abs(P))
    
    best_norm = get_sup_norm(np.array(coeffs))
    best_coeffs = np.array(coeffs)
    
    # Local search
    # We will try to improve by flipping bits.
    # To speed up, we can precompute the contribution of each bit.
    # But since the change is linear, we can just evaluate the new norm.
    
    # We'll do a simple hill climbing:
    # 1. Try all single flips.
    # 2. Pick the best one.
    # 3. Repeat until no improvement.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_norm = get_sup_norm(best_coeffs)
        
        # Try all single flips
        best_flip = -1
        best_new_norm = current_norm
        best_new_coeffs = None
        
        for k in range(24):
            # Flip bit k
            new_coeffs = best_coeffs.copy()
            new_coeffs[k] *= -1
            new_norm = get_sup_norm(new_coeffs)
            if new_norm < best_new_norm:
                best_new_norm = new_norm
                best_flip = k
                best_new_coeffs = new_coeffs
        
        if best_flip != -1:
            best_coeffs = best_new_coeffs
            best_norm = best_new_norm
            improved = True
        else:
            # No single flip improves. Try double flips?
            # This is O(24^2) = 576 evaluations.
            # Each evaluation is O(M) = 4096.
            # Total ~ 2.4M operations. Fast.
            best_double_flip = (-1, -1)
            best_double_norm = current_norm
            best_double_coeffs = None
            
            for k1 in range(24):
                for k2 in range(k1 + 1, 24):
                    new_coeffs = best_coeffs.copy()
                    new_coeffs[k1] *= -1
                    new_coeffs[k2] *= -1
                    new_norm = get_sup_norm(new_coeffs)
                    if new_norm < best_double_norm:
                        best_double_norm = new_norm
                        best_double_flip = (k1, k2)
                        best_double_coeffs = new_coeffs
            
            if best_double_flip != (-1, -1):
                best_coeffs = best_double_coeffs
                best_norm = best_double_norm
                improved = True
    
    # Convert back to list of ints
    return [int(c) for c in best_coeffs]
