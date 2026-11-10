import time
import numpy as np
import math

def construct():
    """
    Finds a ±1 polynomial with 24 coefficients that minimizes the sup norm on |z|=1.
    Strategy: Local search (hill climbing) with multiple random restarts.
    This is a structural change from the hardcoded lookup approach.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 24
    sqrtN = math.sqrt(N)
    
    # Precompute grid for sup-norm estimation.
    # A dense grid is needed to approximate the sup norm accurately.
    # 8192 points is a good balance between accuracy and speed.
    G = 8192
    angles = 2 * np.pi * np.arange(G) / G
    zs = np.exp(1j * angles)
    
    # Precompute powers of z for fast polynomial evaluation
    # P(z) = sum c_k z^k
    # We can use matrix multiplication: P_vals = C @ V
    # where C is (N, ) and V is (G, N) with V[j, k] = z_j^k
    V = np.ones((G, N), dtype=np.complex128)
    for k in range(1, N):
        V[:, k] = V[:, k-1] * zs
        
    def get_sup_norm(coeffs):
        """Compute sup norm of polynomial with given coefficients."""
        # coeffs: array of +1/-1
        # P_vals: (G, ) complex
        P_vals = coeffs.astype(np.float64) @ V
        return np.max(np.abs(P_vals))
    
    def local_search(initial_coeffs):
        """Perform hill climbing from initial_coeffs."""
        coeffs = initial_coeffs.copy()
        current_norm = get_sup_norm(coeffs)
        
        # Try to improve by flipping single bits
        # To speed up, we can update the polynomial incrementally, 
        # but for N=24 and G=8192, full recomputation is fast enough (~0.1ms)
        # 24 flips * 0.1ms = 2.4ms per iteration. 
        # We can do ~1000 iterations in 2.4s.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for k in range(N):
                # Flip bit k
                coeffs[k] *= -1
                new_norm = get_sup_norm(coeffs)
                if new_norm < current_norm:
                    current_norm = new_norm
                    improved = True
                else:
                    # Revert
                    coeffs[k] *= -1
                    
        return coeffs, current_norm

    best_coeffs = None
    best_score = float('inf')
    
    # Run multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        coeffs = np.random.choice([-1, 1], size=N).astype(np.float64)
        
        # Local search
        res_coeffs, res_norm = local_search(coeffs)
        
        # Refine with a finer grid if we have time and found a good candidate
        # For now, just track the best
        if res_norm < best_score:
            best_score = res_norm
            best_coeffs = res_coeffs.copy()
            
        # Periodically check if we are close to the theoretical optimum
        # Theoretical optimum score is 1.2583758
        # Sup norm = 1.2583758 * sqrt(24) ≈ 6.1639
        if best_score < 1.26 * sqrtN:
            # We are very close, maybe try a more intensive search around this
            pass

    # If we haven't found the exact optimum, the best we found is likely close.
    # Convert back to int list
    if best_coeffs is None:
        # Fallback to a known good sequence if search failed or timed out early
        # This ensures we always return something
        return [1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1]
        
    return [int(c) for c in best_coeffs]
