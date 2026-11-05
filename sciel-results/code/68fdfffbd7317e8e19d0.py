import numpy as np
import time

def _eval_sup_norm(coeffs, N=4096):
    """
    Evaluate the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling.
    """
    # coeffs is a list of +1/-1
    # z = exp(i * theta)
    thetas = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * thetas)
    
    # P(z_j) = sum_{k=0}^{n-1} c_k z_j^k
    # We can compute this as coeffs @ z_powers
    # z_powers[k, j] = z_j^k
    # To save memory and time, we can compute iteratively or use broadcasting
    # n is small (24), N is 4096.
    # coeffs shape (n,), z shape (N,)
    # We want sum_k c_k * (z^k)
    
    # Efficient way:
    # Start with P = 0
    # For k from 0 to n-1:
    #   P += c_k * z^k
    
    # Or use matrix multiplication:
    # powers = np.power(z[None, :], np.arange(len(coeffs))[:, None])
    # This creates a (24, 4096) array, which is fine.
    
    powers = np.ones((len(coeffs), N), dtype=complex)
    for k in range(1, len(coeffs)):
        powers[k] = powers[k-1] * z
        
    # coeffs is real, powers is complex
    # P_vals = coeffs @ powers
    P_vals = np.dot(coeffs, powers)
    
    return np.max(np.abs(P_vals))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial candidate: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = p[:24]
    
    # Evaluate initial score
    # We use a fast evaluator for search
    # Note: The task evaluator uses golden section refinement. 
    # Our grid max is a lower bound on the true sup norm.
    # If we minimize the grid max, we likely minimize the true sup norm.
    
    best_score = _eval_sup_norm(best_coeffs)
    
    # Local Search: Hill Climbing
    # We will try to flip bits.
    # To avoid getting stuck in a local minimum, we can do multiple restarts.
    # Or, we can use a more robust method like simulated annealing, 
    # but for this small dimension, hill climbing with restarts is often sufficient.
    
    # Let's try a simple hill climbing first.
    # If it doesn't reach the known optimum, we can add restarts.
    
    # Known optimum score is 1.2583758 * sqrt(24) = 6.156... 
    # Wait, the score in the task is max_norm / sqrt(24).
    # So target max_norm = 1.2583758 * sqrt(24) ≈ 6.1563.
    # Let's check the current GRS score.
    
    # Actually, let's just run the search and see how low we can get.
    
    # We will use a list of candidates to try.
    # Start with GRS.
    # Then try random restarts if we have time.
    
    # To make the search faster, we can precompute the powers matrix?
    # No, coeffs change, but z is fixed.
    # We can precompute the powers matrix once.
    
    N = 4096
    thetas = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * thetas)
    
    # Precompute powers for all k up to 23
    powers = np.ones((24, N), dtype=complex)
    for k in range(1, 24):
        powers[k] = powers[k-1] * z
        
    def fast_eval(coeffs):
        # coeffs is a numpy array or list
        return np.max(np.abs(np.dot(coeffs, powers)))
        
    best_coeffs = np.array(best_coeffs, dtype=np.float64)
    best_score = fast_eval(best_coeffs)
    
    # Hill Climbing
    improved = True
    while improved:
        improved = False
        for i in range(24):
            # Flip bit i
            new_coeffs = best_coeffs.copy()
            new_coeffs[i] *= -1
            new_score = fast_eval(new_coeffs)
            if new_score < best_score:
                best_coeffs = new_coeffs
                best_score = new_score
                improved = True
    
    # Check if we have time for restarts
    # The optimum is 1.2583758.
    # Current best_score is the max norm.
    # current_ratio = best_score / np.sqrt(24)
    
    # If we haven't reached the optimum, we might be in a local minimum.
    # Let's try a few random restarts.
    
    rng = np.random.default_rng(42)
    
    # We will keep the best solution found so far.
    # We will run restarts until we hit the deadline or find the optimum.
    
    # Threshold for "good enough"
    # The exhaustive optimum is 1.2583758.
    # Let's aim for that.
    
    # If we are already very close, stop.
    # 1.2583758 * sqrt(24) = 6.15633...
    # If best_score < 6.157, we are likely at the optimum or very close.
    
    while time.time() < deadline:
        # Random restart
        current_coeffs = rng.choice([-1, 1], size=24)
        
        # Local search from this start
        improved = True
        while improved:
            improved = False
            for i in range(24):
                new_coeffs = current_coeffs.copy()
                new_coeffs[i] *= -1
                new_score = fast_eval(new_coeffs)
                if new_score < fast_eval(current_coeffs): # Re-eval current to save time? No, store it.
                    current_coeffs = new_coeffs
                    improved = True
        
        # Check if this restart found a better solution
        final_score = fast_eval(current_coeffs)
        if final_score < best_score:
            best_score = final_score
            best_coeffs = current_coeffs
            # If we found the known optimum, we can break early?
            # The known optimum ratio is 1.2583758.
            # If best_score / sqrt(24) is very close to 1.2583758, we can stop.
            if abs(best_score / np.sqrt(24) - 1.2583758) < 1e-6:
                break

    # Return the best coefficients found
    # The function must return 24 coefficients in {+1, -1}
    return [int(c) for c in best_coeffs]
