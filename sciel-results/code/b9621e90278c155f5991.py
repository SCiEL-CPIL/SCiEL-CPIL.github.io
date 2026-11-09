import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    N = 24
    
    # Precompute evaluation grid
    # Use a dense grid for fast sup-norm evaluation
    M = 32768  # 2^15 points, FFT-friendly
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # Precompute z = exp(i*theta)
    z = np.exp(1j * theta)
    # Precompute powers z^k for k=0..N-1
    # powers[k] = z^k, shape (M,)
    powers = np.ones((M,), dtype=complex)
    for k in range(1, N):
        powers = powers * z  # This is wrong, let me redo
    
    # Better: precompute the DFT matrix rows
    # Z[k] = z^k for each k, but we need Z[m, k] = z_m^k
    # Let's use: Z = z[:, None] ** np.arange(N)[None, :]  -- too memory heavy for M=32768, N=24? 32768*24 complex = ~7.8 MB, fine.
    Z = z[:, None] ** np.arange(N)[None, :]
    
    def sup_norm(coeffs):
        """Compute sup norm of P(z) = sum c_k z^k on the unit circle grid."""
        # coeffs shape (N,), Z shape (M, N)
        vals = Z @ coeffs
        return np.max(np.abs(vals))
    
    def refine_peak(coeffs, max_idx):
        """Refine the sup norm near the peak at max_idx using golden section."""
        # The peak is near theta[max_idx]
        # We'll do a local search in a small window
        dtheta = 2 * np.pi / M
        # Search in [theta[max_idx] - 3*dtheta, theta[max_idx] + 3*dtheta]
        lo = theta[max_idx] - 3 * dtheta
        hi = theta[max_idx] + 3 * dtheta
        
        def f(t):
            zz = np.exp(1j * t)
            val = sum(coeffs[k] * (zz ** k) for k in range(N))
            return abs(val)
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(50):
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
        
        return max(fc, fd)
    
    def evaluate(coeffs):
        """Full evaluation: grid max then refine the top few peaks."""
        vals = Z @ coeffs
        abs_vals = np.abs(vals)
        max_idx = np.argmax(abs_vals)
        # Refine the peak
        refined = refine_peak(coeffs, max_idx)
        return refined
    
    # Initial candidate: GRS-like or random
    np.random.seed(42)
    
    def random_candidate():
        return np.array([1 if np.random.rand() < 0.5 else -1 for _ in range(N)])
    
    best_coeffs = random_candidate()
    best_score = sup_norm(best_coeffs)
    
    # Local search with multiple restarts
    # To speed up, use the grid sup_norm (not refined) for inner loop,
    # and only refine when we think we've found a good candidate.
    
    def fast_sup_norm(coeffs):
        """Fast grid-based sup norm (no refinement) for search."""
        vals = Z @ coeffs
        return np.max(np.abs(vals))
    
    # Hill climbing with fast evaluation
    current = best_coeffs.copy()
    current_score = fast_sup_norm(current)
    
    # Multiple passes of hill climbing
    max_passes = 5
    for _pass in range(max_passes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping each bit in random order
            indices = np.random.permutation(N)
            for k in indices:
                if time.time() > deadline:
                    break
                old_val = current[k]
                current[k] = -old_val
                new_score = fast_sup_norm(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[k] = old_val
        
        # After each pass, check if we've found a better refined score
        refined_score = evaluate(current)
        if refined_score < best_score:
            best_score = refined_score
            best_coeffs = current.copy()
        
        # If no improvement in fast score, do a random restart
        if not improved:
            # Random restart
            current = random_candidate()
            current_score = fast_sup_norm(current)
    
    # Final refinement of best candidate
    final_score = evaluate(best_coeffs)
    
    return [int(x) for x in best_coeffs]
