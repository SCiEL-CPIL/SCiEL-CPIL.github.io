"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(f):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation 
    excluding the zero-lag term.
    """
    n = len(f)
    # Zero-pad to 2*n to perform linear convolution via FFT
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # conv[0] is the sum of squares (self-correlation at lag 0)
    # We want max |conv[k]| for k > 0
    if n > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0

def _generate_chirp_signal(alpha, beta, gamma, delta, n):
    """
    Generate a signal based on a sum of linear and quadratic phase components.
    f[k] = sign(A * exp(i * (alpha*k^2 + beta*k)) + C * exp(i * (gamma*k^2 + delta*k)))
    This creates a structure with specific spectral properties (chirp-like).
    """
    k = np.arange(n)
    # Complex exponentials
    term1 = np.exp(1j * (alpha * k**2 + beta * k))
    term2 = np.exp(1j * (gamma * k**2 + delta * k))
    
    # Linear combination with real coefficients to create interference
    # We use real and imaginary parts or just the sum of the complex numbers
    # To ensure a real-valued signed step function, we take the sign of the real part
    # of a combined complex signal, or sum of two real sinusoids.
    # Let's use a sum of two real sinusoids with different frequencies/phases.
    # s[k] = A * cos(alpha*k^2 + beta*k) + C * cos(gamma*k^2 + delta*k)
    s = np.cos(alpha * k**2 + beta * k) + np.cos(gamma * k**2 + delta * k)
    
    # Map to {-1, 1}
    # Handle zeros by mapping to 1 (arbitrary choice)
    f = np.sign(s)
    f[f == 0] = 1.0
    return f

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 140s.
    deadline = start_time + 140.0
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Grid search / Random search over the chirp parameters
    # Parameters: alpha, beta, gamma, delta
    # Ranges: alpha, gamma in [0, pi/2], beta, delta in [0, 2*pi]
    # We will do a coarse random search first, then refine the best candidates.
    
    best_f = None
    best_score = float('inf')
    
    # Number of random parameter sets to try
    num_trials = 2000
    
    for i in range(num_trials):
        if time.time() > deadline - 20.0:
            break
        
        # Random parameters
        alpha = rng.uniform(0, np.pi / 2)
        beta = rng.uniform(0, 2 * np.pi)
        gamma = rng.uniform(0, np.pi / 2)
        delta = rng.uniform(0, 2 * np.pi)
        
        f = _generate_chirp_signal(alpha, beta, gamma, delta, N)
        score = _autoconv_max_abs(f)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
            # Store the parameters that generated this best solution
            best_params = (alpha, beta, gamma, delta)
            
        # If we find a very good score, we might want to break early or refine
        # but for now, let's just keep searching.
        
    # Phase 2: Local optimization around the best parameters found
    if best_f is not None:
        # Define a local neighborhood around the best parameters
        # We will perform a fine-grained grid search or gradient-free local search
        # around best_params
        
        alpha, beta, gamma, delta = best_params
        
        # Define step sizes for local refinement
        # Start with a larger step, then reduce
        steps = [0.01, 0.001, 0.0001]
        
        current_params = [alpha, beta, gamma, delta]
        
        for step_size in steps:
            if time.time() > deadline - 5.0:
                break
                
            improved = True
            while improved:
                improved = False
                for dim in range(4):
                    # Try increasing and decreasing the parameter
                    for direction in [-1, 1]:
                        trial_params = current_params.copy()
                        trial_params[dim] += direction * step_size
                        
                        # Ensure parameters are within reasonable bounds
                        # alpha, gamma >= 0
                        if dim in [0, 2] and trial_params[dim] < 0:
                            continue
                        # beta, delta periodic, but let's keep them in [0, 2*pi] for stability
                        if dim in [1, 3]:
                            trial_params[dim] = trial_params[dim] % (2 * np.pi)
                        
                        f = _generate_chirp_signal(trial_params[0], trial_params[1], 
                                                   trial_params[2], trial_params[3], N)
                        score = _autoconv_max_abs(f)
                        
                        if score < best_score:
                            best_score = score
                            best_f = f.copy()
                            current_params = trial_params.copy()
                            improved = True
                            if time.time() > deadline - 5.0:
                                break
                    if time.time() > deadline - 5.0:
                        break
                if time.time() > deadline - 5.0:
                    break

    # If the chirp approach didn't yield a good result, fallback to a simple random init
    # but usually chirps should be better than pure random.
    if best_f is None:
        best_f = rng.choice([-1.0, 1.0], size=N)

    # Final local search on the discrete bits (flipping individual signs)
    # This can help if the continuous parameter optimization gets stuck in a local min
    # that can be escaped by a small discrete change.
    
    if time.time() < deadline - 5.0:
        f = best_f.copy()
        current_score = best_score
        
        # Simple greedy local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Iterate through all positions
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                old_val = f[i]
                f[i] = -old_val
                new_score = _autoconv_max_abs(f)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    f[i] = old_val
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()

    return best_f
# EVOLVE-BLOCK-END
