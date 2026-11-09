import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_max_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _smoothed_objective(params):
    """
    Objective function for continuous optimization.
    params: [freq, phase]
    Returns a smoothed approximation of the max autocorrelation.
    """
    freq, phase = params
    # Generate continuous waveform
    x = np.linspace(0, 2 * np.pi * freq, N) + phase
    # Convert to signs (-1 or 1)
    h = np.sign(np.sin(x))
    # Handle zeros in sign (rare but possible)
    h[h == 0] = 1.0
    
    # Compute autocorrelation
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    
    # Add a small smoothing term to encourage flatness? No, just return the metric.
    # The "smoothing" comes from the fact that small changes in freq/phase 
    # cause small changes in the sign pattern, but the objective itself is non-smooth.
    # However, by restricting the search space to this 2D manifold, we avoid the 
    # ruggedness of the 400D discrete space.
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Strategy: Search over (frequency, phase) space for sinusoidal sign patterns.
    # This is a structural change from discrete bit-flipping to continuous parameter search.
    # We expect low frequencies to perform well for minimizing autocorrelation peaks.
    
    num_frequencies = 50
    num_phases = 20
    
    # Pre-allocate for speed if needed, but simple loops are fine for 1000 points.
    
    # Phase 1: Coarse Grid Search
    for i in range(num_frequencies):
        if time.time() > deadline - 30.0:
            break
        
        # Frequency range: 0.1 to 20.0 cycles
        freq = 0.1 + (i / (num_frequencies - 1)) * 19.9
        
        for j in range(num_phases):
            if time.time() > deadline - 30.0:
                break
            
            phase = (j / num_phases) * 2 * np.pi
            
            params = np.array([freq, phase])
            score = _smoothed_objective(params)
            
            if score < best_score:
                best_score = score
                # Reconstruct h for best
                x = np.linspace(0, 2 * np.pi * freq, N) + phase
                h_candidate = np.sign(np.sin(x))
                h_candidate[h_candidate == 0] = 1.0
                best_h = h_candidate
    
    if best_h is None:
        # Fallback if grid search failed
        best_h = np.ones(N)
        best_score = _compute_max_metric(best_h)
        return best_h

    # Phase 2: Local Refinement using L-BFGS-B around the best grid point
    # We need to optimize over continuous (freq, phase) space.
    
    # Since the objective is non-smooth (sign function), standard gradient methods might struggle.
    # However, L-BFGS-B can still navigate if we treat the sign pattern as a "step" function 
    # and rely on the optimizer to find a local minimum in the parameter space that corresponds 
    # to a stable sign pattern.
    
    # Alternatively, we can do a finer grid search around the best point.
    # Let's try a fine grid search first as it's more robust to non-smoothness.
    
    best_freq, best_phase = None, None
    # We didn't store the exact best freq/phase in the loop above cleanly, 
    # let's re-run the best few or store them.
    # Let's modify Phase 1 to store the best parameters.
    
    # Actually, let's just do a fine grid search around the best score found.
    # We need to know which freq/phase gave the best score.
    
    # Let's re-do Phase 1 slightly differently to track parameters.
    best_params = None
    
    for i in range(num_frequencies):
        if time.time() > deadline - 20.0:
            break
        
        freq = 0.1 + (i / (num_frequencies - 1)) * 19.9
        
        for j in range(num_phases):
            if time.time() > deadline - 20.0:
                break
            
            phase = (j / num_phases) * 2 * np.pi
            params = np.array([freq, phase])
            score = _smoothed_objective(params)
            
            if score < best_score:
                best_score = score
                best_params = params
                x = np.linspace(0, 2 * np.pi * freq, N) + phase
                h_candidate = np.sign(np.sin(x))
                h_candidate[h_candidate == 0] = 1.0
                best_h = h_candidate

    if best_params is None:
        return best_h

    # Fine Grid Search around best_params
    freq_center, phase_center = best_params
    freq_range = 0.5
    phase_range = np.pi / 10
    
    fine_freqs = np.linspace(freq_center - freq_range, freq_center + freq_range, 20)
    fine_phases = np.linspace(phase_center - phase_range, phase_center + phase_range, 20)
    
    for freq in fine_freqs:
        if time.time() > deadline - 10.0:
            break
        for phase in fine_phases:
            if time.time() > deadline - 10.0:
                break
            params = np.array([freq, phase])
            score = _smoothed_objective(params)
            
            if score < best_score:
                best_score = score
                best_params = params
                x = np.linspace(0, 2 * np.pi * freq, N) + phase
                h_candidate = np.sign(np.sin(x))
                h_candidate[h_candidate == 0] = 1.0
                best_h = h_candidate

    # Phase 3: Local Bit Flipping Refinement
    # Now that we have a good candidate from the sinusoidal search, 
    # we can try to improve it further with local bit flips.
    # This combines the structural insight (sinusoidal init) with discrete refinement.
    
    h_current = best_h.copy()
    current_score = best_score
    
    # Use incremental update for speed
    H = np.fft.fft(h_current)
    R = np.fft.ifft(np.abs(H) ** 2).real
    current_max = np.max(np.abs(R[1:]))
    
    # Simple local search with limited steps
    max_steps = 5000
    for step in range(max_steps):
        if time.time() > deadline - 2.0:
            break
        
        idx = np.random.randint(0, N)
        
        # Calculate delta R
        delta_R = -2.0 * h_current[idx] * (np.roll(h_current, -idx) + np.roll(h_current, idx))
        R_new = R + delta_R
        new_max = np.max(np.abs(R_new[1:]))
        
        if new_max < current_max:
            h_current[idx] *= -1
            R = R_new
            current_max = new_max
            best_score = current_max
            best_h = h_current.copy()
        else:
            # Small probability of accepting worse moves to escape local minima?
            # For now, keep it strict to ensure we don't degrade the good sinusoidal solution.
            pass

    return best_h
