"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral phase retrieval: minimize max|autoconv|/(sum x)^2 by
    flattening the power spectrum via iterative projection.
    """
    np.random.seed(42)
    
    # Initialize
    x = np.random.choice([-1.0, 1.0], size=N)
    
    def compute_score(x):
        X = np.fft.fft(x)
        power = X * np.conj(X)
        c = np.fft.ifft(power).real
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        return np.max(np.abs(c)) / (S * S)
    
    def project_binary(x):
        return np.sign(x)
    
    best_x = x.copy()
    best_score = compute_score(x)
    
    # Iterative spectral projection
    for iteration in range(300):
        X = np.fft.fft(x)
        mag = np.abs(X)
        phase = np.angle(X)
        
        # Target: flatten power spectrum
        # Total power = N^2 (since sum x_i^2 = N)
        total_power = N * N
        dc_power = mag[0] ** 2  # |X(0)|^2 = S^2
        
        # Remaining power for non-DC bins
        remaining = total_power - dc_power
        if remaining < 0:
            remaining = 0
        num_non_dc = N - 1
        target_non_dc = np.sqrt(remaining / num_non_dc) if num_non_dc > 0 else 0
        
        # Construct target magnitude
        target_mag = np.full(N, target_non_dc)
        target_mag[0] = mag[0]  # Keep DC fixed
        
        # Blend with current magnitude
        alpha = 0.3
        new_mag = alpha * target_mag + (1 - alpha) * mag
        
        # Reconstruct and project
        X_new = new_mag * np.exp(1j * phase)
        x_new = np.fft.ifft(X_new).real
        x_new = project_binary(x_new)
        
        # Track best
        score = compute_score(x_new)
        if score < best_score:
            best_score = score
            best_x = x_new.copy()
        
        x = x_new
    
    # Local refinement
    current_obj = compute_score(best_x)
    for i in range(N):
        best_x[i] *= -1
        new_obj = compute_score(best_x)
        if new_obj < current_obj:
            current_obj = new_obj
        else:
            best_x[i] *= -1
    
    return best_x

# EVOLVE-BLOCK-END
