"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal.windows import flattop

def construct():
    """
    Construct signed step heights using spectral phase retrieval with a fixed flat-top magnitude spectrum.
    
    Strategy:
    1. Define a target magnitude spectrum M_k as a flat-top window (narrow to maximize S).
    2. Initialize random phases.
    3. Iterate Gerchberg-Saxton:
       - Inverse FFT of M_k * exp(i * phase) to get time domain x.
       - Project x to binary {-1, 1}.
       - Forward FFT to get new phases.
    4. Return the resulting binary sequence.
    """
    np.random.seed(42)
    
    # Define the flat-top window in frequency domain
    # Use a narrow window to concentrate energy at DC, maximizing S
    width = max(1, N // 10) # 10% of the spectrum
    # Create a flat-top window of this width
    w = flattop(width)
    
    # Pad with zeros to length N, centered
    M = np.zeros(N)
    start = (N - width) // 2
    end = start + width
    M[start:end] = w
    
    # Ensure M is real and non-negative
    M = np.abs(M)
    
    # Scale M such that the energy matches a binary sequence
    # For binary x, sum(|X_k|^2) = N^2
    energy = np.sum(M**2)
    if energy > 0:
        M *= np.sqrt(N**2 / energy)
    else:
        # Fallback to delta if window is zero (should not happen)
        M = np.zeros(N)
        M[0] = N
        
    # Initialize random phases
    phi = np.random.uniform(0, 2 * np.pi, size=N)
    # Force symmetry for real x: M[0] is real, M[k] = M[N-k]
    # We will handle this by ensuring the reconstructed x is real
    
    x = np.ones(N) # Start with all 1s
    
    for iteration in range(100):
        # Current X from x
        X = np.fft.fft(x)
        
        # Update phases: X_new = M * exp(i * angle(X))
        # Note: M is fixed
        phase = np.angle(X)
        X_new = M * np.exp(1j * phase)
        
        # Enforce Hermitian symmetry for real x? 
        # M is symmetric if w is symmetric. flattop is symmetric.
        # So X_new should be Hermitian if phase is.
        # angle(X) from real x is Hermitian (anti-symmetric imaginary part? no, angle is odd?)
        # For real x, X[k] = conj(X[N-k]).
        # angle(X[k]) = -angle(X[N-k]).
        # M[k] = M[N-k].
        # So X_new[k] = M[k] exp(i phi[k]).
        # X_new[N-k] = M[N-k] exp(i phi[N-k]) = M[k] exp(-i phi[k]) = conj(X_new[k]).
        # So X_new is Hermitian. ifft will be real.
        
        x_new = np.fft.ifft(X_new).real
        
        # Project to binary
        x = np.sign(x_new)
        
        # If x is all 0s (unlikely), break or handle
        if np.all(x == 0):
             x = np.ones(N)
    
    return x

# EVOLVE-BLOCK-END
