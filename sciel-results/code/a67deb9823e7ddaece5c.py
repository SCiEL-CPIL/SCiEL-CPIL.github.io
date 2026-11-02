"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct signed step heights minimizing the maximum absolute autoconvolution
    divided by squared integral.
    
    Strategy:
    1. Generate a candidate using Spectral Phase Retrieval (Gerchberg-Saxton)
       with a flat magnitude spectrum to minimize autoconvolution sidelobes.
    2. Generate a trivial candidate (constant sequence).
    3. Refine the spectral candidate using local search (bit flipping) to minimize
       the specific objective: max(|c_k|) / (sum(x))^2.
    4. Return the candidate with the lower objective value.
    """
    
    def compute_autoconv(x):
        # Circular autoconvolution via FFT
        X = np.fft.fft(x)
        power = X * np.conj(X)
        c = np.fft.ifft(power).real
        return c

    def objective(x):
        c = compute_autoconv(x)
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        # Max absolute autoconvolution
        # Note: If k=0 is included, c[0]=N is the max for any binary seq.
        # If k=0 is excluded, we look at sidelobes.
        # The task says "maximum absolute autoconvolution", implying all k.
        max_c = np.max(np.abs(c))
        return max_c / (S * S)

    # Candidate 1: Trivial constant sequence
    x_const = np.ones(N)
    obj_const = objective(x_const)
    
    # Candidate 2: Spectral Phase Retrieval (Gerchberg-Saxton)
    # Target: Flat magnitude spectrum |X[k]| = sqrt(N)
    # This corresponds to a delta-like autocorrelation.
    
    np.random.seed(42)
    
    # Initialize random phases
    # Ensure Hermitian symmetry for real output
    phases = np.random.uniform(0, 2*np.pi, N)
    # Make it symmetric
    phases[N//2:] = 0 if N % 2 == 0 else 0 # DC and Nyquist are real
    if N % 2 == 0:
        phases[N//2] = 0
    # Construct X
    X = np.sqrt(N) * np.exp(1j * phases)
    # Enforce Hermitian symmetry: X[k] = conj(X[-k])
    # This is tricky to do perfectly with random phases.
    # Alternative: Start with random x, get X, fix magnitude.
    
    x_guess = np.random.choice([-1.0, 1.0], size=N)
    
    for _ in range(100): # GS Iterations
        # Step 1: Time domain constraint
        x_cont = np.fft.ifft(X).real
        x_bin = np.sign(x_cont)
        # Handle zeros in sign (unlikely for continuous, but possible)
        x_bin[x_bin == 0] = 1.0
        
        # Step 2: Frequency domain constraint
        X_new = np.fft.fft(x_bin)
        # Enforce flat magnitude
        mag = np.abs(X_new)
        # Avoid division by zero
        mag[mag < 1e-10] = 1e-10
        X = X_new / mag * np.sqrt(N)
        
    x_spectral = x_bin
    
    # Refine spectral candidate with local search
    # Objective: min max(|c_k|) / S^2
    # For binary sequences, S changes with flips.
    
    current_x = x_spectral.copy()
    current_obj = objective(current_x)
    
    # Local search: Flip bits that reduce objective
    # O(N^2) per flip if we recompute c.
    # We can update c incrementally, but for N=400, 400*400 = 160k ops per flip.
    # 400 flips * 160k = 64M ops. Fast enough in Python/NumPy?
    # Let's just recompute for simplicity and speed via FFT.
    
    improved = False
    for _ in range(10): # Passes
        for i in range(N):
            current_x[i] *= -1.0
            new_obj = objective(current_x)
            if new_obj < current_obj:
                current_obj = new_obj
                improved = True
            else:
                current_x[i] *= -1.0 # Revert
        
        if not improved:
            break
    
    x_refined = current_x
    obj_refined = current_obj
    
    # Compare candidates
    if obj_refined < obj_const:
        return x_refined
    else:
        return x_const

# EVOLVE-BLOCK-END
