"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_autocorr(x):
    """Compute the full autocorrelation of x using FFT."""
    n = len(x)
    # Zero-pad to avoid circular convolution effects
    f = np.fft.fft(x, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return ac

def _max_sidelobe(x):
    """Compute the maximum absolute value of the autocorrelation sidelobes."""
    ac = _compute_autocorr(x)
    # Sidelobes are indices 1 to n-1 (and symmetric -1 to -(n-1))
    # Since ac is real and symmetric for real x, we just check 1..n-1
    return np.max(np.abs(ac[1:]))

def _initialization_spectral():
    """
    Initialize x by inverse FFT of a target magnitude spectrum.
    We aim for a flat spectrum except for DC, which helps in creating
    a sequence with low peak sidelobes (similar to Barker codes logic).
    """
    # Target magnitude: 1 for all frequencies, but boost DC?
    # Actually, for a binary sequence x in {-1, 1}, |X[0]| = N.
    # Let's try to construct a phase such that |X[k]| is constant (except DC).
    # This is equivalent to finding a sequence with flat autocorrelation.
    # We can start with a random phase, set magnitudes, and take IFFT.
    
    # A better heuristic: Start with a Barker-like code if available, or random.
    # Barker codes are optimal for small N. For N=400, no Barker code exists.
    # Let's use the "Golay" or "Mersenne" inspiration if possible, or just random phases.
    
    # Strategy: 
    # 1. Create a target magnitude spectrum |X_target[k]|.
    #    For a perfect low-peak autocorrelation, we want |X[k]|^2 to be flat.
    #    |X[k]|^2 is the Fourier transform of the autocorrelation.
    #    If autocorrelation is delta-like, |X[k]| is flat.
    #    So, set |X[k]| = constant for k != 0.
    #    For k=0, |X[0]| must be N (sum of x_i). If x_i in {-1,1}, sum is even.
    
    # Let's try: |X[k]| = 1 for k != 0, and |X[0]| = N.
    # This is physically impossible to realize exactly with binary x, but we can
    # initialize the phase to minimize the error.
    
    # Simpler initialization: Random binary sequence, but shuffled to break patterns?
    # Or, use the inverse FFT of a flat spectrum with random phases.
    
    # Let's generate a complex spectrum with random phases and target magnitudes.
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, N)
    mags = np.ones(N)
    mags[0] = N  # DC component
    
    X = mags * np.exp(1j * phases)
    x_complex = np.fft.ifft(X)
    
    # Map to {-1, 1} by thresholding the real part or magnitude?
    # Taking the real part and thresholding is standard.
    x = np.sign(x_complex.real)
    # Ensure no zeros
    x[x == 0] = 1
    
    return x

def _local_search(x, time_limit):
    """
    Refine x using greedy bit-flip local search.
    Objective: Minimize max |autocorr_sidelobe|.
    """
    start_time = time.time()
    best_x = x.copy()
    best_score = _max_sidelobe(best_x)
    
    # Precompute autocorrelation for current best
    best_ac = _compute_autocorr(best_x)
    
    # We will iterate through indices and try flipping them.
    # To speed up, we can use the fact that flipping x[i] changes the autocorrelation
    # in a predictable way? 
    # ac_new[k] = ac_old[k] - 2*x[i]*sum_{j} x[j]*x[j+k] ... no, it's not that simple.
    # ac[k] = sum x[i] x[i+k].
    # If we flip x[i] to -x[i]:
    # ac_new[k] = sum_{j != i, j+k != i} x[j]x[j+k] + (-x[i])x[i+k] + x[i](-x[i+k]) + (if k=0, -x[i]^2)
    # This is O(N) to update ac for each flip. With N=400, N^2 = 160,000 operations per full pass.
    # 150s budget. Python loop over 400 indices * 160k ops is too slow.
    # We need vectorized updates or a smarter heuristic.
    
    # Alternative: Calculate the change in the objective function for each flip in parallel?
    # The objective is max |ac[1:N]|.
    # Flipping one bit changes the entire ac vector.
    
    # Let's stick to a simpler, faster heuristic:
    # 1. Calculate ac for current x.
    # 2. For each index i, calculate the new ac if x[i] is flipped.
    #    ac_new = ac - 2 * x[i] * (x * np.roll(x, -k) ... wait)
    #    Let's derive:
    #    ac[k] = sum_{j=0}^{N-1-k} x[j] x[j+k]
    #    If we flip x[i], it affects terms where j=i or j+k=i.
    #    j=i -> term x[i]x[i+k] becomes -x[i]x[i+k]. Change: -2 x[i]x[i+k].
    #    j+k=i -> j = i-k. term x[i-k]x[i] becomes -x[i-k]x[i]. Change: -2 x[i-k]x[i].
    #    So for each k, the change is:
    #    delta_ac[k] = -2 * x[i] * ( x[(i+k)%N] + x[(i-k)%N] )  ... careful with boundaries.
    #    Since we use linear autocorrelation (zero-padded), we must be careful.
    #    
    #    Let's use the FFT-based update if possible, or just recompute if N is small enough?
    #    N=400. FFT of size 800 is fast.
    #    But doing 400 FFTs per pass is 400 * 800 log 800 ~ 3.2M ops.
    #    In Python, 400 calls to np.fft.fft might be ~0.1s per pass?
    #    If we can do ~100 passes, that's 10s. This is feasible.
    
    current_x = x.copy()
    current_ac = _compute_autocorr(current_x)
    current_max_side = np.max(np.abs(current_ac[1:]))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        # Try flipping each bit
        for i in range(N):
            # Check if flipping improves the max sidelobe
            # Compute new ac
            new_x = current_x.copy()
            new_x[i] = -new_x[i]
            
            # Optimization: Only compute full ac if potential improvement?
            # Or just compute it.
            new_ac = _compute_autocorr(new_x)
            new_max_side = np.max(np.abs(new_ac[1:]))
            
            if new_max_side < current_max_side - 1e-9:
                current_x = new_x
                current_ac = new_ac
                current_max_side = new_max_side
                improved = True
                # We could break here to restart the scan, or continue.
                # Continuing allows multiple flips in one pass.
                
    return current_x, current_max_side

def construct():
    """
    Construct the signed step function.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Initialize
    x = _initialization_spectral()
    
    # 2. Local Search
    # Split time budget: 80% for local search, 20% for final checks/other
    time_for_ls = 100  # seconds
    x, score = _local_search(x, time_for_ls)
    
    # 3. If time remains, try a few random restarts or perturbations?
    # For now, return the best found.
    if time.time() > deadline:
        return x
    
    return x

# EVOLVE-BLOCK-END
