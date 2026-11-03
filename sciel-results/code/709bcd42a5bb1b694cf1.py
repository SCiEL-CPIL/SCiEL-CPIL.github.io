"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a signed step function (sequence of +1/-1) that minimizes the
    maximum absolute autoconvolution divided by the squared integral.
    
    Strategy: Use a gradient-based approach on a continuous relaxation.
    We optimize a vector x in [-1, 1]^N to minimize max |autocorr(x)|,
    then threshold to get signs. We use a spectral approach: 
    the autoconvolution is related to the magnitude squared of the DFT.
    Specifically, autocorr(x) = IFFT(|FFT(x)|^2) (up to scaling).
    We want to minimize max |IFFT(|FFT(x)|^2)|.
    
    A good heuristic is to make the DFT magnitude as flat as possible (like a random sequence),
    which minimizes peak sidelobes. We can use a greedy sign-flipping algorithm
    (like the Golay construction or iterative sign optimization).
    
    We'll use an iterative sign-flipping method:
    1. Start with random signs.
    2. Compute autocorrelation.
    3. Find the lag with maximum absolute autocorrelation.
    4. Try flipping individual signs to reduce the peak autocorrelation.
    5. Accept the flip that gives the largest reduction in max |autocorr|.
    6. Repeat until time limit.
    
    Actually, a more efficient approach: use the fact that flipping sign at position i
    changes the autocorrelation by a known amount. We can compute the gradient of the
    objective with respect to each sign and flip the one with the steepest descent.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initialize with random signs
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    def compute_autocorr(x):
        """Compute the full autoconvolution (autocorrelation) of x."""
        # Use FFT for efficiency
        Np = 1
        while Np < 2 * N:
            Np *= 2
        X = np.fft.rfft(x, n=Np)
        ac = np.fft.irfft(X * np.conj(X), n=Np)[:N]
        return ac
    
    def max_abs_autocorr(x):
        ac = compute_autocorr(x)
        return np.max(np.abs(ac))
    
    # Initial score
    current_score = max_abs_autocorr(x)
    best_x = x.copy()
    best_score = current_score
    
    # Iterative sign-flipping
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current autocorrelation
        ac = compute_autocorr(x)
        current_max = np.max(np.abs(ac))
        
        # Find the lag with maximum absolute autocorrelation
        peak_lag = np.argmax(np.abs(ac))
        peak_val = ac[peak_lag]
        
        # For each position i, compute the change in autocorrelation if we flip x[i]
        # Flipping x[i] changes ac by: -2 * x[i] * (x shifted)
        # Specifically, ac_new[k] = ac[k] - 2*x[i]*x[(i+k)%N] for the circular case
        # But our autocorrelation is linear, not circular. Let's handle it properly.
        
        # For linear autocorrelation: ac[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k]
        # Flipping x[j] changes ac[k] by: -2*x[j] * (sum of x[j+k] over valid range if k>=0
        #                                 or x[j-k] over valid range if k<0)
        # This is complex. Let's use a simpler heuristic: try flipping each position
        # and see which one reduces the peak the most. But that's O(N^2) per iteration.
        
        # Alternative: use a steepest descent approach on the continuous relaxation.
        # Let's switch to a different strategy: use the fact that for a random sequence,
        # the autocorrelation is approximately zero off-peak. We can use a greedy
        # algorithm that tries to cancel the peak by flipping signs.
        
        # For each position i, estimate the change in max |ac| if we flip x[i].
        # The change in ac due to flipping x[i] is: delta_ac[k] = -2*x[i]*x[i+k] (for k>=0, i+k<N)
        #                                                        = -2*x[i]*x[i-k] (for k<0, i-k>=0)
        # We want to minimize max |ac + delta_ac|.
        
        # This is still O(N^2). Let's try a different approach: use simulated annealing
        # or just do random flips with acceptance based on improvement.
        
        # Try flipping a random position
        i = np.random.randint(N)
        x[i] = -x[i]
        new_score = max_abs_autocorr(x)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[i] = -x[i]  # Revert
    
    # Threshold to get signs (already signs)
    return best_x
# EVOLVE-BLOCK-END
