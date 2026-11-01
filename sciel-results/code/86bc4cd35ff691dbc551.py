import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe.
    
    Strategy:
    1. Minimize the L2 norm of the autocorrelation sidelobes using local search.
       This objective is smoother and often guides the search towards sequences 
       with generally low sidelobes.
    2. Polish the result by minimizing the peak sidelobe (L-infinity norm) 
       using a short hill-climbing phase.
    """
    np.random.seed(42)
    
    def compute_acf(f):
        """Compute full autocorrelation using FFT."""
        F = np.fft.fft(f, n=2*N)
        acf = np.fft.ifft(F * np.conj(F)).real
        return acf

    def get_l2_sidelobe(f):
        """Compute L2 norm of non-zero lag autocorrelation."""
        acf = compute_acf(f)
        # Non-zero lags are 1..N-1 and -(N-1)..-1
        # In the 2N array: indices 1..N-1 and N+1..2N-1
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        return np.sum(pos_lags**2) + np.sum(neg_lags**2)

    def get_peak_sidelobe(f):
        """Compute peak absolute value of non-zero lag autocorrelation."""
        acf = compute_acf(f)
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        return np.max(np.abs(np.concatenate([pos_lags, neg_lags])))

    def local_search_l2(f, max_iters=2000):
        """
        Local search to minimize L2 sidelobe energy.
        Tries flipping bits that reduce the L2 norm.
        """
        current_l2 = get_l2_sidelobe(f)
        for _ in range(max_iters):
            improved = False
            # Try all positions? No, too slow. Sample random positions.
            # Or use a steepest descent approach with random sampling.
            indices = np.random.permutation(N)
            for i in indices:
                f[i] = -f[i]
                new_l2 = get_l2_sidelobe(f)
                if new_l2 < current_l2:
                    current_l2 = new_l2
                    improved = True
                else:
                    f[i] = -f[i]
            if not improved:
                break
        return f, current_l2

    def local_search_peak(f, max_iters=500):
        """
        Local search to minimize peak sidelobe.
        """
        current_peak = get_peak_sidelobe(f)
        for _ in range(max_iters):
            improved = False
            indices = np.random.permutation(N)
            for i in indices:
                f[i] = -f[i]
                new_peak = get_peak_sidelobe(f)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    f[i] = -f[i]
            if not improved:
                break
        return f, current_peak

    best_peak = float('inf')
    best_f = None
    
    start_time = time.time()
    time_limit = 100.0 # Leave buffer for overhead
    
    # Perform multiple restarts to find a good basin
    num_restarts = 0
    while (time.time() - start_time) < time_limit:
        # Initialize with random signs
        f = np.random.choice([-1, 1], N)
        
        # Phase 1: Minimize L2 norm
        f, l2_val = local_search_l2(f, max_iters=1000)
        
        # Phase 2: Polish with Peak minimization
        f, peak_val = local_search_peak(f, max_iters=200)
        
        if peak_val < best_peak:
            best_peak = peak_val
            best_f = f.copy()
        
        num_restarts += 1
        # If we found a very good solution, we might stop early or continue
        # For N=400, good peaks are often around 10-20? 
        # Let's just keep restarting until time is up.
        
    return best_f
# EVOLVE-BLOCK-END
