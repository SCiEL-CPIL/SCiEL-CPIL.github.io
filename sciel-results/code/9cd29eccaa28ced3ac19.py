import numpy as np
import time
import scipy.optimize as opt

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe.
    Uses a continuous relaxation approach: minimize L2 norm of sidelobes 
    in continuous domain, round to {-1, 1}, then refine with discrete local search.
    """
    
    def get_sidelobes(f):
        """
        Computes the non-zero lag autocorrelation sidelobes.
        Returns a vector of sidelobes (length 2*N-2).
        """
        # Zero-pad to 2*N to avoid circular convolution artifacts
        F = np.fft.fft(f, n=2*N)
        acf = np.fft.ifft(F * np.conj(F)).real
        
        # Extract non-zero lags
        # acf[0] is lag 0
        # acf[1:N] are lags 1..N-1
        # acf[N+1:2*N] are lags -(N-1)..-1
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        
        return np.concatenate([pos_lags, neg_lags])

    def cost_continuous(f_cont):
        """
        Cost function for continuous optimization.
        Uses sum of squares of sidelobes (L2 norm squared) as a smooth proxy for peak.
        Minimizing L2 often helps reduce the peak as well.
        """
        sidelobes = get_sidelobes(f_cont)
        return np.sum(sidelobes ** 2)

    def cost_discrete_peak(f):
        """
        Cost function for discrete refinement.
        Peak absolute value of sidelobes.
        """
        sidelobes = get_sidelobes(f)
        return np.max(np.abs(sidelobes))

    # 1. Continuous Relaxation
    # Initialize with random continuous values in [-1, 1]
    np.random.seed(42)
    f_cont = np.random.uniform(-1, 1, N)
    
    # Use L-BFGS-B for optimization
    # Bounds ensure values stay in [-1, 1]
    bounds = [(-1.0, 1.0)] * N
    
    # Run optimization
    # maxiter and ftol help control time
    res = opt.minimize(cost_continuous, f_cont, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-9, 'maxfun': 5000})
    
    f_opt_cont = res.x
    
    # 2. Rounding
    # Round to nearest integer in {-1, 1}
    f_discrete = np.sign(f_opt_cont)
    # Handle zeros (if any) by mapping to 1 or -1 randomly or based on neighbor
    # np.sign(0) is 0, so replace 0s with 1s (or -1s)
    f_discrete[f_discrete == 0] = 1.0
    
    # 3. Discrete Local Search Refinement
    # The continuous optimum rounded might not be locally optimal in the discrete space.
    # Perform a simple hill-climbing / local search to improve the peak.
    current_peak = cost_discrete_peak(f_discrete)
    best_f = f_discrete.copy()
    best_peak = current_peak
    
    start_time = time.time()
    max_time = 100.0
    
    # Iterate through indices multiple times (random order) to flip if it improves peak
    # This is a form of coordinate descent / local search
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
            old_val = f_discrete[i]
            f_discrete[i] = -old_val
            new_peak = cost_discrete_peak(f_discrete)
            
            if new_peak < best_peak:
                best_peak = new_peak
                best_f = f_discrete.copy()
                improved = True
            else:
                f_discrete[i] = old_val
                
    return best_f
# EVOLVE-BLOCK-END
