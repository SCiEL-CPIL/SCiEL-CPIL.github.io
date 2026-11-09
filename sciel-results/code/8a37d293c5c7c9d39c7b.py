import numpy as np

N = 400

def construct():
    # Simulated annealing approach for minimizing autocorrelation
    # Objective: minimize sum of squares of non-zero autocorrelations
    
    def compute_autocorr(x):
        """Compute linear autocorrelation and return sum of squares of non-zero lags."""
        R_lin = np.correlate(x, x, mode='full')
        non_central = np.concatenate([R_lin[:N-1], R_lin[N:]])
        return np.sum(non_central**2)
    
    def delta_score(x, i):
        """Compute change in objective when flipping bit i."""
        # When flipping x[i], the change in autocorrelation at lag k is:
        # For k > 0: R[k] changes by -2*x[i]*(x[i+k] + x[i-k]) + 4*x[i]*x[i+k]*x[i-k] if both i+k and i-k are valid
        # This is complex, so let's use a simpler approach: compute old and new scores
        # But for efficiency, let's derive the delta properly
        
        # R[k] = sum_j x[j]*x[j+k]
        # When x[i] flips to -x[i]:
        # New R[k] = R[k] - 2*x[i]*(x[i+k] if i+k<N else 0 + x[i-k] if i-k>=0 else 0) + ... 
        # Actually, let's just compute it directly for clarity and correctness
        
        old_score = compute_autocorr(x)
        
        x_trial = x.copy()
        x_trial[i] *= -1
        new_score = compute_autocorr(x_trial)
        
        return new_score - old_score
    
    # Initialize with random sequence
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N)
    
    current_score = compute_autocorr(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated annealing parameters
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.995
    iterations = 2000
    
    T = T_start
    for it in range(iterations):
        # Pick random index to flip
        i = np.random.randint(N)
        
        # Compute delta
        delta = delta_score(x, i)
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x[i] *= -1
            current_score += delta
            
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        
        T *= cooling_rate
    
    return best_x

# EVOLVE-BLOCK-END
