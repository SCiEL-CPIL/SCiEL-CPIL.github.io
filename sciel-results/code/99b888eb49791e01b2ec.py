import numpy as np
import random

def construct():
    N = 24
    # Precompute the matrix Z for evaluation
    # Z[k, j] = exp(1j * k * angle_j)
    # Dense sampling to approximate max modulus
    # 4096 samples is sufficient for N=24 to locate peaks accurately
    num_samples = 4096
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    k_vals = np.arange(N)
    Z = np.exp(1j * np.outer(k_vals, angles)) # Shape (N, num_samples)
    
    def eval_score(c):
        # c is shape (N,)
        p_vals = Z.T @ c
        return np.max(np.abs(p_vals))
    
    best_coeffs = None
    best_score = float('inf')
    
    # Use Simulated Annealing to escape local minima
    # Multiple restarts to improve global search probability
    for _ in range(10):
        # Initialize with random signs
        c = np.random.choice([-1.0, 1.0], size=N)
        current_score = eval_score(c)
        
        T = 1.0 # Initial temperature
        cooling_rate = 0.999
        min_T = 1e-5
        max_iter = 20000
        
        it = 0
        while T > min_T and it < max_iter:
            # Pick a random bit to flip
            i = random.randint(0, N - 1)
            c[i] = -c[i]
            new_score = eval_score(c)
            
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
            else:
                # Accept with probability exp(-delta / T)
                # Clip delta to avoid underflow/overflow issues if T is very small
                if T > 1e-9:
                    if random.random() < np.exp(-delta / T):
                        current_score = new_score
                    else:
                        c[i] = -c[i] # Revert
                else:
                    c[i] = -c[i] # Revert if T is effectively 0 and move is worse
            
            T *= cooling_rate
            it += 1
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = c.copy()
            
    # Final local refinement (Hill Climbing) on the best SA result
    c = best_coeffs
    current_score = eval_score(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            old_val = c[i]
            c[i] = -old_val
            new_score = eval_score(c)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                c[i] = old_val
                
    return [int(x) for x in c]
