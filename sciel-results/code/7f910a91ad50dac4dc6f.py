"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Discrete Local Search (Greedy Flip) on signs +/- 1
    # Objective: Minimize max(|autocorr|) / (sum)^2
    
    np.random.seed(42)
    f = np.random.choice([-1.0, 1.0], size=N)
    
    # Helper to calculate autocorrelation using FFT
    def calc_autocorr(f):
        n_fft = 2 * N
        F = np.fft.fft(f, n=n_fft)
        P = F * np.conj(F)
        C_full = np.fft.ifft(P).real
        return C_full[:N]

    C = calc_autocorr(f)
    S = np.sum(f)
    
    # Handle case where sum is 0 (score is inf)
    if S == 0:
        # Flip one random element to make sum non-zero
        idx = np.random.randint(N)
        f[idx] *= -1
        S = np.sum(f)
        C = calc_autocorr(f)
        
    score = np.max(np.abs(C)) / (S * S)
    
    best_f = f.copy()
    best_C = C.copy()
    best_S = S
    best_score = score
    
    # Greedy Flip Search
    # We attempt to flip each index in random order multiple times
    # Update rule:
    # Flipping f[i] changes C[k] by -2*f[i]*(f[i+k] + f[i-k])
    # and S by -2*f[i]
    
    max_passes = 50
    
    for _ in range(max_passes):
        # Random permutation of indices
        indices = np.random.permutation(N)
        
        for i in indices:
            old_fi = f[i]
            
            # Calculate delta C
            # d = -2 * old_fi * (f[i+k] + f[i-k])
            # We need to update C array
            
            # Create delta array
            d = np.zeros(N)
            
            # Part 1: f[i+k] for k=0..N-1-i
            # d[0:N-i] += -2 * old_fi * f[i:]
            if N - i > 0:
                d[:N-i] += -2 * old_fi * f[i:]
                
            # Part 2: f[i-k] for k=0..i
            # d[0:i+1] += -2 * old_fi * f[i::-1]
            if i + 1 > 0:
                d[:i+1] += -2 * old_fi * f[i::-1]
                
            new_C = C + d
            new_S = S - 2 * old_fi
            
            # Calculate new score
            if new_S == 0:
                new_score = np.inf
            else:
                new_score = np.max(np.abs(new_C)) / (new_S * new_S)
                
            # Accept if better
            if new_score < best_score:
                f[i] = -old_fi
                C = new_C
                S = new_S
                best_f = f.copy()
                best_C = new_C.copy()
                best_S = new_S
                best_score = new_score
                
    # Return normalized sequence
    if best_S != 0:
        return best_f / best_S
    else:
        return best_f

# EVOLVE-BLOCK-END
