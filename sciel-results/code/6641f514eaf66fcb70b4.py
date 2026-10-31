"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize f with random signs
    np.random.seed(42)
    f = np.random.choice([-1.0, 1.0], size=N)
    
    # Normalize sum to 1
    f = f / np.sum(f)
    
    # Learning rate
    lr = 0.01
    # Number of iterations
    iters = 5000
    
    # Precompute indices for autocorrelation? 
    # Direct calculation is O(N^2). 
    # C[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    
    best_f = f.copy()
    best_score = np.inf
    
    for _ in range(iters):
        # Calculate Autocorrelation
        C = np.zeros(N)
        for k in range(N):
            C[k] = np.dot(f[:N-k], f[k:])
            
        # Find max absolute value
        max_abs_idx = np.argmax(np.abs(C))
        max_val = C[max_abs_idx]
        
        # If max_val is 0, gradient is 0, break or continue
        if abs(max_val) < 1e-12:
            break
            
        sign_max = 1.0 if max_val > 0 else -1.0
        
        # Calculate Gradient of C[k_max] w.r.t f
        # C[k] = sum_i f[i] f[i+k]
        # dC[k]/df[j] = f[j+k] + f[j-k] (if indices valid)
        grad = np.zeros(N)
        k = max_abs_idx
        
        # Contribution from f[i] * f[i+k]
        # Term 1: f[i] * f[i+k]. Deriv w.r.t f[j]:
        # If j = i, deriv is f[i+k] = f[j+k]
        # If j = i+k, deriv is f[i] = f[j-k]
        
        # So for each i in 0..N-1-k:
        # grad[i] += f[i+k]
        # grad[i+k] += f[i]
        
        grad[:N-k] += f[k:]
        grad[k:] += f[:N-k]
        
        # Gradient of |C[k]| is sign(C[k]) * grad(C[k])
        grad *= sign_max
        
        # Update f
        f -= lr * grad
        
        # Project to sum(f) = 1
        f -= (np.sum(f) - 1.0) / N
        
        # Calculate score
        # Score is max(|C|) / (sum(f))^2. Since sum(f)=1, score = max(|C|)
        current_score = np.max(np.abs(C)) # C is from previous step, need to recalc or approximate?
        # Actually, we should recalc C for the score check to be accurate, 
        # but for speed, we can just check the max of the new C in the next loop's start?
        # Let's just track the max_val we just used as a proxy, or recalc.
        # Recalc is cheap enough.
        
        # Recalc C for accurate score tracking
        C_new = np.zeros(N)
        for k in range(N):
            C_new[k] = np.dot(f[:N-k], f[k:])
        current_score = np.max(np.abs(C_new))
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            
        # Optional: Adaptive learning rate or early stopping
        # if _ % 100 == 0 and current_score > best_score * 1.01:
        #     lr *= 0.5
            
    return best_f
# EVOLVE-BLOCK-END
