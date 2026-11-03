"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    def autoconv(f):
        return np.convolve(f, f, mode='full')
    
    def score(f):
        s = np.sum(f)
        if abs(s) < 1e-10:
            return np.inf
        c = autoconv(f)
        return np.max(np.abs(c)) / (s * s)
    
    def optimize(f_init, steps=500, eta=0.05):
        f = f_init.copy()
        # Normalize to sum 1
        s = np.sum(f)
        if abs(s) < 1e-10:
            # If sum is zero, perturb
            f += np.random.normal(0, 0.1, N)
            s = np.sum(f)
        f = f / s
        
        for _ in range(steps):
            c = autoconv(f)
            max_idx = np.argmax(np.abs(c))
            val = c[max_idx]
            if abs(val) < 1e-10:
                break
            sign = 1.0 if val > 0 else -1.0
            
            # Lag calculation
            # np.convolve(a, b, 'full') index i corresponds to lag i - (N-1)
            k = max_idx - (N - 1)
            
            # Gradient of |c_k| wrt f_i
            # c_k = sum_{i} f_i f_{i+k}
            # d c_k / d f_i = f_{i-k} + f_{i+k}
            
            grad = np.zeros(N)
            
            if k >= 0:
                # f[i+k] valid for i in [0, N-1-k]
                grad[:N-k] += f[k:]
                # f[i-k] valid for i in [k, N-1]
                grad[k:] += f[:N-k]
            else:
                # k is negative, let kk = -k
                kk = -k
                # f[i-kk] valid for i in [kk, N-1]
                grad[kk:] += f[:N-kk]
                # f[i+kk] valid for i in [0, N-1-kk]
                grad[:N-kk] += f[kk:]
                
            grad *= sign
            
            # Update
            f -= eta * grad
            
            # Project to sum=1
            f -= np.mean(f)
            f += 1.0 / N
            
        return f

    # Initializations
    inits = []
    
    # 1. Random
    rng = np.random.default_rng(42)
    inits.append(rng.normal(0, 1, N))
    
    # 2. Chirp (Linear Frequency Modulation)
    t = np.arange(N)
    inits.append(np.cos(2 * np.pi * (t * t) / (2 * N)))
    
    # 3. Cosine
    inits.append(np.cos(2 * np.pi * t / N))
    
    # 4. Square
    sq = np.ones(N)
    sq[N//2:] = -1
    inits.append(sq)
    
    # 5. Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    inits.append(alt)
    
    best_f = None
    best_score = np.inf
    
    for init in inits:
        f_opt = optimize(init, steps=500, eta=0.05)
        sc = score(f_opt)
        if sc < best_score:
            best_score = sc
            best_f = f_opt
            
    # Also check the initializations themselves
    for init in inits:
        sc = score(init)
        if sc < best_score:
            best_score = sc
            best_f = init.copy()

    if best_f is None:
        return np.ones(N)
        
    return best_f
# EVOLVE-BLOCK-END
