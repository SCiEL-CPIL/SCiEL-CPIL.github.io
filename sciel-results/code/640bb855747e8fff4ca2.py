"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _objective(f):
    """Compute max(|autoconv|) / (sum(f))^2"""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def _ac_offmax(f, size):
    """Compute max absolute off-zero autoconvolution value."""
    n = len(f)
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    # Exclude lag 0 (index n-1)
    off = np.concatenate([auto[:n - 1], auto[n:]])
    return np.max(np.abs(off))


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    n = N
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    
    # Initialize with all ones
    best_f = np.ones(N)
    best_score = _objective(best_f)
    
    # Precompute FFT size for speed
    def fast_obj(f):
        s = np.sum(f)
        if abs(s) < 1e-12:
            return float('inf')
        offmax = _ac_offmax(f, size)
        return offmax / (s * s)
    
    # Strategy 1: Random ±1 sequences with local refinement
    rng = np.random.default_rng(42)
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        f = rng.choice([-1.0, 1.0], size=n)
        s = np.sum(f)
        if abs(s) < 1e-9:
            continue
        
        score = fast_obj(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
        
        # Local search: greedy single-flip
        current = f.copy()
        cur_score = score
        cur_sum = s
        
        for _pass in range(5):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                
                old_val = current[i]
                new_sum = cur_sum - 2.0 * old_val
                if abs(new_sum) < 1e-9:
                    continue
                
                # Try flip
                current[i] = -old_val
                offmax = _ac_offmax(current, size)
                new_score = offmax / (new_sum * new_sum)
                current[i] = old_val
                
                if new_score < cur_score - 1e-12:
                    # Accept improvement
                    current[i] = -old_val
                    cur_sum = new_sum
                    cur_score = new_score
                    improved = True
                    
                    if cur_score < best_score:
                        best_score = cur_score
                        best_f = current.copy()
            
            if not improved:
                break
    
    # Strategy 2: Sequences with mostly +1 and some -1
    for n_neg in range(0, 30):
        if time.time() > deadline:
            break
        
        for trial in range(10):
            if time.time() > deadline:
                break
            
            f = np.ones(n)
            neg_idx = rng.choice(n, size=n_neg, replace=False)
            f[neg_idx] = -1.0
            
            s = np.sum(f)
            if abs(s) < 1e-9:
                continue
            
            score = fast_obj(f)
            if score < best_score:
                best_score = score
                best_f = f.copy()
            
            # Quick local refinement
            current = f.copy()
            cur_score = score
            cur_sum = s
            
            for _pass in range(2):
                if time.time() > deadline:
                    break
                
                improved = False
                for i in range(n):
                    if time.time() > deadline:
                        break
                    
                    old_val = current[i]
                    new_sum = cur_sum - 2.0 * old_val
                    if abs(new_sum) < 1e-9:
                        continue
                    
                    current[i] = -old_val
                    offmax = _ac_offmax(current, size)
                    new_score = offmax / (new_sum * new_sum)
                    current[i] = old_val
                    
                    if new_score < cur_score - 1e-12:
                        current[i] = -old_val
                        cur_sum = new_sum
                        cur_score = new_score
                        improved = True
                        
                        if cur_score < best_score:
                            best_score = cur_score
                            best_f = current.copy()
                
                if not improved:
                    break
    
    # Final polish: more intensive local search on best candidate
    if time.time() < deadline - 10:
        current = best_f.copy()
        cur_score = best_score
        cur_sum = np.sum(current)
        
        for _pass in range(10):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                
                old_val = current[i]
                new_sum = cur_sum - 2.0 * old_val
                if abs(new_sum) < 1e-9:
                    continue
                
                current[i] = -old_val
                offmax = _ac_offmax(current, size)
                new_score = offmax / (new_sum * new_sum)
                current[i] = old_val
                
                if new_score < cur_score - 1e-12:
                    current[i] = -old_val
                    cur_sum = new_sum
                    cur_score = new_score
                    improved = True
                    best_score = cur_score
                    best_f = current.copy()
            
            if not improved:
                break
    
    return best_f
# EVOLVE-BLOCK-END
