import numpy as np
import time

N = 100

def compute_min_autocorrelation(h):
    """Compute min over s of sum_i h_i * h_{i+s}."""
    # Use FFT to compute autocorrelation
    # Zero pad to 2*N to avoid circular convolution issues
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    AC = np.fft.ifft(H * np.conj(H)).real
    # The autocorrelation for lags 0 to N-1 is in AC[0:N]
    # We care about lags 1 to N-1 (shifts)
    # Note: np.fft.ifft returns unnormalized result. 
    # Standard autocorr is (1/N) * ifft(|FFT|^2). 
    # But we just want to compare values, so scaling doesn't matter for argmin, 
    # but we need the actual value for the objective? 
    # The objective is min_s A_s.
    # Let's just return the min value of the first N elements.
    return np.min(AC[1:N])

def project_to_feasible(h):
    """Project h to {x | sum(x)=N/2, 0<=x<=1}."""
    # Simple projection: adjust mean, then clip, then re-adjust mean?
    # This is an iterative process or a specific algorithm.
    # For simplicity and speed, we can use a simple "water filling" or just 
    # adjust the largest/smallest elements to fix the sum.
    
    target_sum = N / 2.0
    current_sum = h.sum()
    diff = target_sum - current_sum
    
    if abs(diff) < 1e-9:
        return h
    
    # Identify indices that can absorb the change
    if diff > 0:
        # Need to increase sum. Find indices with h < 1.
        # Sort by h descending? No, just add to those with most room.
        # To minimize disruption, distribute evenly?
        # Simple approach: add diff/N to all, then clip, then fix residual.
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Fix residual
        residual = target_sum - h.sum()
        if residual > 0:
            # Add to indices with most room (h closest to 0)
            indices = np.argsort(h)
            for idx in indices:
                if residual <= 0: break
                room = 1.0 - h[idx]
                add = min(room, residual)
                h[idx] += add
                residual -= add
        elif residual < 0:
            # Subtract from indices with most excess (h closest to 1)
            indices = np.argsort(h)[::-1]
            for idx in indices:
                if residual >= 0: break
                room = h[idx]
                sub = min(room, -residual)
                h[idx] -= sub
                residual += sub
    else:
        # Need to decrease sum
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        residual = target_sum - h.sum()
        if residual > 0:
            indices = np.argsort(h)
            for idx in indices:
                if residual <= 0: break
                room = 1.0 - h[idx]
                add = min(room, residual)
                h[idx] += add
                residual -= add
        elif residual < 0:
            indices = np.argsort(h)[::-1]
            for idx in indices:
                if residual >= 0: break
                room = h[idx]
                sub = min(room, -residual)
                h[idx] -= sub
                residual += sub
    return h

def construct():
    deadline = time.time() + 110.0
    np.random.seed(42)
    
    # Initialize with Thue-Morse
    h = np.zeros(N)
    for i in range(N):
        cnt = bin(i).count('1')
        h[i] = 1.0 if cnt % 2 == 1 else 0.0
        
    # Ensure exact sum
    h = project_to_feasible(h)
    
    best_h = h.copy()
    best_score = -compute_min_autocorrelation(best_h) # We want to maximize min autocorr, so minimize -min_autocorr
    
    # Local search
    step_size = 0.1
    lr = 0.5
    
    while time.time() < deadline:
        improved = False
        # Try perturbing random positions
        for _ in range(1000): # Fixed number of steps per pass, check time
            if time.time() > deadline:
                break
                
            i = np.random.randint(N)
            old_val = h[i]
            
            # Random perturbation
            new_val = np.clip(old_val + np.random.uniform(-lr, lr), 0.0, 1.0)
            h[i] = new_val
            
            # Project to feasible
            h = project_to_feasible(h)
            
            score = -compute_min_autocorrelation(h)
            
            if score < best_score - 1e-9:
                best_h = h.copy()
                best_score = score
                improved = True
            else:
                # Revert? Or keep if it's a random walk?
                # For hill climbing, revert.
                # But projection changed other elements. 
                # So we should store the previous state.
                pass
        
        # If no improvement, maybe do a random restart or larger step
        if not improved:
             # Try a random restart of a few bits
             for _ in range(5):
                 i = np.random.randint(N)
                 h[i] = np.random.uniform(0, 1)
             h = project_to_feasible(h)
             score = -compute_min_autocorrelation(h)
             if score < best_score:
                 best_h = h.copy()
                 best_score = score
    
    return best_h
