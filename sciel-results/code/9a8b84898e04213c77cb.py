"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_overlaps(h):
    """Compute C(k) = sum_i h[i]*(1-h[(i+k)%N]) for all k."""
    # C(k) = N/2 - autocorr(h, k)
    # autocorr(h, k) = sum_i h[i]*h[(i+k)%N]
    # Use FFT for efficiency
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    # Circular autocorrelation via FFT
    H = np.fft.fft(h_pad, n=2*N)
    acf = np.fft.ifft(H * np.conj(H)).real[:N]
    # C(k) = N/2 - acf[k]
    return N / 2.0 - acf

def _soft_max_overlaps(h, beta=50.0):
    """Smooth max of overlaps using log-sum-exp."""
    C = _compute_overlaps(h)
    # soft max = (1/beta) * log(sum(exp(beta*C)))
    # For numerical stability
    C_max = np.max(C)
    return C_max + np.log(np.sum(np.exp(beta * (C - C_max)))) / beta

def _project(h):
    """Project h onto {h in [0,1]^N : sum(h) = N/2}."""
    # Simple projection: clip and then adjust
    h_clipped = np.clip(h, 0.0, 1.0)
    # Adjust to satisfy sum constraint
    diff = np.sum(h_clipped) - N / 2.0
    if abs(diff) < 1e-10:
        return h_clipped
    # Distribute the correction
    h_adj = h_clipped - diff / N
    h_adj = np.clip(h_adj, 0.0, 1.0)
    # May need a few iterations
    for _ in range(10):
        diff = np.sum(h_adj) - N / 2.0
        if abs(diff) < 1e-12:
            break
        # Only adjust elements not at boundary
        mask = (h_adj > 1e-10) & (h_adj < 1.0 - 1e-10)
        if np.sum(mask) == 0:
            break
        h_adj[mask] -= diff / np.sum(mask)
        h_adj = np.clip(h_adj, 0.0, 1.0)
    return h_adj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with random initialization
    np.random.seed(42)
    h0 = np.random.uniform(0, 1, N)
    h0 = _project(h0)
    
    best_h = h0.copy()
    best_score = _soft_max_overlaps(h0, beta=50.0)
    # Also track actual max overlap
    best_actual_max = np.max(_compute_overlaps(h0))
    
    # Try multiple restarts with different seeds and beta values
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            h_curr = h0.copy()
        else:
            seed = restart * 1000 + 42
            np.random.seed(seed)
            h_curr = np.random.uniform(0, 1, N)
            h_curr = _project(h_curr)
        
        # Gradient descent with projection
        beta = 50.0
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient of soft-max
            C = _compute_overlaps(h_curr)
            C_max = np.max(C)
            weights = np.exp(beta * (C - C_max)) / np.sum(np.exp(beta * (C - C_max)))
            
            # Gradient of C(k) w.r.t. h
            # C(k) = N/2 - sum_i h[i]*h[(i+k)%N]
            # dC(k)/dh[j] = -sum_i [delta_{j,i} h[(i+k)%N] + h[i] delta_{j,(i+k)%N}]
            # = -h[(j-k)%N] - sum_{i: (i+k)%N = j} h[i]
            # = -h[(j-k)%N] - h[(j-k)%N] = -2*h[(j-k)%N]... wait
            # Let me recompute: C(k) = sum_i h[i]*(1-h[(i+k)%N])
            # dC(k)/dh[j] = sum_i [delta_{i,j}(1-h[(i+k)%N]) - h[i]*delta_{(i+k)%N,j}]
            # = (1-h[(j+k)%N]) - h[(j-k)%N]
            # So grad = sum_k weights[k] * [(1-h[(j+k)%N]) - h[(j-k)%N]]
            
            grad = np.zeros(N)
            for k in range(N):
                if weights[k] < 1e-10:
                    continue
                # Contribution to each j
                # (1-h[(j+k)%N]) - h[(j-k)%N]
                idx1 = (np.arange(N) + k) % N
                idx2 = (np.arange(N) - k) % N
                grad += weights[k] * ((1 - h_curr[idx1]) - h_curr[idx2])
            
            # Update
            h_new = h_curr - lr * grad
            h_new = _project(h_new)
            
            # Check improvement
            curr_score = _soft_max_overlaps(h_new, beta=beta)
            if curr_score < best_score:
                best_score = curr_score
                best_h = h_new.copy()
                best_actual_max = np.max(_compute_overlaps(h_new))
            
            h_curr = h_new
            
            # Adaptive learning rate
            if step % 200 == 0 and step > 0:
                lr *= 0.8
        
        # After each restart, try to polish with higher beta
        if time.time() > deadline - 10.0:
            break
        beta2 = 200.0
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
            C = _compute_overlaps(best_h)
            C_max = np.max(C)
            weights = np.exp(beta2 * (C - C_max)) / np.sum(np.exp(beta2 * (C - C_max)))
            
            grad = np.zeros(N)
            for k in range(N):
                if weights[k] < 1e-10:
                    continue
                idx1 = (np.arange(N) + k) % N
                idx2 = (np.arange(N) - k) % N
                grad += weights[k] * ((1 - best_h[idx1]) - best_h[idx2])
            
            h_new = best_h - 0.005 * grad
            h_new = _project(h_new)
            curr_score = _soft_max_overlaps(h_new, beta=beta2)
            if curr_score < best_score:
                best_score = curr_score
                best_h = h_new.copy()
                best_actual_max = np.max(_compute_overlaps(h_new))
            best_h = h_new
    
    # Final: return the best found, ensuring exact sum
    h_final = best_h.copy()
    # Ensure sum is exactly N/2
    h_final = np.clip(h_final, 0.0, 1.0)
    diff = np.sum(h_final) - N / 2.0
    if abs(diff) > 1e-10:
        # Distribute correction to elements not at boundary
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_final[mask] -= diff / np.sum(mask)
        h_final = np.clip(h_final, 0.0, 1.0)
    
    return h_final
# EVOLVE-BLOCK-END
