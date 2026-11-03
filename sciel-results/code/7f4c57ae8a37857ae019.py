"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _overlap_vector(h):
    """Compute overlap O(s) for all shifts s."""
    N = len(h)
    O = np.zeros(N)
    for s in range(N):
        shifted = np.roll(h, -s)
        O[s] = np.sum(h * (1.0 - shifted))
    return O


def _project(h, target_sum):
    """Project h onto {0<=h[i]<=1, sum(h)=target_sum}."""
    h = np.clip(h, 0.0, 1.0)
    # Binary search for offset to satisfy sum constraint
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_try = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_try)
        if s > target_sum:
            hi = mid
        else:
            lo = mid
    h_try = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    return h_try


def _softmax_grad(O, beta):
    """Gradient of log-sum-exp(beta * O) w.r.t. O, times beta for chain rule."""
    max_O = np.max(O)
    exp_vals = np.exp(beta * (O - max_O))
    weights = exp_vals / np.sum(exp_vals)
    return beta * weights


def construct():
    deadline = time.time() + 140.0  # 140s deadline, 10s safety margin
    target_sum = N / 2.0
    
    # Initial guess: alternating blocks
    h = np.zeros(N)
    # Try a structured initial guess: blocks of 1s and 0s
    # Pattern: 11110000 repeated (4 ones, 4 zeros) gives period 8
    # 100 / 8 = 12.5, so 12 full periods (96) + 4 extra
    for i in range(N):
        h[i] = 1.0 if (i % 8) < 4 else 0.0
    h = _project(h, target_sum)
    
    # Compute initial score
    O = _overlap_vector(h)
    best_score = np.max(O)
    best_h = h.copy()
    
    # PGD with increasing beta
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        # Adaptive step size
        lr = 0.1 / beta  # scale with beta
        max_iters = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            O = _overlap_vector(h)
            grad_O = _softmax_grad(O, beta)
            
            # Gradient of soft-max w.r.t. h
            # d/dh[i] of O(s) = (1-h[(i+s) mod N]) - h[i]*(1-h[(i+s) mod N])... 
            # Actually O(s) = sum_i h[i](1-h[(i+s)%N])
            # dO(s)/dh[i] = (1-h[(i+s)%N]) + h[i]*(-1)*d/dh[i]h[(i+s)%N]
            # If j = (i+s)%N, then d/dh[i] h[j] = 1 if j==i else 0
            # So dO(s)/dh[i] = (1-h[(i+s)%N]) - h[i]*delta_{i,(i+s)%N}
            # For s=0: dO(0)/dh[i] = (1-h[i]) - h[i] = 1-2h[i]
            # For s!=0: dO(s)/dh[i] = (1-h[(i+s)%N])
            
            # Let me compute gradient more carefully
            # O(s) = sum_i h[i] * (1 - h[(i+s)%N])
            # grad_h[i] = sum_s grad_O[s] * dO(s)/dh[i]
            
            grad_h = np.zeros(N)
            for s in range(N):
                if s == 0:
                    # dO(0)/dh[i] = 1 - 2*h[i]
                    grad_h += grad_O[s] * (1.0 - 2.0 * h)
                else:
                    # dO(s)/dh[i] = (1 - h[(i+s)%N])
                    shifted_h = np.roll(h, -s)
                    grad_h += grad_O[s] * (1.0 - shifted_h)
            
            # Update
            h_new = h - lr * grad_h
            h_new = _project(h_new, target_sum)
            
            # Check convergence
            if np.max(np.abs(h_new - h)) < 1e-10:
                h = h_new
                break
            h = h_new
        
        # Evaluate and update best
        O = _overlap_vector(h)
        score = np.max(O)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final polish: try to reduce max overlap further with local moves
    if time.time() < deadline - 5.0:
        # Simple local optimization: try small perturbations
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            O = _overlap_vector(best_h)
            max_idx = np.argmax(O)
            # Try to reduce O[max_idx] by adjusting h
            # O(s) = sum_i h[i](1-h[(i+s)%N])
            # To reduce O(s), we want to make h[i] and h[(i+s)%N] more correlated
            # Try swapping or adjusting pairs
            improved = False
            for i in range(N):
                j = (i + max_idx) % N
                if i == j:
                    continue
                # Try making h[i] and h[j] more similar
                # If h[i] < h[j], try increasing h[i] and decreasing h[j]
                # If h[i] > h[j], try decreasing h[i] and increasing h[j]
                delta = 0.05
                if h[i] < h[j]:
                    h_try = best_h.copy()
                    h_try[i] = min(1.0, h[i] + delta)
                    h_try[j] = max(0.0, h[j] - delta)
                else:
                    h_try = best_h.copy()
                    h_try[i] = max(0.0, h[i] - delta)
                    h_try[j] = min(1.0, h[j] + delta)
                
                # Check sum constraint
                if abs(np.sum(h_try) - target_sum) > 1e-6:
                    continue
                
                O_try = _overlap_vector(h_try)
                if np.max(O_try) < best_score - 1e-8:
                    best_h = h_try
                    best_score = np.max(O_try)
                    improved = True
                    break
            if not improved:
                break
    
    return best_h
# EVOLVE-BLOCK-END
