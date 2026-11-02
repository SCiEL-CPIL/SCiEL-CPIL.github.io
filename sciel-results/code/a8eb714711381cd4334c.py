"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute autoconvolution of x (x * reverse(x))."""
    return np.correlate(x, x[::-1], mode='full')

def _score(x):
    """Score: max absolute autoconvolution / (sum(x))^2. Lower is better."""
    ac = _autoconv(x)
    integral = np.sum(x)
    if integral == 0:
        return np.max(np.abs(ac))
    return np.max(np.abs(ac)) / (integral * integral)

def _soft_max(vals, beta):
    """Soft maximum approximation using log-sum-exp."""
    vals = np.asarray(vals)
    m = np.max(vals)
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 140s safety margin (150s limit)
    
    # Initial: alternating pattern gives good baseline
    x = np.ones(N)
    x[1::2] = -1
    
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Coordinate descent with multiple move types
    def try_move(x, i, new_val):
        """Try setting x[i] to new_val, return new x and score if better."""
        x[i] = new_val
        s = _score(x)
        return x, s
    
    def coordinate_descent(x, max_iters, deadline):
        """Basic coordinate descent: flip individual elements."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(N):
                old_val = x[i]
                x[i] = -old_val
                s = _score(x)
                if s < best_score:
                    best_score = s
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = old_val
        return x, best_score
    
    # Phase 1: Basic coordinate descent
    x, best_score = coordinate_descent(x, 50, deadline)
    
    if time.time() > deadline - 30:
        return best_x
    
    # Phase 2: Block moves (flip consecutive blocks)
    def block_descent(x, block_sizes, max_iters, deadline):
        """Try flipping blocks of consecutive elements."""
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            for bs in block_sizes:
                for start in range(0, N - bs + 1, bs):
                    old_block = x[start:start+bs].copy()
                    x[start:start+bs] = -x[start:start+bs]
                    s = _score(x)
                    if s < best_score:
                        best_score = s
                        best_x = x.copy()
                        improved = True
                    else:
                        x[start:start+bs] = old_block
            if not improved:
                break
        return x, best_score
    
    x, best_score = block_descent(x, [2, 3, 4, 5], 10, deadline)
    
    if time.time() > deadline - 20:
        return best_x
    
    # Phase 3: Soft-max gradient descent (continuous relaxation then rounding)
    def softmax_gradient_descent(x_init, lr, max_iters, beta, deadline):
        """Gradient descent on soft-max of autoconvolution."""
        x = x_init.astype(float)
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            ac = _autoconv(x)
            abs_ac = np.abs(ac)
            # Soft max gradient
            m = np.max(abs_ac)
            exps = np.exp(beta * (abs_ac - m))
            weights = exps / np.sum(exps)
            grad = weights * np.sign(ac)
            
            # Gradient of autoconvolution w.r.t. x
            # ac[k] = sum_j x[j] * x[j+k-N] (for appropriate k)
            # Simplified: use finite differences or analytical
            # For autoconvolution, grad_x[i] = sum_k grad_ac[k] * d_ac[k]/dx[i]
            # d_ac[k]/dx[i] = x[i+k] + x[i-k] (approximately)
            
            # Use a simpler approach: perturb each element
            for j in range(0, N, 2):  # Every other element for speed
                old = x[j]
                x[j] += lr * 0.1
                ac_new = _autoconv(x)
                new_obj = _soft_max(np.abs(ac_new), beta)
                x[j] -= lr * 0.1
                ac_old = _autoconv(x)
                old_obj = _soft_max(np.abs(ac_old), beta)
                if new_obj < old_obj:
                    x[j] = old + lr * 0.1
                else:
                    x[j] = old - lr * 0.1
            
            # Round to ±1
            x = np.sign(x)
            s = _score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
        
        return x, best_score
    
    # Run soft-max GD from current best
    x, best_score = softmax_gradient_descent(best_x, 1.0, 100, 100, deadline)
    
    if time.time() > deadline - 10:
        return best_x
    
    # Phase 4: Peak-targeted polish
    def peak_polish(x, max_iters, deadline):
        """Directly reduce dominant autoconvolution peaks."""
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            ac = _autoconv(x)
            abs_ac = np.abs(ac)
            peak_idx = np.argmax(abs_ac)
            peak_val = ac[peak_idx]
            
            # Find elements contributing most to this peak
            # ac[peak_idx] = sum_j x[j] * x[j + peak_idx - N]
            offset = peak_idx - N
            contributions = x * np.roll(x, -offset)
            
            # Try flipping elements with large contribution to peak
            contrib_abs = np.abs(contributions)
            top_indices = np.argsort(-contrib_abs)[:20]
            
            improved = False
            for idx in top_indices:
                old_val = x[idx]
                x[idx] = -old_val
                s = _score(x)
                if s < best_score:
                    best_score = s
                    best_x = x.copy()
                    improved = True
                else:
                    x[idx] = old_val
            
            if not improved:
                break
        
        return x, best_score
    
    x, best_score = peak_polish(best_x, 50, deadline)
    
    if time.time() > deadline - 5:
        return best_x
    
    # Phase 5: Final coordinate descent polish
    x, best_score = coordinate_descent(best_x, 20, deadline)
    
    return best_x

# EVOLVE-BLOCK-END
