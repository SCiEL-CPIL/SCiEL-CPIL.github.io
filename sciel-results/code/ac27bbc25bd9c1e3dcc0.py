import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def compute_grad(h):
        """Compute gradient of the objective with respect to h."""
        s = np.sum(h)
        c = np.convolve(h, h)
        idx_max = np.argmax(c)
        # Gradient of max(c) w.r.t. h at the maximizing index
        # c[k] = sum_i h[i] * h[k-i] (with appropriate boundaries for full convolution)
        # dc/dh[j] = sum_i [delta(i, j) * h[k-i] + h[i] * delta(i, k-j)]
        # For full convolution: c[k] = sum_{i} h[i] * h[k-i] where valid indices
        # Actually for full convolution, c[k] = sum_{i} h[i] * h[k-i] for k in range(2N-1)
        # The contribution to c[idx_max] from h[j] is:
        # If we increase h[j], then c[idx_max] increases by h[idx_max - j] (if idx_max-j is valid) 
        # plus h[j] doesn't appear in the other term directly... 
        # Let me think more carefully.
        # c[k] = sum_{i=0}^{N-1} h[i] * h[k-i] where we need 0 <= k-i < N
        # So c[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[k-i]
        # dc[k]/dh[j] = (if max(0,k-N+1) <= j <= min(N-1,k)) then h[k-j] else 0
        # Plus the symmetric term: (if max(0,k-N+1) <= k-j <= min(N-1,k)) then h[j] else 0
        # Wait, that's the same condition: j is in [max(0,k-N+1), min(N-1,k)] AND k-j is in [0, N-1]
        # which is equivalent to j in [max(0,k-N+1), min(N-1,k)]
        # So dc[k]/dh[j] = 2 * h[k-j] if j is in the valid range, else 0
        # Actually no. Let me redo.
        # c[k] = sum_i h[i] * h[k-i]
        # dc[k]/dh[j] = (d/dh[j]) sum_i h[i] * h[k-i]
        # = sum_i [delta(i,j) * h[k-i] + h[i] * delta(k-i, j)]
        # = h[k-j] + h[j] (if the indices are valid)
        # Wait, delta(k-i, j) means k-i=j, so i=k-j.
        # So dc[k]/dh[j] = h[k-j] (from first term, i=j) + h[k-j] (from second term, i=k-j)
        # = 2*h[k-j] if both i=j and i=k-j are valid indices, i.e., 0<=j<N and 0<=k-j<N
        # But wait, if j = k-j, i.e., k=2j, then it's h[j] + h[j] = 2*h[j], which is correct.
        # Otherwise, if j != k-j, we get h[k-j] + h[k-j]... no wait.
        # First term: i=j gives h[k-j] (provided 0<=k-j<N)
        # Second term: i=k-j gives h[j] (provided 0<=k-j<N, which is the same condition)
        # So dc[k]/dh[j] = h[k-j] + h[j] if 0<=j<N and 0<=k-j<N, else 0.
        
        k = idx_max
        grad_c = np.zeros(N)
        for j in range(N):
            if 0 <= k - j < N:
                grad_c[j] = h[k - j] + h[j]
        
        # Now the objective is |2*N*max(c)/s^2|. Since we're at the max, and assuming max(c)>0 and s>0
        # f = 2*N*c_max/s^2
        # df/dh[j] = 2*N * (dc_max/dh[j] * s^2 - c_max * 2*s*ds/dh[j]) / s^4
        # = 2*N * (dc_max/dh[j] * s - 2*c_max) / s^3
        # where ds/dh[j] = 1
        
        # But we need to handle the abs. If the value is negative, the gradient flips.
        # For now, let's assume we're in the regime where the expression inside abs is positive.
        
        val_inner = 2.0 * N * c[idx_max] / (s * s)
        sign = 1.0 if val_inner >= 0 else -1.0
        
        grad = 2.0 * N * (grad_c * s - 2.0 * c[idx_max]) / (s ** 3) * sign
        
        # Normalize by s^2 to make it scale-invariant-ish
        # Actually, the objective is scale invariant, so the gradient should reflect that.
        # Let's just return the gradient of the raw function.
        
        return grad

    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Use gradient-based optimization with multiple restarts
    # The key insight is that we can use the gradient to efficiently navigate the space
    
    # Phase 1: Generate diverse initial points
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append((val_best, h_best.copy()))
    
    # 2. Structured perturbations
    for k in range(1, 30):
        for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
            if time.time() > deadline - 10:
                break
            f = np.sin(2 * np.pi * k * np.arange(N) / N)
            f = f - np.mean(f)
            h = 1.0 + eps * f
            val = compute_value(h)
            candidates.append((val, h.copy()))
    
    # 3. Random initializations
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        h = rng.normal(1.0, 1.0, N)
        val = compute_value(h)
        candidates.append((val, h.copy()))
    
    # Sort by value
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Gradient descent from the best candidates
    for val_init, h_init in candidates[:10]:
        if time.time() > deadline - 10:
            break
        
        h_cur = h_init.copy()
        val_cur = val_init
        lr = 0.1  # Learning rate
        
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient
            grad = compute_grad(h_cur)
            
            # Normalize gradient
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-12:
                break
            grad = grad / norm_grad
            
            # Take gradient step
            h_new = h_cur - lr * grad
            
            # Ensure sum is not too close to zero
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-6:
                # Project to have sum = sum(h_cur)
                h_new = h_new + (np.sum(h_cur) - s_new) / N
            
            val_new = compute_value(h_new)
            
            if val_new < val_cur:
                h_cur = h_new
                val_cur = val_new
                # Adapt learning rate
                lr = min(lr * 1.1, 1.0)
            else:
                # Decrease learning rate
                lr = max(lr * 0.8, 0.01)
                # Try a smaller step
                h_new = h_cur - lr * grad
                s_new = np.sum(h_new)
                if abs(s_new) < 1e-6:
                    h_new = h_new + (np.sum(h_cur) - s_new) / N
                val_new = compute_value(h_new)
                if val_new < val_cur:
                    h_cur = h_new
                    val_cur = val_new
        
        if val_cur < val_best:
            val_best = val_cur
            h_best = h_cur.copy()
        
        candidates.append((val_cur, h_cur.copy()))
    
    # Phase 3: Fine-tuning with random local search from the best
    if time.time() < deadline - 10:
        h_cur = h_best.copy()
        val_cur = val_best
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for _ in range(20):
                # Random perturbation
                idx = rng.integers(0, N)
                delta = rng.normal(0, 0.1)
                h_new = h_cur.copy()
                h_new[idx] += delta
                s_new = np.sum(h_new)
                if abs(s_new) < 1e-6:
                    continue
                val_new = compute_value(h_new)
                if val_new < val_cur:
                    h_cur = h_new
                    val_cur = val_new
                    improved = True
            
            if not improved:
                # Try block perturbations
                for _ in range(10):
                    start = rng.integers(0, max(1, N - 20))
                    length = rng.integers(3, 20)
                    end = min(start + length, N)
                    delta = rng.normal(0, 0.2)
                    h_new = h_cur.copy()
                    h_new[start:end] += delta
                    s_new = np.sum(h_new)
                    if abs(s_new) < 1e-6:
                        continue
                    val_new = compute_value(h_new)
                    if val_new < val_cur:
                        h_cur = h_new
                        val_cur = val_new
                        improved = True
            
            if not improved:
                break
        
        if val_cur < val_best:
            val_best = val_cur
            h_best = h_cur.copy()
    
    return h_best
