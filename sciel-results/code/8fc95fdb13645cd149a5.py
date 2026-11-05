"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Structural approach:
    1. Normalize sum(h) = N to remove scale invariance
    2. Use projected gradient with explicit active-set tracking
    3. Exploit that min max_autocorr with fixed sum is a linear program in disguise
    4. Coordinate descent on the active autocorrelation component
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    def normalize_sum(h, target=N):
        """Scale h so that sum(h) = target"""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return h
        return h * target / s
    
    # Evaluate seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    rng = np.random.RandomState(42)
    
    def active_set_optimize(h0, max_iters=2000):
        """
        Optimize by tracking which lag gives the max autocorrelation.
        For a fixed active lag k, minimize conv[k] subject to sum(h)=N.
        This is a quadratic program with linear constraint.
        Use projected gradient with exact line search.
        """
        h = normalize_sum(h0.copy())
        best_local = evaluate(h)[0]
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            conv = np.convolve(h, h)
            max_lag = np.argmax(conv)
            max_val = conv[max_lag]
            
            # Gradient of conv[max_lag] w.r.t. h
            # conv[k] = sum_i h[i]*h[i+k]
            # d(conv[k])/d(h[j]) = h[j+k] + h[j-k] (with bounds)
            grad = np.zeros(N)
            for j in range(N):
                # Contribution from h[j] in conv[max_lag]
                # h[j] appears in conv[max_lag] when:
                # - as first factor: i=j, need j+max_lag < N, term: h[j+max_lag]
                # - as second factor: i+k=j, need i>=0, term: h[i] = h[j-max_lag]
                contrib = 0.0
                if j + max_lag < N:
                    contrib += h[j + max_lag]
                if j - max_lag >= 0:
                    contrib += h[j - max_lag]
                grad[j] = contrib
            
            # Projected gradient: minimize grad^T h subject to sum(h)=N
            # Use: h_new = h - lr * (grad - mean(grad))
            # The -mean(grad) ensures sum is preserved
            grad_centered = grad - np.mean(grad)
            
            # Adaptive learning rate
            lr = 1.0 / (np.max(np.abs(grad_centered)) + 1e-10)
            
            h_new = h - lr * grad_centered
            h_new = normalize_sum(h_new)
            
            val_new, valid_new = evaluate(h_new)
            if valid_new and val_new < best_local:
                h = h_new
                best_local = val_new
                if best_local < 0.7:
                    break
            else:
                # Try smaller step
                lr *= 0.5
                h_new = h - lr * grad_centered
                h_new = normalize_sum(h_new)
                val_new, valid_new = evaluate(h_new)
                if valid_new and val_new < best_local:
                    h = h_new
                    best_local = val_new
                else:
                    # Stagnation, break
                    break
        
        return h, best_local
    
    # Generate diverse initial points
    init_points = []
    
    # Seed
    init_points.append(np.ones(N).astype(float))
    
    # Random with various distributions
    for _ in range(100):
        if time.time() - start_time > DEADLINE:
            break
        h = rng.standard_normal(N)
        init_points.append(h)
    
    # Structured: alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    init_points.append(h_alt)
    
    # Structured: blocks
    for n_blocks in [2, 3, 4, 5, 6, 8]:
        block_len = N // n_blocks
        h_block = np.zeros(N)
        for b in range(n_blocks):
            sign = 1.0 if b % 2 == 0 else -1.0
            h_block[b*block_len:(b+1)*block_len] = sign
        init_points.append(h_block)
    
    # Structured: sinusoidal
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        init_points.append(np.cos(2 * np.pi * freq * t))
        init_points.append(np.sin(2 * np.pi * freq * t))
    
    # Structured: triangular wave
    for period in [N//4, N//3, N//2, N//3*2]:
        h_tri = np.zeros(N)
        for i in range(N):
            phase = (i % period) / period
            if phase < 0.5:
                h_tri[i] = 1.0
            else:
                h_tri[i] = -1.0
        init_points.append(h_tri)
    
    # Random block structures
    for _ in range(50):
        if time.time() - start_time > DEADLINE:
            break
        n_blocks = rng.randint(2, 10)
        h_rb = np.zeros(N)
        pos = 0
        for _ in range(n_blocks):
            length = max(1, (N - pos) // (n_blocks - 1) if n_blocks > 1 else N - pos)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            h_rb[pos:pos+length] = sign
            pos += length
        if pos < N:
            h_rb[pos:] = 1.0 if rng.random() < 0.5 else -1.0
        init_points.append(h_rb)
    
    # Evaluate all and sort
    evals = []
    for h in init_points:
        if time.time() - start_time > DEADLINE:
            break
        val, valid = evaluate(h)
        if valid:
            evals.append((val, h))
    
    evals.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    n_optimize = min(20, len(evals))
    for idx in range(n_optimize):
        if time.time() - start_time > DEADLINE:
            break
        
        h0 = evals[idx][1].copy()
        h_opt, val_opt = active_set_optimize(h0, max_iters=3000)
        
        if val_opt < best_value:
            best_value = val_opt
            best_h = h_opt.copy()
        
        # Also try from the optimized point with different random restarts
        for _ in range(3):
            if time.time() - start_time > DEADLINE:
                break
            h_perturbed = h_opt + 0.1 * rng.standard_normal(N)
            h_perturbed = normalize_sum(h_perturbed)
            h_opt2, val_opt2 = active_set_optimize(h_perturbed, max_iters=1500)
            if val_opt2 < best_value:
                best_value = val_opt2
                best_h = h_opt2.copy()
    
    # Final coordinate descent polish
    if time.time() - start_time < DEADLINE:
        h_curr = best_h.copy()
        h_curr = normalize_sum(h_curr)
        
        for outer in range(100):
            if time.time() - start_time > DEADLINE:
                break
            
            conv = np.convolve(h_curr, h_curr)
            max_lag = np.argmax(conv)
            
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Compute gradient contribution for this coordinate
                # d(conv[max_lag])/d(h[coord])
                contrib = 0.0
                if coord + max_lag < N:
                    contrib += h_curr[coord + max_lag]
                if coord - max_lag >= 0:
                    contrib += h_curr[coord - max_lag]
                
                # Try moving in the direction that decreases conv[max_lag]
                # while maintaining sum
                # Moving h[coord] by delta changes sum by delta, so we need to adjust others
                # Simple: move h[coord] and scale the rest to maintain sum
                
                for delta in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    # Rescale to maintain sum
                    s = np.sum(h_try)
                    if abs(s) > 1e-12:
                        h_try = h_try * N / s
                    
                    val_try, valid_try = evaluate(h_try)
                    if valid_try and val_try < best_value:
                        best_value = val_try
                        best_h = h_try.copy()
                        h_curr = h_try
                        improved = True
                        break
                
                if best_value < 0.7:
                    break
            
            if best_value < 0.7:
                break
            if not improved:
                break
    
    return best_h
# EVOLVE-BLOCK-END
