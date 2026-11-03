import numpy as np
import time

N = 600
TIME_LIMIT = 110  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def compute_conv(f):
    """Compute the autoconvolution."""
    return np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)

def optimize_coordinate_descent(f_start, max_iters=20000, time_limit=20):
    """
    Optimize using coordinate descent with pairwise moves and momentum.
    We maintain sum(f) = 1 and f >= 0.
    """
    f = f_start.copy()
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s > 1e-15:
        f /= s
    else:
        f = np.ones(N) / N
    
    current_score = compute_score(f)
    best_f = f.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Momentum for each coordinate
    momentum = np.zeros(N)
    momentum_decay = 0.9
    
    # Precompute indices for efficient updates
    # For coordinate descent, we update pairs (i, j) to maintain sum constraint
    
    iters_since_improve = 0
    lr = 0.05
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random pair of indices to update
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
        
        # Current values
        fi = f[i]
        fj = f[j]
        
        # Try to move mass from one to the other
        # We want to decrease the max convolution
        # Try several step sizes
        best_move = None
        best_move_score = current_score
        
        # Try moving mass from i to j and vice versa
        for direction in [1, -1]:
            for alpha in [lr * 0.5, lr, lr * 2, lr * 4]:
                # Move alpha * direction * (fi - fj) from i to j
                delta = alpha * direction * (fi - fj)
                if fi + delta < -1e-15 or fj - delta < -1e-15:
                    continue
                
                f_new = f.copy()
                f_new[i] = fi + delta
                f_new[j] = fj - delta
                
                # Compute score (expensive, but we can approximate)
                # For speed, only compute if the move looks promising
                new_score = compute_score(f_new)
                
                if new_score < best_move_score:
                    best_move_score = new_score
                    best_move = (i, j, delta)
        
        if best_move is not None:
            i, j, delta = best_move
            f[i] += delta
            f[j] -= delta
            current_score = best_move_score
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
                iters_since_improve = 0
            else:
                iters_since_improve += 1
            
            # Update momentum
            momentum[i] = momentum_decay * momentum[i] + (1 - momentum_decay) * delta
            momentum[j] = momentum_decay * momentum[j] - (1 - momentum_decay) * delta
            
            # Adaptive learning rate
            if iters_since_improve > 100:
                lr *= 0.95
                if lr < 1e-6:
                    lr = 1e-6
        else:
            iters_since_improve += 1
            if iters_since_improve > 500:
                lr *= 0.9
                if lr < 1e-6:
                    break
    
    return best_f, best_score

def optimize_block_descent(f_start, max_iters=5000, time_limit=20):
    """
    Optimize using block coordinate descent on small blocks.
    """
    f = f_start.copy()
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s > 1e-15:
        f /= s
    else:
        f = np.ones(N) / N
    
    current_score = compute_score(f)
    best_f = f.copy()
    best_score = current_score
    
    start_time = time.time()
    block_size = 3
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random block
        start = np.random.randint(0, N - block_size + 1)
        block_idx = np.arange(start, start + block_size)
        
        # Optimize the block while keeping others fixed
        # We want to minimize max conv subject to sum(block) = sum(f[block])
        block_sum = np.sum(f[block_idx])
        
        # Try several candidate distributions for the block
        best_block = f[block_idx].copy()
        best_block_score = current_score
        
        # Generate candidate distributions
        candidates = []
        
        # Uniform
        candidates.append(np.ones(block_size) * block_sum / block_size)
        
        # Triangular
        x = np.linspace(0, 1, block_size)
        tri = 1 - np.abs(2 * x - 1)
        candidates.append(tri * block_sum / np.sum(tri))
        
        # Peak at start
        peak = np.zeros(block_size)
        peak[0] = 1.0
        candidates.append(peak * block_sum)
        
        # Peak at end
        peak = np.zeros(block_size)
        peak[-1] = 1.0
        candidates.append(peak * block_sum)
        
        # Peak at middle
        peak = np.zeros(block_size)
        peak[block_size // 2] = 1.0
        candidates.append(peak * block_sum)
        
        # Random
        for _ in range(3):
            r = np.random.rand(block_size)
            candidates.append(r * block_sum / np.sum(r))
        
        # Evaluate each candidate
        for cand in candidates:
            if np.any(cand < -1e-15):
                continue
            cand = np.maximum(cand, 0)
            cand_sum = np.sum(cand)
            if cand_sum < 1e-15:
                continue
            cand = cand * block_sum / cand_sum
            
            f_new = f.copy()
            f_new[block_idx] = cand
            
            new_score = compute_score(f_new)
            if new_score < best_block_score:
                best_block_score = new_score
                best_block = cand
        
        f[block_idx] = best_block
        current_score = best_block_score
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
    
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Define diverse initial shapes
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N) / N)
    
    # 2. Triangular
    x = np.linspace(0, 1, N)
    tri = 1 - np.abs(2 * x - 1)
    starts.append(tri / np.sum(tri))
    
    # 3. Cosine taper
    cos = 0.5 * (1 + np.cos(2 * np.pi * x))
    starts.append(cos / np.sum(cos))
    
    # 4. Plateau (middle 50%)
    plat = np.zeros(N)
    plat[N//4 : 3*N//4] = 1.0
    starts.append(plat / np.sum(plat))
    
    # 5. Gaussian-like
    g = np.exp(-((x - 0.5)**2) / (2 * 0.1**2))
    starts.append(g / np.sum(g))
    
    # 6. Two peaks
    two_peak = np.zeros(N)
    two_peak[N//4] = 1.0
    two_peak[3*N//4] = 1.0
    starts.append(two_peak / np.sum(two_peak))
    
    # 7. Random smooth
    for _ in range(3):
        r = np.random.rand(N)
        r = np.convolve(r, np.ones(15)/15, mode='same')
        starts.append(r / np.sum(r))
        
    best_f = None
    best_score = np.inf
    
    # Phase 1: Block descent from multiple starts
    for i, start in enumerate(starts):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        remaining_time = TIME_LIMIT - 30 - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        f_opt, score = optimize_block_descent(start, max_iters=3000, time_limit=min(15, remaining_time))
        if score < best_score:
            best_score = score
            best_f = f_opt
            
    # Phase 2: Coordinate descent refinement
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 10:
        remaining_time = TIME_LIMIT - 10 - (time.time() - start_time)
        if remaining_time > 5:
            f_opt, score = optimize_coordinate_descent(best_f, max_iters=30000, time_limit=remaining_time)
            if score < best_score:
                best_score = score
                best_f = f_opt
    
    # Phase 3: Final block descent polish
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 5:
        remaining_time = TIME_LIMIT - 5 - (time.time() - start_time)
        if remaining_time > 3:
            f_opt, score = optimize_block_descent(best_f, max_iters=2000, time_limit=remaining_time)
            if score < best_score:
                best_score = score
                best_f = f_opt

    # Ensure nonnegative and return
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
