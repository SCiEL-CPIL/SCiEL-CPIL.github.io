<<<<<<< SEARCH
import time
import numpy as np
from scipy import signal

N = 50

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)"""
    h = np.array(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution
    conv = signal.convolve(h, h, mode='full')
    conv_inf = np.max(np.abs(conv))
    if conv_inf == 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def construct():
    deadline = time.time() + 100.0  # Safety margin before 120s limit
    best_h = [1.0] * N  # Seed as fallback
    best_ratio = _compute_ratio(best_h)
    
    # Try various structured configurations
    candidates = []
    
    # 1. Uniform
    candidates.append([1.0] * N)
    
    # 2. Single spike
    for pos in [0, N//2, N-1]:
        h = [0.0] * N
        h[pos] = 1.0
        candidates.append(h)
    
    # 3. Two spikes at various separations
    for sep in [1, 5, 10, 15, 20, 25, 30, 40, 49]:
        if sep < N:
            h = [0.0] * N
            h[0] = 1.0
            h[sep] = 1.0
            candidates.append(h)
    
    # 4. Block of consecutive ones
    for block_len in [1, 2, 5, 10, 20, 25, 50]:
        h = [0.0] * N
        h[:block_len] = [1.0] * block_len
        candidates.append(h)
        # Also centered
        if block_len < N:
            h2 = [0.0] * N
            start = (N - block_len) // 2
            h2[start:start+block_len] = [1.0] * block_len
            candidates.append(h2)
    
    # 5. Alternating patterns
    for period in [2, 3, 4, 5, 10]:
        h = [0.0] * N
        for i in range(N):
            if i % period == 0:
                h[i] = 1.0
        candidates.append(h)
    
    # 6. Gaussian-like
    center = N // 2
    for sigma in [1, 2, 5, 10, 20]:
        h = [0.0] * N
        for i in range(N):
            h[i] = np.exp(-0.5 * ((i - center) / sigma)**2)
        candidates.append(h)
    
    # 7. Cosine window
    for freq in [1, 2, 3, 5]:
        h = [0.0] * N
        for i in range(N):
            val = np.cos(2 * np.pi * freq * i / N)
            h[i] = max(0.0, val)
        candidates.append(h)
    
    # 8. Random configurations (limited by time)
    np.random.seed(42)
    num_random = 200
    for r in range(num_random):
        if time.time() > deadline - 30.0:
            break
        # Random sparse
        num_nonzero = np.random.randint(1, 10)
        positions = np.random.choice(N, size=num_nonzero, replace=False)
        h = [0.0] * N
        for pos in positions:
            h[pos] = np.random.uniform(0.1, 1.0)
        candidates.append(h)
        
        # Random dense
        h = list(np.random.uniform(0.0, 1.0, N))
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 5.0:
            break
        ratio = _compute_ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h
    
    return best_h
=======
import time
import numpy as np
from scipy import signal

N = 50

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)"""
    h = np.array(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution
    conv = signal.convolve(h, h, mode='full')
    conv_inf = np.max(np.abs(conv))
    if conv_inf == 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _local_optimize(h0, deadline, max_iter=5000):
    """Local optimization using coordinate ascent on weights and positions."""
    h = np.array(h0, dtype=np.float64)
    best_ratio = _compute_ratio(h)
    best_h = h.copy()
    
    # Identify non-zero positions
    nonzero_mask = h > 1e-10
    positions = np.where(nonzero_mask)[0]
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        
        # Try adjusting weights
        for i in positions:
            old_val = h[i]
            if old_val == 0:
                continue
            
            # Try small perturbations
            for factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 0.0]:
                new_h = h.copy()
                new_h[i] = old_val * factor
                if np.any(new_h < -1e-10):
                    continue
                ratio = _compute_ratio(new_h)
                if ratio > best_ratio + 1e-12:
                    best_ratio = ratio
                    best_h = new_h.copy()
                    h = new_h
                    improved = True
        
        # Try moving positions
        for i in positions:
            old_val = h[i]
            if old_val == 0:
                continue
            
            for delta in [-2, -1, 1, 2]:
                new_pos = i + delta
                if new_pos < 0 or new_pos >= N:
                    continue
                if new_pos in positions and new_pos != i:
                    continue
                
                new_h = h.copy()
                new_h[i] = 0
                new_h[new_pos] = old_val
                ratio = _compute_ratio(new_h)
                if ratio > best_ratio + 1e-12:
                    best_ratio = ratio
                    best_h = new_h.copy()
                    h = new_h
                    # Update positions
                    positions = np.where(h > 1e-10)[0]
                    improved = True
        
        if not improved:
            # Try adding a new non-zero element
            best_new_ratio = best_ratio
            best_new_h = None
            best_new_pos = -1
            
            for new_pos in range(N):
                if new_pos in positions:
                    continue
                new_h = h.copy()
                new_h[new_pos] = 0.5 * np.mean(h[h > 1e-10]) if np.any(h > 1e-10) else 1.0
                ratio = _compute_ratio(new_h)
                if ratio > best_new_ratio + 1e-12:
                    best_new_ratio = ratio
                    best_new_h = new_h.copy()
                    best_new_pos = new_pos
            
            if best_new_h is not None:
                best_ratio = best_new_ratio
                best_h = best_new_h
                h = best_new_h
                positions = np.where(h > 1e-10)[0]
                improved = True
            
            if not improved:
                break
    
    return best_h, best_ratio

def construct():
    deadline = time.time() + 110.0  # Safety margin before 150s limit
    best_h = [1.0] * N  # Seed as fallback
    best_ratio = _compute_ratio(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Two-point clusters with different weight ratios
    for sep in range(1, N):
        for weight_ratio in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
            h = [0.0] * N
            h[0] = 1.0
            h[sep] = weight_ratio
            candidates.append(h)
    
    # 2. Three-point clusters
    for sep1 in [1, 2, 5, 10]:
        for sep2 in [1, 2, 5, 10]:
            if sep1 + sep2 < N:
                h = [0.0] * N
                h[0] = 1.0
                h[sep1] = 0.8
                h[sep1 + sep2] = 0.6
                candidates.append(h)
    
    # 3. Small blocks with varying weights
    for block_len in [2, 3, 4, 5]:
        for weight_pattern in [
            [1.0, 0.5],
            [1.0, 0.7, 0.3],
            [1.0, 0.8, 0.6, 0.4],
            [1.0, 0.9, 0.8, 0.7, 0.6],
        ]:
            if len(weight_pattern) == block_len:
                h = [0.0] * N
                for i, w in enumerate(weight_pattern):
                    h[i] = w
                candidates.append(h)
    
    # 4. Sparse random configurations
    np.random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 60.0:
            break
        num_nonzero = np.random.randint(2, 8)
        positions = np.random.choice(N, size=num_nonzero, replace=False)
        h = [0.0] * N
        for pos in positions:
            h[pos] = np.random.uniform(0.1, 2.0)
        candidates.append(h)
    
    # Evaluate candidates and keep top ones for local optimization
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 50.0:
            break
        ratio = _compute_ratio(h)
        scored_candidates.append((ratio, h))
    
    # Sort by ratio and take top candidates
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Local optimize top candidates
    top_k = min(20, len(scored_candidates))
    for ratio, h in scored_candidates[:top_k]:
        if time.time() > deadline - 15.0:
            break
        if ratio <= best_ratio:
            continue
        optimized_h, opt_ratio = _local_optimize(h, deadline)
        if opt_ratio > best_ratio:
            best_ratio = opt_ratio
            best_h = optimized_h
    
    return best_h.tolist()
>>>>>>> REPLACE
