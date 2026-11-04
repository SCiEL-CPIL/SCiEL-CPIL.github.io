"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlaps(h):
    """Compute overlap at each cyclic shift for step function h."""
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        # overlap(k) = sum_i h[i] * (1 - h[(i+k) % n])
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap over all cyclic shifts."""
    return float(np.max(_compute_overlaps(h)))

def construct():
    deadline = time.time() + 120.0  # safety margin: finish before 150s kill
    
    # Start with a binary pattern: alternating 1,0 gives good spread
    # But we need exactly N/2 ones. Try a "maximally spread" initial pattern.
    # Use a greedy construction: place ones as far apart as possible.
    
    best_h = np.zeros(N)
    best_score = float('inf')
    
    # Initial: use a pattern based on golden ratio spacing for good distribution
    # Place N/2 ones at positions floor(i * N / (N/2)) = floor(2*i) for i in 0..N/2-1
    # That's just even indices: 0, 2, 4, ...
    # Better: use a "balanced" sequence
    
    # Try several initial patterns
    initial_patterns = []
    
    # Pattern 1: alternating 1,0,1,0,...
    p1 = np.zeros(N)
    p1[::2] = 1.0
    initial_patterns.append(p1.copy())
    
    # Pattern 2: blocks of 1,1,0,0,1,1,0,0,...
    p2 = np.zeros(N)
    for i in range(N):
        if (i // 2) % 2 == 0:
            p2[i] = 1.0
    initial_patterns.append(p2.copy())
    
    # Pattern 3: use a low-discrepancy sequence (van der Corput style)
    p3 = np.zeros(N)
    n_ones = N // 2
    # Place ones at positions that maximize minimum distance
    # Use greedy: start with one at 0, then add next one at farthest point
    positions = [0]
    for _ in range(n_ones - 1):
        # Find position with max min-distance to existing positions (circular)
        best_pos = 0
        best_min_dist = -1
        for pos in range(N):
            if pos in positions:
                continue
            min_dist = min(min(abs(pos - p), N - abs(pos - p)) for p in positions)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos
        positions.append(best_pos)
    for pos in positions:
        p3[pos] = 1.0
    initial_patterns.append(p3.copy())
    
    # Pattern 4: m-sequence-like (linear feedback shift register pattern)
    # For N=100, not a power of 2, so approximate
    p4 = np.zeros(N)
    # Use a simple periodic pattern with period 5: 1,1,1,0,0 repeated
    for i in range(N):
        if i % 5 < 3:
            p4[i] = 1.0
    initial_patterns.append(p4.copy())
    
    best_h = None
    best_score = float('inf')
    
    for pat in initial_patterns:
        score = _max_overlap(pat)
        if score < best_score:
            best_score = score
            best_h = pat.copy()
    
    # Local search: flip bits to reduce max overlap
    # At each step, try flipping each bit and keep if it improves
    h = best_h.copy()
    current_score = best_score
    
    # Precompute overlaps for efficiency
    current_overlaps = _compute_overlaps(h)
    
    max_iter = 500  # number of full passes
    for iteration in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            
            # Try flipping bit i
            old_val = h[i]
            new_val = 1.0 - old_val
            h[i] = new_val
            
            new_overlaps = _compute_overlaps(h)
            new_score = float(np.max(new_overlaps))
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
                current_overlaps = new_overlaps
            else:
                h[i] = old_val
        
        if not improved:
            # Try random restarts from perturbed versions
            for restart in range(10):
                if time.time() > deadline - 5.0:
                    break
                h_perturbed = h.copy()
                # Flip 5 random bits
                flip_indices = np.random.choice(N, size=5, replace=False)
                h_perturbed[flip_indices] = 1.0 - h_perturbed[flip_indices]
                
                # Do a few local search steps from here
                h_trial = h_perturbed.copy()
                trial_score = _max_overlap(h_trial)
                for _ in range(50):
                    if time.time() > deadline - 5.0:
                        break
                    trial_improved = False
                    for j in range(N):
                        old = h_trial[j]
                        h_trial[j] = 1.0 - old
                        s = _max_overlap(h_trial)
                        if s < trial_score - 1e-12:
                            trial_score = s
                            trial_improved = True
                        else:
                            h_trial[j] = old
                    if not trial_improved:
                        break
                
                if trial_score < current_score:
                    h = h_trial.copy()
                    current_score = trial_score
                    improved = True
                    break
    
    # Final: try to smooth the solution - allow non-binary values
    # Use gradient-based refinement on continuous h
    h_continuous = h.astype(float)
    lr = 0.05
    for step in range(2000):
        if time.time() > deadline - 2.0:
            break
        
        # Compute overlaps
        overlaps = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h_continuous, k)
            overlaps[k] = np.sum(h_continuous * (1.0 - shifted))
        
        # The objective is max(overlaps), use soft-max for gradient
        beta = 50.0
        log_overlaps = np.log(np.maximum(overlaps, 1e-10))
        weights = np.exp(beta * log_overlaps)
        weights /= np.sum(weights)
        
        # Gradient of soft-max(overlaps) w.r.t. h
        # d(overlap_k)/d(h_i) = (1 - h_{i+k}) - h_i * delta_{i, i+k}... 
        # Actually overlap_k = sum_i h_i (1 - h_{i+k})
        # d(overlap_k)/d(h_j) = (1 - h_{j+k}) - h_{j-k}
        grad = np.zeros(N)
        for k in range(N):
            w = weights[k]
            if w < 1e-10:
                continue
            # d(overlap_k)/d(h_j) = (1 - h_{(j+k)%N}) - h_{(j-k)%N}
            for j in range(N):
                grad[j] += w * ((1.0 - h_continuous[(j + k) % N]) - h_continuous[(j - k) % N])
        
        # Update
        h_continuous -= lr * grad
        # Project to [0, 1]
        h_continuous = np.clip(h_continuous, 0.0, 1.0)
        # Enforce sum constraint: sum should be N/2
        current_sum = np.sum(h_continuous)
        if abs(current_sum - N / 2.0) > 1e-10:
            # Adjust proportionally
            h_continuous = h_continuous * (N / 2.0) / current_sum
            h_continuous = np.clip(h_continuous, 0.0, 1.0)
    
    # Final check
    final_score = _max_overlap(h_continuous)
    if final_score < current_score:
        h = h_continuous
    
    # Ensure exact sum constraint
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    if abs(current_sum - N / 2.0) > 1e-10:
        h = h * (N / 2.0) / current_sum
        h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END
