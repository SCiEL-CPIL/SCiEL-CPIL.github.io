"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all circular shifts."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    ones = h.sum()
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, -k)
        prod = (h * shifted).sum()
        ov = ones - prod
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _build_h_from_pattern(pattern, N):
    """Tile a base pattern to length N, ensuring sum = N/2."""
    p = np.array(pattern, dtype=np.float64)
    plen = len(p)
    reps = N // plen
    rem = N % plen
    h = np.tile(p, reps)
    if rem > 0:
        h = np.concatenate([h, p[:rem]])
    # Adjust to make sum exactly N/2
    target = N / 2.0
    current_sum = h.sum()
    diff = target - current_sum
    if abs(diff) > 1e-9:
        if diff > 0:
            zeros = np.where(h == 0)[0]
            for i in range(int(round(diff))):
                if i < len(zeros):
                    h[zeros[i]] = 1.0
        else:
            ones = np.where(h == 1)[0]
            for i in range(int(round(-diff))):
                if i < len(ones):
                    h[ones[i]] = 0.0
    return h

def _try_pattern(pattern, N, deadline):
    """Try a binary pattern, return (score, h) or None if invalid/too slow."""
    if time.time() > deadline:
        return None
    p = np.array(pattern, dtype=np.float64)
    plen = len(p)
    if sum(pattern) != plen // 2:
        return None
    h = _build_h_from_pattern(pattern, N)
    if abs(h.sum() - N / 2.0) > 1e-9:
        return None
    score = _compute_max_overlap(h)
    return score, h

def _local_search(h, N, deadline, max_iters=2000):
    """Local search: flip pairs of bits (one 0->1, one 1->0) to maintain sum constraint."""
    h = h.copy()
    current_score = _compute_max_overlap(h)
    best_score = current_score
    best_h = h.copy()
    
    ones_indices = np.where(h == 1)[0]
    zeros_indices = np.where(h == 0)[0]
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try flipping a 1 to 0 and a 0 to 1
        # Random sampling of pairs
        for _ in range(20):
            if time.time() > deadline:
                break
            if len(ones_indices) == 0 or len(zeros_indices) == 0:
                break
            i = ones_indices[np.random.randint(len(ones_indices))]
            j = zeros_indices[np.random.randint(len(zeros_indices))]
            if i == j:
                continue
            
            h[i] = 0.0
            h[j] = 1.0
            new_score = _compute_max_overlap(h)
            
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_h = h.copy()
                improved = True
                # Update index sets
                ones_indices = np.where(h == 1)[0]
                zeros_indices = np.where(h == 0)[0]
            else:
                h[i] = 1.0
                h[j] = 0.0
        
        if not improved and iteration > 100:
            # Try more aggressive: accept equal scores to escape plateaus
            accepted_equal = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                if len(ones_indices) == 0 or len(zeros_indices) == 0:
                    break
                i = ones_indices[np.random.randint(len(ones_indices))]
                j = zeros_indices[np.random.randint(len(zeros_indices))]
                if i == j:
                    continue
                
                h[i] = 0.0
                h[j] = 1.0
                new_score = _compute_max_overlap(h)
                
                if new_score <= best_score + 1e-12:
                    if new_score < best_score - 1e-12:
                        best_score = new_score
                        best_h = h.copy()
                    accepted_equal = True
                    ones_indices = np.where(h == 1)[0]
                    zeros_indices = np.where(h == 0)[0]
                else:
                    h[i] = 1.0
                    h[j] = 0.0
            
            if not accepted_equal:
                break
    
    return best_score, best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Try structured binary patterns
    candidates = []
    
    # Alternating
    candidates.append([1, 0])
    
    # Block patterns
    for block in [2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 50]:
        candidates.append([1] * block + [0] * block)
    
    # M-sequence-like patterns for small periods
    # Period 6 with 3 ones
    for combo in product([0, 1], repeat=6):
        if sum(combo) == 3:
            candidates.append(list(combo))
    
    # Period 8 with 4 ones - only a subset to keep time bounded
    # Use known good patterns
    candidates.append([1, 1, 1, 0, 1, 0, 0, 0])
    candidates.append([1, 1, 0, 1, 1, 0, 0, 0])
    candidates.append([1, 0, 1, 1, 0, 1, 0, 0])
    candidates.append([1, 1, 0, 0, 1, 1, 0, 0])
    candidates.append([1, 0, 0, 1, 0, 0, 1, 0])
    
    # Period 10 with 5 ones
    candidates.append([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    candidates.append([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    candidates.append([1, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    
    # Period 12 with 6 ones
    candidates.append([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    
    # Period 14 with 7 ones
    candidates.append([1] * 7 + [0] * 7)
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    
    # Period 16 with 8 ones
    candidates.append([1] * 8 + [0] * 8)
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    
    # Period 20 with 10 ones
    candidates.append([1] * 10 + [0] * 10)
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    
    # Period 25 with 12.5 ones - not integer, skip
    # Period 50 with 25 ones
    candidates.append([1] * 25 + [0] * 25)
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    for pat in candidates:
        if time.time() > deadline:
            break
        result = _try_pattern(pat, N, deadline)
        if result is not None:
            score, h = result
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Phase 2: Local search from best candidate
    if best_h is not None and time.time() < deadline - 10:
        ls_score, ls_h = _local_search(best_h, N, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_h = ls_h
    
    # Phase 3: Try more random initializations with local search
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Random binary pattern with sum N/2
            h_rand = np.zeros(N)
            ones_pos = np.random.choice(N, N // 2, replace=False)
            h_rand[ones_pos] = 1.0
            score, h_ls = _local_search(h_rand, N, deadline - 5)
            if score < best_score:
                best_score = score
                best_h = h_ls
    
    if best_h is None:
        # Fallback: alternating
        best_h = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    
    return best_h
# EVOLVE-BLOCK-END
