"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from collections import Counter

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _doubled_areas(points):
    """Return array of doubled absolute areas for all non-degenerate triples."""
    n = len(points)
    p = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = (p[j][0]-p[i][0])*(p[k][1]-p[i][1]) - (p[k][0]-p[i][0])*(p[j][1]-p[i][1])
                a = abs(det)
                if a > 0:
                    areas.append(a)
    return np.array(areas)

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    p = np.array(points, dtype=np.int64)
    count = Counter()
    for i in range(n):
        pi = p[i]
        for j in range(i+1, n):
            pj = p[j]
            dx_ij = pj[0] - pi[0]
            dy_ij = pj[1] - pi[1]
            for k in range(j+1, n):
                pk = p[k]
                det = dx_ij * (pk[1] - pi[1]) - (pk[0] - pi[0]) * dy_ij
                a = abs(det)
                if a > 0:
                    count[a] += 1
    if not count:
        return 0
    return max(count.values())

def _score_fast(points):
    """Vectorized score computation."""
    n = len(points)
    if n < 3:
        return 0
    p = np.array(points, dtype=np.int64)
    # Compute all triple determinants
    idx = np.arange(n)
    # Use broadcasting for efficiency
    # p_i = p[:, None, :], p_j = p[None, :, :], p_k = p[None, None, :]
    # det = (p_j - p_i) x (p_k - p_i)
    pi = p[:, None, None, :]  # (n,1,1,2)
    pj = p[None, :, None, :]  # (1,n,1,2)
    pk = p[None, None, :, :]  # (1,1,n,2)
    
    # Only consider i<j<k
    # det = (pj-pi)[0]*(pk-pi)[1] - (pj-pi)[1]*(pk-pi)[0]
    det = (pj[:,:,0] - pi[:,:,0]) * (pk[:,:,1] - pi[:,:,1]) - (pj[:,:,1] - pi[:,:,1]) * (pk[:,:,0] - pi[:,:,0])
    
    # Mask: i < j < k
    mask = np.zeros((n,n,n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask[i,j,k] = True
    
    abs_det = np.abs(det)
    valid = mask & (abs_det > 0)
    valid_areas = abs_det[valid]
    
    if len(valid_areas) == 0:
        return 0
    
    # Count multiplicities
    vals, counts = np.unique(valid_areas, return_counts=True)
    return int(counts.max())

def construct():
    deadline = 85.0  # safety margin before 90s kill
    
    # Start with a good initial configuration
    # A 6x5 grid gives some structure
    initial = [(x, y) for x in range(6) for y in range(5)]
    best = list(initial)
    best_score = _score_fast(best)
    
    if _elapsed() > deadline - 5:
        return best
    
    # Local search: try moving individual points
    import random
    random.seed(42)
    
    # Precompute candidate positions
    all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Iterative local search with random restarts
    current = list(best)
    current_score = best_score
    
    while _elapsed() < deadline - 5:
        # Try random single-point moves
        improved = False
        for _ in range(200):
            if _elapsed() > deadline - 5:
                break
            
            idx = random.randint(0, len(current) - 1)
            old_pos = current[idx]
            
            # Try a random new position
            new_pos = random.choice(all_positions)
            if new_pos == old_pos:
                continue
            
            # Check if position is already used
            if new_pos in current:
                continue
            
            candidate = list(current)
            candidate[idx] = new_pos
            
            # Quick check: ensure all distinct
            if len(set(candidate)) != len(candidate):
                continue
            
            cand_score = _score_fast(candidate)
            
            if cand_score > current_score:
                current = candidate
                current_score = cand_score
                improved = True
                if current_score > best_score:
                    best = list(current)
                    best_score = current_score
        
        # If no improvement in this round, try a random restart
        if not improved:
            # Perturb a few points randomly
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, len(current) - 1)
                new_pos = random.choice(all_positions)
                if new_pos not in current:
                    current[idx] = new_pos
            
            current_score = _score_fast(current)
            if current_score > best_score:
                best = list(current)
                best_score = current_score
    
    return best

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
