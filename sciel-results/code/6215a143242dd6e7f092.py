"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

DEADLINE = 325  # seconds, safety margin before 330s kill

# Precomputation
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}
NUM_POINTS = len(POINTS)
ALL_MASK = (1 << NUM_POINTS) - 1

# Precompute squared distances
xs = np.array([p[0] for p in POINTS])
ys = np.array([p[1] for p in POINTS])
dx = xs[:, np.newaxis] - xs[np.newaxis, :]
dy = ys[:, np.newaxis] - ys[np.newaxis, :]
DIST = (dx * dx + dy * dy).astype(np.int32)

# Precompute ISO[u][v] bitmask
# ISO[u][v] has bit k set if (u, v, k) forms an isosceles triangle
# u, v, k must be distinct.
ISO = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

# Helper to check isosceles
def _is_isosceles(d_uv, d_uw, d_vw):
    # d_uv, d_uw, d_vw are squared distances
    return d_uv == d_uw or d_uv == d_vw or d_uw == d_vw

for u in range(NUM_POINTS):
    for v in range(u + 1, NUM_POINTS):
        d_uv = DIST[u, v]
        mask = 0
        for k in range(NUM_POINTS):
            if k == u or k == v:
                continue
            d_uk = DIST[u, k]
            d_vk = DIST[v, k]
            if _is_isosceles(d_uv, d_uk, d_vk):
                mask |= (1 << k)
        ISO[u][v] = mask
        ISO[v][u] = mask

def _compute_bad(S):
    """Compute the bitmask of points that cannot be added to S."""
    bad = 0
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            bad |= ISO[S[i]][S[j]]
    return bad

def _is_maximal(S, bad_mask):
    """Check if S is maximal. S is maximal if no point outside S can be added."""
    # Points that can be added are those not in S and not in bad_mask
    # If any such point exists, it's not maximal.
    S_mask = 0
    for idx in S:
        S_mask |= (1 << idx)
    
    # Candidates for addition: NOT in S, NOT in bad_mask
    addable = ALL_MASK & ~S_mask & ~bad_mask
    return addable == 0

def _greedy_random(rng):
    """Randomized greedy: pick points in random order, add if possible."""
    order = list(range(NUM_POINTS))
    rng.shuffle(order)
    S = []
    bad_mask = 0
    S_mask = 0
    
    for p_idx in order:
        # Check if p_idx can be added
        if (bad_mask >> p_idx) & 1:
            continue
        if (S_mask >> p_idx) & 1:
            continue
            
        # Add p_idx
        S.append(p_idx)
        S_mask |= (1 << p_idx)
        
        # Update bad_mask
        # New conflicts are (p_idx, u) for all u in S (excluding p_idx itself, but p_idx was just added, so u in old S)
        # Wait, S was updated before this loop?
        # Let's use the list S.
        # When adding p, new pairs are (p, u) for u in S_before.
        # Since we append p, we can iterate over S[:-1]
        for u in S[:-1]:
            bad_mask |= ISO[p_idx][u]
            
    return S

def _local_search(S, rng, max_iters=1000):
    """Try to remove points while maintaining maximality."""
    S = list(S)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        indices = list(range(len(S)))
        rng.shuffle(indices)
        
        for idx in indices:
            p = S[idx]
            candidate = S[:idx] + S[idx+1:]
            
            # Compute bad for candidate
            bad_mask = _compute_bad(candidate)
            
            if _is_maximal(candidate, bad_mask):
                S = candidate
                improved = True
                break
                
    return S

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Multiple randomized greedy starts
    while time.time() - start_time < DEADLINE * 0.2:
        candidate = _greedy_random(rng)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
            
    # Phase 2: Local search on best and near-best candidates
    if best is not None:
        # Keep a few good candidates for diversity
        candidates = [best]
        
        while time.time() - start_time < DEADLINE * 0.9:
            # Try removal on current best
            improved = _local_search(best, rng, max_iters=200)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
            
            # Occasionally try a fresh random start + local search
            if time.time() - start_time < DEADLINE * 0.6:
                candidate = _greedy_random(rng)
                candidate = _local_search(candidate, rng, max_iters=100)
                if len(candidate) < best_size:
                    best = candidate
                    best_size = len(candidate)
                    candidates = [best]
                elif len(candidate) <= best_size + 2:
                    candidates.append(candidate)
                    if len(candidates) > 5:
                        candidates.pop()
            
            # Try local search on other good candidates
            if len(candidates) > 1 and time.time() - start_time < DEADLINE * 0.8:
                for c in candidates[1:]:
                    if len(c) <= best_size + 2:
                        improved_c = _local_search(c, rng, max_iters=50)
                        if len(improved_c) < best_size:
                            best = improved_c
                            best_size = len(best)
                            candidates = [best]
                            break
    
    # Final: ensure maximality of best result
    if best is not None and time.time() - start_time < DEADLINE - 5:
        # _local_search ensures maximality if it accepts a move.
        # But if it didn't find a move, it might not be maximal?
        # No, _local_search only accepts if _is_maximal is True.
        # So if it returns, it is maximal.
        # However, if best was from greedy, it might not be maximal?
        # Greedy adds if possible. It stops when no more can be added.
        # So greedy result is maximal.
        # So best is always maximal.
        
        pass

    # Convert indices back to (x, y) tuples
    return [POINTS[i] for i in best] if best is not None else []
# EVOLVE-BLOCK-END
