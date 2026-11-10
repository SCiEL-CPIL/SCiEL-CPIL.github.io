# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Precompute forbidden differences and conflict masks once at module load.
_M = 205
_FORBIDDEN = set()
for _x in range(1, _M):
    _FORBIDDEN.add((_x * _x) % _M)
_FORBIDDEN.discard(0)

# conflict_mask[i] is a bitmask (as a Python int with _M bits) where bit j is 1
# iff (i - j) mod _M is in _FORBIDDEN (i.e., i and j conflict).
_CONFLICT_MASK = [0] * _M
for _i in range(_M):
    _mask = 0
    for _j in range(_M):
        if _i == _j:
            continue
        if (_i - _j) % _M in _FORBIDDEN:
            _mask |= (1 << _j)
    _CONFLICT_MASK[_i] = _mask

_ALL_MASK = (1 << _M) - 1


def _is_valid(selected_mask: int) -> bool:
    """Check that no two selected residues conflict."""
    m = selected_mask
    while m:
        low = m & (-m)
        i = low.bit_length() - 1
        m ^= low
        if (_CONFLICT_MASK[i] & m) != 0:
            return False
    return True


def _greedy_random(rng: random.Random) -> int:
    """Random greedy: build a valid set by adding random residues that fit."""
    order = list(range(_M))
    rng.shuffle(order)
    mask = 0
    for cand in order:
        if (_CONFLICT_MASK[cand] & mask) == 0:
            mask |= (1 << cand)
    return mask


def _branch_and_bound_helper(candidates: int, current_mask: int, current_size: int, best_size_ref: list, best_mask_ref: list, deadline: float):
    if time.time() > deadline:
        return

    if candidates == 0:
        if current_size > best_size_ref[0]:
            best_size_ref[0] = current_size
            best_mask_ref[0] = current_mask
        return

    # Quick bound: popcount
    if current_size + bin(candidates).count('1') <= best_size_ref[0]:
        return

    # Greedy Coloring Upper Bound
    verts = []
    temp = candidates
    while temp:
        low = temp & (-temp)
        i = low.bit_length() - 1
        verts.append(i)
        temp ^= low
    
    if not verts:
        return

    # Calculate degrees
    degrees = {}
    for v in verts:
        degrees[v] = bin(_CONFLICT_MASK[v] & candidates).count('1')
    
    # Sort by degree descending
    sorted_verts = sorted(verts, key=lambda x: degrees[x], reverse=True)
    
    # Greedy Coloring
    color_of = {}
    max_color = 0
    
    for v in sorted_verts:
        used_colors = set()
        neighbors_mask = _CONFLICT_MASK[v] & candidates
        temp_n = neighbors_mask
        while temp_n:
            low_n = temp_n & (-temp_n)
            u = low_n.bit_length() - 1
            temp_n ^= low_n
            if u in color_of:
                used_colors.add(color_of[u])
        
        c = 1
        while c in used_colors:
            c += 1
        
        color_of[v] = c
        if c > max_color:
            max_color = c
        
        if current_size + max_color <= best_size_ref[0]:
            return

    # Branching
    # Pick vertex with highest degree
    v = sorted_verts[0]
    
    # Branch 1: Include v
    # Remove v and its neighbors from candidates
    new_candidates = candidates & ~(_CONFLICT_MASK[v] | (1 << v))
    new_mask = current_mask | (1 << v)
    new_size = current_size + 1
    
    _branch_and_bound_helper(new_candidates, new_mask, new_size, best_size_ref, best_mask_ref, deadline)
    
    # Branch 2: Exclude v
    new_candidates2 = candidates & ~(1 << v)
    _branch_and_bound_helper(new_candidates2, current_mask, current_size, best_size_ref, best_mask_ref, deadline)


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero
    square modulo 205. Uses branch-and-bound for Maximum Independent Set.
    """
    start = time.time()
    deadline = start + 540.0

    # Initial greedy solution to provide a good lower bound
    best_size = 0
    best_mask = 0
    
    # Deterministic greedy
    mask = 0
    for cand in range(_M):
        if (_CONFLICT_MASK[cand] & mask) == 0:
            mask |= (1 << cand)
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # Random restarts to improve initial bound
    rng = random.Random(20240520)
    for _ in range(100): # 100 quick greedy restarts
        m = _greedy_random(rng)
        sz = bin(m).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Branch and Bound
    # Start with all vertices
    candidates = _ALL_MASK
    # We don't have a current mask yet, we are building one from scratch.
    # So current_mask = 0, current_size = 0.
    
    best_size_ref = [best_size]
    best_mask_ref = [best_mask]
    
    # We need to ensure we don't re-add vertices that are already in best_mask?
    # No, B&B explores the space of ALL independent sets.
    # The initial best_size is just a lower bound for pruning.
    
    # However, B&B from scratch might be slow.
    # We can seed it by including the vertices from the best greedy solution?
    # If we force them to be included, we might miss a better solution that excludes some.
    # So we run B&B from scratch.
    
    _branch_and_bound_helper(candidates, 0, 0, best_size_ref, best_mask_ref, deadline)
    
    # Extract residues
    residues = []
    m = best_mask_ref[0]
    while m:
        low = m & (-m)
        i = low.bit_length() - 1
        m ^= low
        residues.append(i)
    residues.sort()
    
    # Validity check
    if not residues:
        return np.array([0], dtype=np.int64)
        
    sel = set(residues)
    for a in residues:
        for b in residues:
            if a == b:
                continue
            if (a - b) % _M in _FORBIDDEN:
                return np.array([0], dtype=np.int64)
                
    return np.array(residues, dtype=np.int64)

# EVOLVE-BLOCK-END
