"""Multi-Start Local Search with Perturb-and-Repair for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
import itertools

# Precomputation cache
_Q = 3
_N = 8
_RADIUS = 1
_TOTAL = _Q ** _N
_WEIGHT = None
_WORDS = None
_NEIGHBORS = None
# Reverse neighbors: for each point, which codewords cover it?
# This is the same as _NEIGHBORS since distance is symmetric.
# So _NEIGHBORS[i] is the set of points covered by codeword i.
# And for a point p, the codewords covering it are _NEIGHBORS[p].

def _init_precomputation(q, n, radius):
    global _WORDS, _NEIGHBORS, _TOTAL, _Q, _N, _WEIGHT
    if q != 3 or n != 8 or radius != 1:
        return False
    
    _Q = q
    _N = n
    _TOTAL = q ** n
    _WEIGHT = q ** np.arange(n - 1, -1, -1)
    
    # Generate all words
    _WORDS = np.empty((_TOTAL, n), dtype=np.int64)
    for i in range(_TOTAL):
        v = i
        for j in range(n - 1, -1, -1):
            _WORDS[i, j] = v % q
            v //= q
            
    # Precompute neighbors as list of indices
    _NEIGHBORS = []
    for i in range(_TOTAL):
        w = _WORDS[i].copy()
        nb = [i]
        for j in range(n):
            old = w[j]
            for a in range(q):
                if a == old:
                    continue
                w[j] = a
                nb.append(int(w.dot(_WEIGHT)))
                w[j] = old
        _NEIGHBORS.append(nb)
    return True

def _randomized_greedy_init(q, n, radius, rng):
    """Randomized greedy initialization."""
    if not _init_precomputation(q, n, radius):
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    uncovered = np.ones(_TOTAL, dtype=bool)
    remaining = _TOTAL
    codewords = []
    
    covered_count = np.array([len(nb) for nb in _NEIGHBORS], dtype=np.int64)
    
    import heapq
    heap = [(-covered_count[i], i) for i in range(_TOTAL)]
    heapq.heapify(heap)
    
    while remaining > 0:
        while True:
            neg_key, i = heapq.heappop(heap)
            if -neg_key == covered_count[i]:
                break
        
        codewords.append(i)
        
        for nb in _NEIGHBORS[i]:
            if uncovered[nb]:
                uncovered[nb] = False
                remaining -= 1
                for j in _NEIGHBORS[nb]:
                    covered_count[j] -= 1
                    if remaining > 0:
                        heapq.heappush(heap, (-covered_count[j], j))
                        
    return [tuple(_WORDS[i]) for i in codewords]

def _local_search(codewords, start_time, timeout, rng):
    """
    Local search with Perturb-and-Repair.
    """
    if not _init_precomputation(3, 8, 1):
        return codewords
        
    C = set(codewords)
    C_dict = {c: int(np.array(c, dtype=np.int64).dot(_WEIGHT)) for c in C}
    
    # Compute cover counts
    cover_count = np.zeros(_TOTAL, dtype=np.int64)
    for idx in C_dict.values():
        for nb in _NEIGHBORS[idx]:
            cover_count[nb] += 1
            
    current_size = len(C)
    
    def is_redundant(idx):
        return np.min(cover_count[_NEIGHBORS[idx]]) > 1

    def repair(uncovered_mask):
        """
        Greedily add codewords to cover uncovered_mask.
        Strategy: Pick a random uncovered point, find all codewords covering it,
        pick the one that covers the most *other* uncovered points.
        """
        while np.any(uncovered_mask):
            # Pick a random uncovered point
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            p = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
            
            # Candidates: all codewords that cover p
            candidates = _NEIGHBORS[p]
            
            best_idx = -1
            best_score = -1
            
            for c_idx in candidates:
                # If c_idx is already in C, it doesn't add new coverage?
                # Actually, if c_idx is in C, it was already counted in cover_count.
                # But we are adding it to C? No, C is a set.
                # If c_idx is already in C, adding it does nothing.
                # So we should skip candidates already in C?
                # Yes.
                if tuple(_WORDS[c_idx]) in C:
                    continue
                    
                # Calculate score: how many uncovered points does c_idx cover?
                score = 0
                for nb in _NEIGHBORS[c_idx]:
                    if uncovered_mask[nb]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_idx = c_idx
                    
            if best_idx == -1:
                # Fallback: just pick the first candidate not in C
                for c_idx in candidates:
                    if tuple(_WORDS[c_idx]) not in C:
                        best_idx = c_idx
                        break
                if best_idx == -1:
                    break # Should not happen
            
            # Add best_idx
            c_tuple = tuple(_WORDS[best_idx])
            C.add(c_tuple)
            C_dict[c_tuple] = best_idx
            
            for nb in _NEIGHBORS[best_idx]:
                cover_count[nb] += 1
                if uncovered_mask[nb]:
                    uncovered_mask[nb] = False

    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        
        # Phase 1: Delete redundant codewords
        removed_any = False
        for c in list(C):
            idx = C_dict[c]
            if is_redundant(idx):
                C.remove(c)
                del C_dict[c]
                for nb in _NEIGHBORS[idx]:
                    cover_count[nb] -= 1
                removed_any = True
                improved = True
                current_size -= 1
        if removed_any:
            continue
        
        # Phase 2: Perturb and Repair
        # Snapshot
        old_C = set(C)
        old_cover_count = cover_count.copy()
        old_size = len(C)
        
        # Pick a random codeword to remove
        c_to_remove = list(C)[rng.randint(0, len(C)-1)]
        idx_to_remove = C_dict[c_to_remove]
        
        # Remove
        C.remove(c_to_remove)
        del C_dict[c_to_remove]
        for nb in _NEIGHBORS[idx_to_remove]:
            cover_count[nb] -= 1
            
        # Identify uncovered
        uncovered_mask = (cover_count == 0).copy()
        
        # Repair
        repair(uncovered_mask)
        
        # Phase 3: Delete redundant again
        for c in list(C):
            idx = C_dict[c]
            if is_redundant(idx):
                C.remove(c)
                del C_dict[c]
                for nb in _NEIGHBORS[idx]:
                    cover_count[nb] -= 1
                improved = True
                
        # Check improvement
        if len(C) < old_size:
            improved = True
        else:
            # Revert
            C = old_C
            C_dict = {c: int(np.array(c, dtype=np.int64).dot(_WEIGHT)) for c in C}
            cover_count = old_cover_count

    return [list(c) for c in C]

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    _init_precomputation(q, n, radius)
    
    start_time = time.time()
    timeout = 115.0
    
    rng = random.Random(42)
    
    best_solution = None
    best_size = float('inf')
    
    while (time.time() - start_time) < timeout:
        init_cw = _randomized_greedy_init(q, n, radius, rng)
        
        ls_start = time.time()
        ls_timeout = min(5.0, timeout - (time.time() - start_time))
        
        if ls_timeout <= 0:
            break
            
        solution = _local_search(init_cw, ls_start, ls_timeout, rng)
        
        size = len(solution)
        if size < best_size:
            best_size = size
            best_solution = solution
            
    if best_solution is None:
        init_cw = _randomized_greedy_init(q, n, radius, rng)
        best_solution = [list(c) for c in init_cw]
        
    return best_solution
# EVOLVE-BLOCK-END
