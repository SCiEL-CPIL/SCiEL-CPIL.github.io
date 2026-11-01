"""Stochastic Local Search (Kick and Repair) for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import heapq
import itertools
import numpy as np
import time
import random

# Precomputation cache for the specific task
_Q = 3
_N = 8
_RADIUS = 1
_TOTAL = _Q ** _N
_WEIGHT = _Q ** np.arange(_N - 1, -1, -1)
_WORDS = None
_NEIGHBORS = None

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
            
    # Precompute neighbors
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

def _greedy_init(q, n, radius):
    """Fast greedy initialization."""
    if not _init_precomputation(q, n, radius):
        # Fallback for other params
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    uncovered = np.ones(_TOTAL, dtype=bool)
    remaining = _TOTAL
    codewords_idx = []
    
    covered_count = np.array([len(nb) for nb in _NEIGHBORS], dtype=np.int64)
    heap = [(-covered_count[i], i) for i in range(_TOTAL)]
    heapq.heapify(heap)
    
    while remaining > 0:
        while True:
            neg_key, i = heapq.heappop(heap)
            if -neg_key == covered_count[i]:
                break
        codewords_idx.append(i)
        for nb in _NEIGHBORS[i]:
            if uncovered[nb]:
                uncovered[nb] = False
                remaining -= 1
                for j in _NEIGHBORS[nb]:
                    covered_count[j] -= 1
                    if remaining > 0:
                        heapq.heappush(heap, (-covered_count[j], j))
                        
    return codewords_idx

def _local_search_remove(current_set_idx, cover_count):
    """
    Try to remove redundant codewords.
    current_set_idx: set of indices
    cover_count: np array
    Returns updated set and cover_count.
    """
    improved = True
    while improved:
        improved = False
        # Find a removable codeword
        for idx in list(current_set_idx):
            nb_list = _NEIGHBORS[idx]
            if np.min(cover_count[nb_list]) > 1:
                current_set_idx.remove(idx)
                for nb in nb_list:
                    cover_count[nb] -= 1
                improved = True
                break
    return current_set_idx, cover_count

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    _init_precomputation(q, n, radius)
    
    # 1. Greedy Initialization
    init_idx = _greedy_init(q, n, radius)
    best_set_idx = set(init_idx)
    
    # Calculate cover count for best_set
    best_cover_count = np.zeros(_TOTAL, dtype=np.int64)
    for idx in best_set_idx:
        for nb in _NEIGHBORS[idx]:
            best_cover_count[nb] += 1
            
    best_size = len(best_set_idx)
    
    start_time = time.time()
    timeout = 110.0
    
    # 2. Iterated Local Search with Kick and Repair
    while (time.time() - start_time) < timeout:
        # Current solution
        current_set_idx = set(best_set_idx)
        current_cover_count = best_cover_count.copy()
        
        # Local Search: Remove redundant
        current_set_idx, current_cover_count = _local_search_remove(current_set_idx, current_cover_count)
        
        if len(current_set_idx) < best_size:
            best_set_idx = current_set_idx
            best_cover_count = current_cover_count
            best_size = len(best_set_idx)
            
        # Kick: Randomly remove 1-3 codewords to create holes
        kick_size = random.randint(1, 3)
        if len(current_set_idx) <= kick_size:
            continue
        kick_candidates = list(current_set_idx)
        random.shuffle(kick_candidates)
        to_remove = kick_candidates[:kick_size]
        
        for idx in to_remove:
            current_set_idx.remove(idx)
            for nb in _NEIGHBORS[idx]:
                current_cover_count[nb] -= 1
                
        # Repair: Greedily add codewords to cover uncovered points
        uncovered_mask = current_cover_count == 0
        uncovered_count = np.sum(uncovered_mask)
        
        if uncovered_count > 0:
            while uncovered_count > 0 and (time.time() - start_time) < timeout:
                # Find best word to add
                # Pick a random uncovered point to focus on
                uncov_indices = np.where(uncovered_mask)[0]
                if len(uncov_indices) == 0:
                    break
                target = uncov_indices[random.randint(0, len(uncov_indices)-1)]
                
                # Get neighbors of target
                candidates = _NEIGHBORS[target]
                
                # Pick the candidate that covers the most uncovered points
                best_cand = -1
                max_cov = -1
                for cand in candidates:
                    if cand in current_set_idx:
                        continue
                    cov = np.sum(uncovered_mask[_NEIGHBORS[cand]])
                    if cov > max_cov:
                        max_cov = cov
                        best_cand = cand
                        
                if best_cand == -1:
                    # Should not happen if target is uncovered and its neighbors are valid candidates
                    # But if all neighbors are already in set, they would have covered it.
                    # So there must be a candidate not in set.
                    break
                    
                # Add best_cand
                current_set_idx.add(best_cand)
                for nb in _NEIGHBORS[best_cand]:
                    if current_cover_count[nb] == 0:
                        uncovered_count -= 1
                    current_cover_count[nb] += 1
                uncovered_mask = current_cover_count == 0

        # After repair, try to remove again
        current_set_idx, current_cover_count = _local_search_remove(current_set_idx, current_cover_count)
        
        if len(current_set_idx) < best_size:
            best_set_idx = current_set_idx
            best_cover_count = current_cover_count
            best_size = len(best_set_idx)

    return [list(_WORDS[idx]) for idx in best_set_idx]
# EVOLVE-BLOCK-END
