"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product


def _to_int(w, q):
    v = 0
    for x in w:
        v = v * q + x
    return v


def _to_word(i, q, n):
    w = [0] * n
    for j in range(n - 1, -1, -1):
        w[j] = i % q
        i //= q
    return w


def _build_neighbor_map(q, n, radius):
    """Build neighbor list for each word in {0..q-1}^n."""
    total = q ** n
    neighbor_map = []
    for wi in range(total):
        w = _to_word(wi, q, n)
        nbrs = set()
        for r in range(radius + 1):
            for pos in combinations(range(n), r):
                for vals in product(range(q), repeat=r):
                    nw = list(w)
                    for i, v in zip(pos, vals):
                        nw[i] = v
                    ni = _to_int(nw, q)
                    nbrs.add(ni)
        neighbor_map.append(nbrs)
    return neighbor_map


def _build_incidence(neighbor_map, total_space):
    """Build dense incidence matrix: inc[cw, w] = 1 if w is in neighbor_map[cw]."""
    inc = np.zeros((total_space, total_space), dtype=np.int8)
    for cw in range(total_space):
        for w in neighbor_map[cw]:
            inc[cw, w] = 1
    return inc


def _greedy_cover(inc, total_space, deadline, seed=42):
    """Greedy set cover using vectorized incidence matrix."""
    rng = random.Random(seed)
    uncovered = np.ones(total_space, dtype=np.int8)
    codewords = []
    
    while True:
        if time.time() >= deadline:
            break
        if uncovered.sum() == 0:
            break
        
        # Find codeword covering most uncovered words
        gains = inc @ uncovered.astype(np.int32)
        best_cw = int(np.argmax(gains))
        if gains[best_cw] == 0:
            break
        
        codewords.append(best_cw)
        uncovered[inc[best_cw] == 1] = 0
    
    return codewords


def _local_search_reduce(inc, neighbor_map, codewords, total_space, deadline):
    """Try to remove redundant codewords."""
    if time.time() >= deadline:
        return codewords

    changed = True
    while changed:
        if time.time() >= deadline:
            break
        changed = False
        # Compute coverage counts
        counts = inc[codewords].sum(axis=0)

        # Check all covered
        if counts.min() == 0:
            break

        # Try removing each codeword (from end to start)
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() >= deadline:
                break
            cw = codewords[i]
            removable = True
            for w in neighbor_map[cw]:
                if counts[w] <= 1:
                    removable = False
                    break
            if removable:
                codewords.pop(i)
                changed = True
                # Recompute counts after removal
                counts = inc[codewords].sum(axis=0)
                break

    return codewords


def _critical_word_swap(inc, neighbor_map, codewords, total_space, deadline, seed=42, max_iters=200):
    """
    Critical word heuristic: identify words covered by only one codeword,
    then try to swap that codeword with an alternative that covers all its critical words.
    """
    rng = random.Random(seed)
    codewords = list(codewords)
    cw_set = set(codewords)
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Compute coverage counts
        counts = inc[codewords].sum(axis=0)
        
        # Find critical words (covered by exactly 1 codeword)
        critical_mask = (counts == 1)
        n_critical = critical_mask.sum()
        
        if n_critical == 0:
            break
        
        critical_words = np.where(critical_mask)[0]
        
        # Randomly pick a critical word
        cw_word = int(critical_words[rng.randint(0, len(critical_words) - 1)])
        
        # Find which codeword covers this word
        covering_idx = None
        for i, cw in enumerate(codewords):
            if inc[cw, cw_word] == 1:
                covering_idx = i
                break
        
        if covering_idx is None:
            continue
        
        old_cw = codewords[covering_idx]
        
        # Find all codewords that cover the critical word
        candidates = np.where(inc[:, cw_word] == 1)[0]
        candidates = [c for c in candidates if c != old_cw]
        
        if not candidates:
            continue
        
        # Critical words covered by old_cw
        old_criticals = set()
        for w in neighbor_map[old_cw]:
            if counts[w] == 1:
                old_criticals.add(w)
        
        best_candidate = None
        best_score = -1
        
        for cand in candidates:
            if time.time() >= deadline:
                break
            
            # How many of old_cw's critical words does cand cover?
            cand_covers_old_criticals = sum(1 for w in old_criticals if inc[cand, w] == 1)
            
            if cand_covers_old_criticals < len(old_criticals):
                continue
            
            # Additional benefit: cand covers more critical words overall
            cand_covers_criticals = sum(1 for w in critical_words if inc[cand, w] == 1)
            
            score = cand_covers_criticals + rng.random() * 0.01
            
            if score > best_score:
                best_score = score
                best_candidate = cand
        
        if best_candidate is not None:
            # Perform the swap
            codewords[covering_idx] = best_candidate
            cw_set.discard(old_cw)
            cw_set.add(best_candidate)
            
            # Try to remove redundant codewords after swap
            counts = inc[codewords].sum(axis=0)
            changed = True
            while changed:
                if time.time() >= deadline:
                    break
                changed = False
                for i in range(len(codewords) - 1, -1, -1):
                    if time.time() >= deadline:
                        break
                    cw = codewords[i]
                    removable = True
                    for w in neighbor_map[cw]:
                        if counts[w] <= 1:
                            removable = False
                            break
                    if removable:
                        codewords.pop(i)
                        cw_set.discard(cw)
                        counts = inc[codewords].sum(axis=0)
                        changed = True
                        break

    return codewords


def _local_search_iterate(inc, neighbor_map, codewords, total_space, deadline, seed=42):
    """Iterate reduce and swap until no improvement."""
    if time.time() >= deadline:
        return codewords

    improved = True
    while improved:
        if time.time() >= deadline:
            break
        before = len(codewords)
        codewords = _local_search_reduce(inc, neighbor_map, codewords, total_space, deadline)
        codewords = _critical_word_swap(inc, neighbor_map, codewords, total_space, deadline, seed=seed)
        improved = len(codewords) < before

    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    total_space = q ** n

    # Build neighbor map
    neighbor_map = _build_neighbor_map(q, n, radius)
    
    # Build dense incidence matrix
    inc = _build_incidence(neighbor_map, total_space)

    best_codewords = None
    best_len = float('inf')

    # Multiple restarts with different random seeds
    num_restarts = 8
    per_restart_time = (deadline - 10.0) / num_restarts

    for restart in range(num_restarts):
        restart_deadline = start_time + 5.0 + per_restart_time * (restart + 1)
        if time.time() > restart_deadline - 3.0:
            break

        # Greedy with vectorized incidence matrix
        greedy_deadline = restart_deadline - 4.0
        codewords = _greedy_cover(inc, total_space, deadline=greedy_deadline, seed=restart)

        # Local search
        ls_deadline = restart_deadline - 1.0
        codewords = _local_search_iterate(inc, neighbor_map, codewords, total_space, deadline=ls_deadline, seed=restart)

        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords[:]

    # Final local search with remaining time
    if best_codewords is not None and time.time() < deadline - 2.0:
        best_codewords = _local_search_iterate(inc, neighbor_map, best_codewords, total_space, deadline=deadline - 1.0, seed=999)

    # Convert to word lists
    result = [_to_word(cw, q, n) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
