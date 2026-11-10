"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random
import itertools
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(tail) for tail in itertools.product(range(q), repeat=n)]
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    def to_int(word):
        val = 0
        for c in word:
            val = val * 3 + c
        return val
    
    def to_word(val, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % 3
            val //= 3
        return word
    
    total = 3 ** n
    n_coords = n
    
    # Precompute neighborhoods
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        w = to_word(i, n_coords)
        nbrs = [i]
        for pos in range(n_coords):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                nw = w.copy()
                nw[pos] = val
                nbrs.append(to_int(nw))
        neighbors[i] = nbrs
    
    # Initial code: direct sum (729 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_code = [to_int(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in base_code:
        for w in neighbors[cw]:
            cover_count[w] += 1
    
    selected_set = set(base_code)
    
    # Phase 1: Greedy removal
    improved = True
    while improved and time.time() - start_time < time_limit - 20:
        improved = False
        for cw in list(selected_set):
            can_remove = True
            for w in neighbors[cw]:
                if cover_count[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                for w in neighbors[cw]:
                    cover_count[w] -= 1
                selected_set.discard(cw)
                improved = True
                break
    
    # Phase 2: Swap to reduce size
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        for cw in list(selected_set):
            only_cw = [w for w in neighbors[cw] if cover_count[w] == 1]
            if not only_cw:
                continue
            
            # Sample a subset of candidates to speed up
            candidates = random.sample(range(total), min(500, total))
            candidates = [j for j in candidates if j not in selected_set]
            
            for j in candidates:
                if all(w in neighbors[j] for w in only_cw):
                    for w in neighbors[cw]:
                        cover_count[w] -= 1
                    for w in neighbors[j]:
                        cover_count[w] += 1
                    selected_set.discard(cw)
                    selected_set.add(j)
                    improved = True
                    break
            if improved:
                break
    
    result = [to_word(cw, n_coords) for cw in selected_set]
    return result
# EVOLVE-BLOCK-END
