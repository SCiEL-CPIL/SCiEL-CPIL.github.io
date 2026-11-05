"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if q != 3 or n != 8 or radius != 1:
        return _generic_construct(q, n, radius, start_time, deadline)
    
    # Specialized for q=3, n=8, radius=1
    # Step 1: Generate initial code (729 words from tetracode)
    code = _initial_code(q, n)
    
    # Step 2: Greedy removal of redundant codewords
    code = _efficient_greedy_remove(code, q, n, radius, start_time, deadline)
    
    return code


def _initial_code(q, n):
    """Generate initial code using tetracode construction."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    return code


def _word_to_idx(w, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + w[i]
    return idx


def _idx_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def _get_neighbor_indices(idx, q, n, radius):
    """Get indices of all words within Hamming distance radius of word at idx."""
    w = _idx_to_word(idx, q, n)
    result = [idx]
    
    if radius >= 1:
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w[pos] = val
                    result.append(_word_to_idx(w, q, n))
            w[pos] = orig
    
    return result


def _efficient_greedy_remove(code, q, n, radius, start_time, deadline):
    """Remove redundant codewords using coverage counting."""
    total_words = q ** n
    num_cw = len(code)
    
    # Convert codewords to indices
    cw_indices = [_word_to_idx(cw, q, n) for cw in code]
    
    # Precompute neighbor index lists
    cw_neighbors = [_get_neighbor_indices(cw_indices[i], q, n, radius) for i in range(num_cw)]
    
    # Build coverage count
    cover_count = [0] * total_words
    for i in range(num_cw):
        for nb in cw_neighbors[i]:
            cover_count[nb] += 1
    
    # Greedy removal
    active = [True] * num_cw
    
    for iteration in range(20):
        if time.time() > deadline - 5:
            break
        
        progress = False
        for i in range(num_cw):
            if time.time() > deadline - 5:
                break
            if not active[i]:
                continue
            
            # Check if redundant
            redundant = True
            for nb in cw_neighbors[i]:
                if cover_count[nb] <= 1:
                    redundant = False
                    break
            
            if redundant:
                active[i] = False
                progress = True
                for nb in cw_neighbors[i]:
                    cover_count[nb] -= 1
        
        if not progress:
            break
    
    new_code = [code[i] for i in range(num_cw) if active[i]]
    return new_code


def _generic_construct(q, n, radius, start_time, deadline):
    """Generic construction."""
    all_words = list(itertools.product(range(q), repeat=n))
    rng = random.Random(42)
    rng.shuffle(all_words)
    
    uncovered = set(all_words)
    code = []
    
    for w in all_words:
        if time.time() > deadline - 10:
            break
        if not uncovered:
            break
        
        covers = [u for u in uncovered if sum(1 for a, b in zip(w, u) if a != b) <= radius]
        if covers:
            code.append(list(w))
            uncovered -= set(covers)
    
    return code
# EVOLVE-BLOCK-END
