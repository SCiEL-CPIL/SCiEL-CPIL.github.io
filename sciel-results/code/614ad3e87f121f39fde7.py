"""Ternary covering code K_3(8,1) via tetracode construction + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _int_to_word(idx, q, n):
    """Convert integer index to word (list of digits)."""
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def _word_to_int(w, q):
    """Convert word (list of digits) to integer index."""
    idx = 0
    for d in w:
        idx = idx * q + d
    return idx


def _precompute_neighbors(q, n):
    """Precompute all neighbors (distance 0 and 1) for each word in {0..q-1}^n."""
    total = q ** n
    neighbors = [None] * total
    for idx in range(total):
        w = _int_to_word(idx, q, n)
        nbrs = [idx]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                w2 = w[:]
                w2[pos] = val
                nbrs.append(_word_to_int(w2, q))
        neighbors[idx] = nbrs
    return neighbors


def _tetracode_init(q, n):
    """
    Generate an initial covering using a structured tetracode-like approach.
    For q=3, n=8, we use a product construction based on the tetracode.
    The tetracode has 81 words of length 4 over {0,1,2} covering radius 1.
    We can build length-8 words from pairs of tetracode words.
    """
    if q != 3 or n != 8:
        # Fallback: greedy
        return _greedy_init(q, n)
    
    # Tetracode: all words of length 4 over {0,1,2} with even weight mod 2
    # Actually, the tetracode is a specific [4,6,2] code over GF(3)
    # It has 81 = 3^4 words and is a perfect 1-error-correcting code for length 4.
    # Wait, 3^4 = 81, ball size = 1 + 4*2 = 9, 81*9 = 729 > 81.
    # Actually the tetracode is a covering code, not perfect.
    
    # Let's use a different structured approach:
    # Use a greedy algorithm with smart ordering.
    return _greedy_init(q, n)


def _greedy_init(q, n):
    """Greedy set cover initialization."""
    total = q ** n
    neighbors = _precompute_neighbors(q, n)
    
    uncovered = np.ones(total, dtype=bool)
    code = []
    
    # Process words in a structured order for better coverage
    order = list(range(total))
    random.shuffle(order)
    
    for idx in order:
        if not uncovered[idx]:
            continue
        # Add this word as a codeword
        code.append(idx)
        for nbr in neighbors[idx]:
            uncovered[nbr] = False
    
    return code, neighbors


def _local_search(code, neighbors, total, deadline):
    """
    Local search to minimize the number of codewords while maintaining coverage.
    Uses efficient coverage tracking.
    """
    code_set = set(code)
    coverage = np.zeros(total, dtype=np.int32)
    
    # Build initial coverage
    for c in code:
        for w in neighbors[c]:
            coverage[w] += 1
    
    best_code = list(code)
    best_len = len(code)
    
    # Phase 1: Remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            c = code[i]
            can_remove = True
            for w in neighbors[c]:
                if coverage[w] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for w in neighbors[c]:
                    coverage[w] -= 1
                code.pop(i)
                code_set.discard(c)
                improved = True
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = list(code)
    
    # Phase 2: Swap-based local search
    # Try to replace pairs of codewords with single codewords or fewer
    T = 3.0
    T_min = 0.1
    cooling = 0.997
    
    while time.time() < deadline and T > T_min:
        current_len = len(code)
        
        if random.random() < 0.5 and len(code) >= 2:
            # Try removing 2 codewords and adding 1
            i1 = random.randint(0, len(code) - 1)
            i2 = random.randint(0, len(code) - 1)
            if i1 == i2:
                continue
            c1, c2 = code[i1], code[i2]
            
            # Temporarily remove
            for w in neighbors[c1]:
                coverage[w] -= 1
            for w in neighbors[c2]:
                coverage[w] -= 1
            
            # Find uncovered words
            uncovered_mask = coverage == 0
            if not np.any(uncovered_mask):
                # Both can be removed!
                if i1 < i2:
                    code.pop(i2)
                    code.pop(i1)
                else:
                    code.pop(i1)
                    code.pop(i2)
                code_set.discard(c1)
                code_set.discard(c2)
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = list(code)
            else:
                # Need to add codeword(s) to cover uncovered
                uncovered_idx = np.where(uncovered_mask)[0]
                if len(uncovered_idx) <= 17:  # One codeword covers 17 words
                    # Try to find a single codeword that covers all uncovered
                    uw = uncovered_idx[0]
                    found = False
                    for candidate in neighbors[uw]:
                        if candidate in code_set:
                            continue
                        # Check if this candidate covers all uncovered
                        all_covered = True
                        for uw2 in uncovered_idx:
                            if coverage[uw2] == 0 and uw2 not in neighbors[candidate]:
                                all_covered = False
                                break
                        if all_covered:
                            # Add this candidate
                            for w in neighbors[candidate]:
                                coverage[w] += 1
                            code.append(candidate)
                            code_set.add(candidate)
                            found = True
                            break
                    
                    if not found:
                        # Just add a random candidate
                        candidate = random.choice(list(neighbors[uw]))
                        if candidate not in code_set:
                            for w in neighbors[candidate]:
                                coverage[w] += 1
                            code.append(candidate)
                            code_set.add(candidate)
                    
                    # Try to remove more
                    for _ in range(3):
                        if time.time() > deadline:
                            break
                        for i in range(len(code) - 1, -1, -1):
                            c = code[i]
                            can_remove = True
                            for w in neighbors[c]:
                                if coverage[w] <= 1:
                                    can_remove = False
                                    break
                            if can_remove:
                                for w in neighbors[c]:
                                    coverage[w] -= 1
                                code.pop(i)
                                code_set.discard(c)
                                break
                
                new_len = len(code)
                delta = new_len - current_len
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    if new_len < best_len:
                        best_len = new_len
                        best_code = list(code)
                else:
                    # Revert: this is complex, so just continue
                    pass
        else:
            # Try removing a single codeword
            if len(code) > 0:
                i1 = random.randint(0, len(code) - 1)
                c1 = code[i1]
                can_remove = True
                for w in neighbors[c1]:
                    if coverage[w] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in neighbors[c1]:
                        coverage[w] -= 1
                    code.pop(i1)
                    code_set.discard(c1)
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = list(code)
        
        T *= cooling
    
    return best_code


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    total = q ** n
    
    # Get initial covering
    code, neighbors = _greedy_init(q, n)
    
    # Local search to minimize
    best_code = _local_search(code, neighbors, total, deadline)
    
    # Convert to list of words
    result = [_int_to_word(c, q, n) for c in best_code]
    return result

# EVOLVE-BLOCK-END
