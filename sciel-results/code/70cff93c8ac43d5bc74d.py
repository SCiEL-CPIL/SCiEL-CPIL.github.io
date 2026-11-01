import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 120.0
    
    all_words = list(itertools.product(range(3), repeat=6))
    total_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks
    coverage = [0] * total_words
    for i, c in enumerate(all_words):
        mask = 1 << i
        for pos in range(6):
            for val in range(3):
                if val == c[pos]:
                    continue
                new_c = list(c)
                new_c[pos] = val
                mask |= 1 << word_to_idx[tuple(new_c)]
        coverage[i] = mask
    
    full_mask = (1 << total_words) - 1
    
    # Spectral-inspired scoring: use algebraic structure of H(6,3)
    # The graph is vertex-transitive; use coordinate-based spectral features
    # For each word, compute a "spectral centrality" score
    spectral_scores = np.zeros(total_words)
    for i, c in enumerate(all_words):
        arr = np.array(c, dtype=np.float64)
        # Features: sum, sum of squares, max deviation from center
        s = np.sum(arr)
        sq = np.sum(arr ** 2)
        dev = np.max(np.abs(arr - 1.0))
        # Combine: words with "balanced" coordinates (sum near 3) are central
        spectral_scores[i] = 3.0 - abs(s - 3.0) + 0.5 * (6.0 - sq) - 0.2 * dev
    
    # Sort by spectral score descending, with random tie-breaking
    order = np.argsort(-spectral_scores)
    
    best_size = total_words
    best_code = None
    
    # Multiple restarts with different tie-breaking
    num_restarts = 80
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(r * 7919 + 13)
        
        # Create candidate order: primary by spectral score, secondary random
        shuffled_order = list(order)
        # Add random perturbation to break ties
        for idx in shuffled_order:
            spectral_scores[idx] += rng.uniform(-0.01, 0.01)
        shuffled_order = np.argsort(-spectral_scores)
        
        covered_mask = 0
        chosen = []
        
        while covered_mask != full_mask:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_idx = -1
            best_tiebreak = -1e9
            
            for idx in shuffled_order:
                new_cov = coverage[idx] & ~covered_mask
                gain = new_cov.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    # Tiebreak: prefer words with more balanced coordinates
                    c = all_words[idx]
                    arr = np.array(c, dtype=np.float64)
                    s = np.sum(arr)
                    tiebreak = 3.0 - abs(s - 3.0)
                    best_tiebreak = tiebreak
                elif gain == best_gain and gain > 0:
                    c = all_words[idx]
                    arr = np.array(c, dtype=np.float64)
                    s = np.sum(arr)
                    tiebreak = 3.0 - abs(s - 3.0)
                    if tiebreak > best_tiebreak:
                        best_tiebreak = tiebreak
                        best_idx = idx
            
            if best_gain <= 0:
                break
            
            chosen.append(best_idx)
            covered_mask |= coverage[best_idx]
        
        # Local search: try to remove and re-add codewords
        if len(chosen) < best_size and time.time() < deadline - 5.0:
            chosen = _local_search(chosen, coverage, all_words, total_words, full_mask, deadline - 5.0, rng)
        
        if covered_mask == full_mask and len(chosen) < best_size:
            best_size = len(chosen)
            best_code = [all_words[i] for i in chosen]
    
    if best_code is None:
        # Fallback
        return _generic_construct(3, 6, 1)
    
    return [list(c) for c in best_code]


def _local_search(chosen, coverage, all_words, total_words, full_mask, deadline, rng):
    """Try to reduce code size by replacing codewords."""
    chosen_set = set(chosen)
    covered_mask = 0
    for idx in chosen:
        covered_mask |= coverage[idx]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_chosen = len(chosen)
        for i in range(n_chosen):
            if time.time() > deadline:
                break
            idx = chosen[i]
            # Try removing idx
            temp_mask = 0
            for j, k in enumerate(chosen):
                if j != i:
                    temp_mask |= coverage[k]
            
            if temp_mask == full_mask:
                # Can remove idx
                chosen.pop(i)
                covered_mask = temp_mask
                improved = True
                break
            else:
                # Uncovered words
                uncovered = full_mask & ~temp_mask
                # Try to find a replacement that covers uncovered words
                best_gain = 0
                best_repl = -1
                for candidate in range(total_words):
                    if candidate in chosen_set:
                        continue
                    new_cov = coverage[candidate] & uncovered
                    gain = new_cov.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_repl = candidate
                if best_repl >= 0 and best_gain > 0:
                    chosen[i] = best_repl
                    chosen_set.discard(idx)
                    chosen_set.add(best_repl)
                    covered_mask = temp_mask | coverage[best_repl]
                    improved = True
                    break
    return chosen


def _generic_construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    coverage = [0] * total_words
    for i, c in enumerate(all_words):
        mask = 1 << i
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    new_c = list(c)
                    new_c[pos] = val
                    mask |= 1 << word_to_idx[tuple(new_c)]
        coverage[i] = mask
    
    full_mask = (1 << total_words) - 1
    covered_mask = 0
    chosen = []
    
    while covered_mask != full_mask:
        best_gain = -1
        best_idx = -1
        for i in range(total_words):
            if i in chosen:
                continue
            gain = (coverage[i] & ~covered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain <= 0:
            break
        chosen.append(best_idx)
        covered_mask |= coverage[best_idx]
    
    return [list(all_words[i]) for i in chosen]
