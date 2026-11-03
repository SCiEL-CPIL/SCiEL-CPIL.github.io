import time
import random
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115  # safety margin
    
    # Precompute all words
    all_words = np.array(list(_itertools_product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Map each word to its index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    # Precompute coverage: for each codeword, which words it covers (itself + all at distance 1)
    # A word is at distance 1 from codeword c if they differ in exactly one position
    # Coverage of c: c itself, and for each position i, change c[i] to each other value
    coverage = np.zeros((N, N), dtype=bool)
    for i in range(N):
        c = all_words[i]
        coverage[i, i] = True
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                c2 = c.copy()
                c2[pos] = val
                idx = word_to_idx[tuple(c2)]
                coverage[i, idx] = True
    
    # Greedy set cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    while uncovered.any():
        if time.time() > deadline - 5:
            break
        # Find codeword covering most uncovered words
        counts = coverage[:, uncovered].sum(axis=1)
        best = np.argmax(counts)
        if counts[best] == 0:
            break
        selected.append(best)
        uncovered[coverage[best]] = False
    
    # Local search: try to remove codewords
    selected_set = set(selected)
    # Build coverage count for each word
    cover_count = np.zeros(N, dtype=np.int32)
    for idx in selected_set:
        cover_count[coverage[idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in list(selected_set):
            if time.time() > deadline - 5:
                break
            # Try removing idx
            cover_count[coverage[idx]] -= 1
            if cover_count.min() >= 1:
                selected_set.remove(idx)
                improved = True
            else:
                cover_count[coverage[idx]] += 1
    
    return [all_words[i].tolist() for i in sorted(selected_set)]


def _itertools_product(iterable, repeat=1):
    import itertools
    return itertools.product(iterable, repeat=repeat)
