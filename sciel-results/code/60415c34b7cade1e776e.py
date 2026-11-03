import random
import numpy as np

def construct(q, n, radius):
    # Generate all words
    all_words = []
    for i in range(q ** n):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))
    
    # Precompute coverage: for each word, which words are within radius
    # For radius=1, a word covers itself and words differing in exactly one position
    # Build a mapping from word index to list of covered word indices
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    coverage = []
    for i, w in enumerate(all_words):
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nw = tuple(nw)
                j = word_to_idx[nw]
                covered.append(j)
        coverage.append(covered)
    
    total = len(all_words)
    covered = [False] * total
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    codewords = []
    uncovered_count = total
    
    while uncovered_count > 0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = sum(1 for j in coverage[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        codewords.append(best_idx)
        for j in coverage[best_idx]:
            if not covered[j]:
                covered[j] = True
                uncovered_count -= 1
    
    # Local search: try to remove codewords
    # Repeatedly try to remove each codeword; if coverage is still full, remove it
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords) - 1, -1, -1):
            # Check if removing codewords[idx] still leaves full coverage
            # Build coverage from remaining codewords
            temp_covered = [False] * total
            for ci in range(len(codewords)):
                if ci == idx:
                    continue
                for j in coverage[codewords[ci]]:
                    temp_covered[j] = True
            if all(temp_covered):
                codewords.pop(idx)
                improved = True
                break
    
    # Convert to list of lists
    return [list(all_words[i]) for i in codewords]
