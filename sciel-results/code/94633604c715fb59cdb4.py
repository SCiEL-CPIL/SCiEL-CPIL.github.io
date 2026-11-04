"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _solve_3_6_1_bitset()
    else:
        return _generic_greedy(q, n, radius)

def _solve_3_6_1_bitset():
    N = 729
    # Precompute cover masks
    # word idx -> bitmask of covered words
    cover_masks = [0] * N
    
    # Helper to convert idx to word and back
    def idx_to_word(idx):
        w = [0] * 6
        for i in range(5, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(6):
            idx = idx * 3 + w[i]
        return idx

    for idx in range(N):
        w = idx_to_word(idx)
        mask = 1 << idx
        for i in range(6):
            for v in range(3):
                if v != w[i]:
                    nw = w[:]
                    nw[i] = v
                    nidx = word_to_idx(nw)
                    mask |= (1 << nidx)
        cover_masks[idx] = mask

    full_mask = (1 << N) - 1
    
    # Greedy
    uncovered = full_mask
    code = []
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for c in range(N):
            # Count new coverage
            new_cov = cover_masks[c] & uncovered
            # Popcount for Python int
            count = bin(new_cov).count('1')
            if count > best_count:
                best_count = count
                best_idx = c
                if count == 0:
                    break
        if best_idx == -1 or best_count == 0:
            # Should not happen if full_mask is correct
            break
        code.append(best_idx)
        uncovered &= ~cover_masks[best_idx]
        
    # Local Search: Remove redundant
    improved = True
    while improved:
        improved = False
        for i in range(len(code)):
            # Try removing i
            temp_code = code[:i] + code[i+1:]
            # Check if temp_code covers everything
            # Optimization: just check if the union of masks of temp_code is full_mask
            # This is O(K) bit ops
            union_mask = 0
            for c in temp_code:
                union_mask |= cover_masks[c]
            if union_mask == full_mask:
                code = temp_code
                improved = True
                break # Restart with smaller code

    # Convert indices back to words
    return [idx_to_word(c) for c in code]

def _generic_greedy(q, n, radius):
    # Fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Precompute distances or cover sets
    # For generic case, we might be slower, but this is a fallback
    cover_sets = []
    for c in all_words:
        covered = set()
        covered.add(c)
        for i in range(n):
            for v in range(q):
                if v != c[i]:
                    if radius >= 1:
                        nc = c[:]
                        nc[i] = v
                        covered.add(tuple(nc))
        cover_sets.append(covered)
        
    # Greedy
    uncovered = set(all_words)
    code = []
    while uncovered:
        best_idx = -1
        best_count = -1
        for i, c in enumerate(all_words):
            if tuple(c) in [tuple(x) for x in code]:
                continue
            new_cov = cover_sets[i] & uncovered
            if len(new_cov) > best_count:
                best_count = len(new_cov)
                best_idx = i
        if best_idx == -1:
            break
        code.append(all_words[best_idx])
        uncovered -= cover_sets[best_idx]
        
    return code
# EVOLVE-BLOCK-END
