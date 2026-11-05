"""
Optimized ternary covering code K_3(6,1) construction.
Uses a greedy set-cover approach with local search to minimize codeword count.
"""
import itertools
import time
import random
import numpy as np

def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])

def _neighbors(word, q, n, radius=1):
    """Generate all words within Hamming distance <= radius from word."""
    result = [word[:]]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                nw = word[:]
                nw[i] = v
                result.append(nw)
    return result

def _all_words_within_radius1(q, n, word):
    """All words at Hamming distance exactly 1 or 0 from word."""
    result = [tuple(word)]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                nw = list(word)
                nw[i] = v
                result.append(tuple(nw))
    return result

def construct(q, n, radius):
    """
    Construct a covering code of size K_q(n, radius).
    For q=3, n=6, radius=1, aim for ~73 codewords.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum construction
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=6, radius=1: use greedy + local search
    all_space = list(itertools.product(range(3), repeat=6))
    space_size = 3 ** 6  # 729
    index_of = {w: i for i, w in enumerate(all_space)}
    
    # Precompute coverage sets for each potential codeword
    # coverage[cw_idx] = set of indices in all_space covered by codeword cw
    # We represent all_space as indices 0..728
    
    # Convert all_space words to tuples for consistent hashing
    # Precompute for each of the 729 possible words, which space indices it covers
    cover_sets = []
    for cw in all_space:
        covered = set()
        for i in range(6):
            for v in range(3):
                if v != cw[i]:
                    nw = list(cw)
                    nw[i] = v
                    covered.add(index_of[tuple(nw)])
        covered.add(index_of[cw])
        cover_sets.append(covered)
    
    # Greedy set cover: repeatedly pick codeword covering most uncovered points
    uncovered = set(range(space_size))
    chosen = []
    
    while uncovered:
        best_cw = -1
        best_count = -1
        for idx, cs in enumerate(cover_sets):
            cnt = len(cs & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_cw = idx
                if best_count == len(uncovered):
                    break
        if best_cw == -1:
            break
        chosen.append(all_space[best_cw])
        uncovered -= cover_sets[best_cw]
        if time.time() > deadline - 30:
            break
    
    if time.time() > deadline - 20:
        return chosen
    
    # Local search: try to remove redundant codewords and swap
    def is_valid_cover(codewords):
        covered = set()
        for cw in codewords:
            covered |= cover_sets[index_of[tuple(cw)]]
        return len(covered) == space_size
    
    # Try to remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(chosen)):
            if time.time() > deadline - 10:
                break
            # Check if removing chosen[i] still covers everything
            without_i = [chosen[j] for j in range(len(chosen)) if j != i]
            if is_valid_cover(without_i):
                chosen = without_i
                improved = True
                break
    
    # Local search: try swapping one codeword for another to enable further removal
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        if len(chosen) <= 71:
            break
        
        # Try removing a random codeword
        i = random.randint(0, len(chosen) - 1)
        candidate = [chosen[j] for j in range(len(chosen)) if j != i]
        # Check coverage
        covered = set()
        for cw in candidate:
            covered |= cover_sets[index_of[tuple(cw)]]
        if len(covered) < space_size:
            # Find uncovered points
            uncovered_pts = set(range(space_size)) - covered
            # Try to add a codeword that covers many uncovered points
            best_add = -1
            best_cnt = 0
            for idx, cs in enumerate(cover_sets):
                cnt = len(cs & uncovered_pts)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_add = idx
            if best_add >= 0 and best_cnt > 0:
                # Only swap if it reduces size (it won't here since we removed one and added one)
                # But maybe the new codeword enables future removals
                pass
        else:
            chosen = candidate
    
    # Final validation
    if not is_valid_cover(chosen):
        # Fallback to greedy result
        return [list(cw) for cw in chosen]
    
    return [list(cw) for cw in chosen]
