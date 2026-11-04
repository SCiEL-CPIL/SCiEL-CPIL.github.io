"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _ball_indices(word, q, n, radius):
    """Return the flat indices of all words within Hamming distance `radius` of `word`."""
    # Precompute all words within radius via itertools
    idxs = []
    # radius 1: word itself + 1 position * (q-1) changes
    if radius == 1:
        idxs.append(int(np.ravel_multi_index(tuple(word), (q,)*n)))
        for pos in range(n):
            for v in range(q):
                if v == word[pos]:
                    continue
                w2 = list(word)
                w2[pos] = v
                idxs.append(int(np.ravel_multi_index(tuple(w2), (q,)*n)))
    return idxs


def _tetracode(q, n, radius):
    """Direct product: tetracode on first 4 coords, all words on remaining n-4 coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover_fast(q, n, radius, deadline):
    """
    Greedy set cover: at each step pick the codeword whose ball covers the most
    currently-uncovered words. Use numpy bitsets for speed.
    """
    total = q ** n
    # Precompute balls as sorted lists of flat indices
    # For n=6, q=3: 729 words, each ball has 1 + 6*2 = 13 indices
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    balls = [_ball_indices(w, q, n, radius) for w in all_words]
    
    # uncovered: boolean array
    uncovered = np.ones(total, dtype=np.uint8)
    uncovered_count = total
    chosen = []
    
    # Precompute ball sets as frozensets for fast intersection, or use numpy
    # For 729 words and 13-element balls, use a different approach:
    # For each candidate, compute how many of its ball positions are still uncovered
    
    # Use a list of sets for uncovered tracking
    uncovered_set = set(range(total))
    
    while uncovered_set and time.time() < deadline:
        best_cand = -1
        best_count = 0
        for i, ball in enumerate(balls):
            # count how many in ball are still uncovered
            cnt = sum(1 for idx in ball if idx in uncovered_set)
            if cnt > best_count:
                best_count = cnt
                best_cand = i
        if best_cand < 0:
            break
        chosen.append(all_words[best_cand])
        # Remove covered
        for idx in balls[best_cand]:
            uncovered_set.discard(idx)
    
    return chosen


def _remove_redundant(codewords, q, n, radius):
    """Try to remove each codeword; if the covering property still holds, keep it removed."""
    total = q ** n
    # Build coverage: for each word, which codewords cover it
    # For small spaces, just check coverage directly
    chosen = list(codewords)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            # Try removing chosen[i]
            remaining = chosen[:i] + chosen[i+1:]
            # Check if all words are still covered
            if not remaining:
                # Can't remove if it's the only one (unless n=0)
                continue
            # Quick check: build union of balls
            covered = set()
            for cw in remaining:
                for idx in _ball_indices(cw, q, n, radius):
                    covered.add(idx)
            if len(covered) == total:
                chosen = remaining
                changed = True
                break
    return chosen


def _construct_k3_6_1(q, n, radius, deadline):
    """
    Specialized construction for q=3, n=6, radius=1.
    Start with the 81-word direct product, then greedily remove redundant words.
    """
    # Start with tetracode product (81 words)
    codewords = _tetracode(q, n, radius)
    
    # Aggressive removal: repeatedly try to remove each word
    # Use a faster coverage check
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute balls for all 729 possible words
    balls_cache = {}
    for w in all_words:
        balls_cache[w] = _ball_indices(list(w), q, n, radius)
    
    # For removal, check: after removing word i, is every word still within distance 1
    # of some remaining codeword?
    # Optimization: for each word, track how many codewords cover it (multiplicity)
    # A word can be removed if no word has multiplicity 1 where it's the sole coverer
    
    # Build coverage counts
    from collections import defaultdict
    cover_count = defaultdict(int)  # word -> number of codewords covering it
    for cw in codewords:
        for idx in balls_cache[tuple(cw)]:
            cover_count[idx] += 1
    
    # A codeword at position i can be removed if for every word w in its ball,
    # cover_count[w] > 1 (i.e., some other codeword also covers w)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            cw = tuple(codewords[i])
            can_remove = True
            for idx in balls_cache[cw]:
                if cover_count[idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove codeword i
                for idx in balls_cache[cw]:
                    cover_count[idx] -= 1
                codewords.pop(i)
                changed = True
                break  # Restart scan after modification
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    # Reserve 10s safety margin
    deadline = start_time + 110.0
    
    if q == 3 and n == 6 and radius == 1:
        return _construct_k3_6_1(q, n, radius, deadline)
    else:
        # Fallback: direct product construction
        return _tetracode(q, n, radius)
# EVOLVE-BLOCK-END
