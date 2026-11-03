"""Rewritten program: greedy set-cover with iterative removal for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists of ints."""
    return sum(1 for x, y in zip(a, b) if x != y)

def _ball_around(center, q, n):
    """Yield all words within Hamming distance 1 of center (radius=1 ball)."""
    yield tuple(center)
    for i in range(n):
        for v in range(q):
            if v != center[i]:
                lst = list(center)
                lst[i] = v
                yield tuple(lst)

def _all_words(q, n):
    """Yield all words in {0..q-1}^n."""
    return product(range(q), repeat=n)

def _greedy_cover(q, n, radius, rng, deadline):
    """
    Greedy set-cover: repeatedly pick the codeword that covers the most
    currently-uncovered words. Returns list of codewords.
    """
    # Precompute the ball (set of words within distance `radius`) for each possible codeword.
    # For radius=1, ball size = 1 + n*(q-1). For q=3, n=7: 1+7*2=15.
    # Total words: 3^7 = 2187.
    total_words = q ** n
    # Map word tuple -> index
    word_to_idx = {}
    all_words_list = []
    for w in _all_words(q, n):
        word_to_idx[w] = len(all_words_list)
        all_words_list.append(w)
    
    # Precompute ball for each codeword as a set of word indices
    # A codeword is also a word in {0..q-1}^n
    balls = {}
    for cw in _all_words(q, n):
        ball_set = set()
        # radius 1
        ball_set.add(word_to_idx[cw])
        for i in range(n):
            for v in range(q):
                if v != cw[i]:
                    lst = list(cw)
                    lst[i] = v
                    ball_set.add(word_to_idx[tuple(lst)])
        balls[cw] = ball_set
    
    # Uncovered words
    uncovered = set(range(total_words))
    codewords = []
    
    # Precompute for each word index, which codewords cover it (inverse)
    # This helps in the removal phase later
    word_to_cw = [set() for _ in range(total_words)]
    for cw, bs in balls.items():
        for wi in bs:
            word_to_cw[wi].add(cw)
    
    while uncovered and time.time() < deadline:
        # Pick codeword covering most uncovered
        best_cw = None
        best_count = -1
        for cw in balls:
            cnt = len(uncovered & balls[cw])
            if cnt > best_count:
                best_count = cnt
                best_cw = cw
                if best_count == len(uncovered):
                    break
        if best_cw is None or best_count <= 0:
            break
        codewords.append(best_cw)
        uncovered -= balls[best_cw]
    
    # Iterative removal: try to remove each codeword if coverage is maintained
    # Build coverage count for each word
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        for wi in balls[cw]:
            cover_count[wi] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(codewords)):
            cw = codewords[idx]
            # Check if any word in its ball has cover_count == 1
            # If so, removing it would leave that word uncovered
            removable = True
            for wi in balls[cw]:
                if cover_count[wi] == 1:
                    removable = False
                    break
            if removable:
                # Remove it
                for wi in balls[cw]:
                    cover_count[wi] -= 1
                codewords.pop(idx)
                improved = True
                break  # Restart scan since indices changed
    
    return codewords

def _greedy_cover_fast(q, n, radius, rng, deadline):
    """
    Faster greedy: use bitsets (numpy) for uncovered set to speed up counting.
    """
    total_words = q ** n
    all_words_list = list(_all_words(q, n))
    word_to_idx = {w: i for i, w in enumerate(all_words_list)}
    
    # Build balls as sorted lists of word indices
    balls = {}
    for cw in all_words_list:
        ball_list = [word_to_idx[cw]]
        for i in range(n):
            for v in range(q):
                if v != cw[i]:
                    lst = list(cw)
                    lst[i] = v
                    ball_list.append(word_to_idx[tuple(lst)])
        balls[cw] = ball_list
    
    # Convert to numpy arrays for faster set operations
    balls_np = {cw: np.array(bs, dtype=np.int32) for cw, bs in balls.items()}
    
    # Uncovered as boolean array
    uncovered = np.ones(total_words, dtype=bool)
    codewords = []
    
    while uncovered.any() and time.time() < deadline:
        # Pick codeword covering most uncovered
        best_cw = None
        best_count = -1
        for cw in balls_np:
            cnt = np.sum(uncovered[balls_np[cw]])
            if cnt > best_count:
                best_count = cnt
                best_cw = cw
                if best_count == np.sum(uncovered):
                    break
        if best_cw is None or best_count <= 0:
            break
        codewords.append(best_cw)
        uncovered[balls_np[best_cw]] = False
    
    # Iterative removal
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        cover_count[balls_np[cw]] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(codewords)):
            cw = codewords[idx]
            if np.any(cover_count[balls_np[cw]] == 1):
                continue
            # Remove
            cover_count[balls_np[cw]] -= 1
            codewords.pop(idx)
            improved = True
            break
    
    return codewords

def _local_search(q, n, radius, codewords, rng, deadline):
    """
    Try to improve by swapping one codeword for another.
    """
    total_words = q ** n
    all_words_list = list(_all_words(q, n))
    word_to_idx = {w: i for i, w in enumerate(all_words_list)}
    
    # Build balls
    balls = {}
    for cw in all_words_list:
        ball_list = [word_to_idx[cw]]
        for i in range(n):
            for v in range(q):
                if v != cw[i]:
                    lst = list(cw)
                    lst[i] = v
                    ball_list.append(word_to_idx[tuple(lst)])
        balls[cw] = np.array(ball_list, dtype=np.int32)
    
    # Current coverage
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        cover_count[balls[cw]] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each codeword and see if we can re-cover with a different one
        for idx in range(len(codewords)):
            cw = codewords[idx]
            # Words that only this codeword covers
            only_covered = balls[cw][cover_count[balls[cw]] == 1]
            if len(only_covered) == 0:
                # Can remove
                cover_count[balls[cw]] -= 1
                codewords.pop(idx)
                improved = True
                break
            else:
                # Try to find a replacement that covers all only_covered words
                # and doesn't break anything else
                # This is expensive; skip for now in local search
                pass
        if not improved:
            # Try adding a random codeword and then removing one
            # Actually, just try random removals
            pass
    
    return codewords

def construct(q, n, radius):
    """
    Construct a covering code of size K_q(n, radius) upper bound.
    Returns a list of codewords, each a list of n integers in 0..q-1.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    rng = random.Random(42)
    
    # For the specific case q=3, n=7, radius=1:
    # Use fast greedy set-cover with numpy
    if q == 3 and n == 7 and radius == 1:
        codewords = _greedy_cover_fast(q, n, radius, rng, deadline)
        
        # Try local search / further removal if time permits
        if time.time() < deadline - 5:
            codewords = _local_search(q, n, radius, codewords, rng, deadline)
        
        return [list(cw) for cw in codewords]
    
    # General fallback: direct sum construction (same as seed)
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(q), repeat=n - 4)]
    
    # For small n, brute force
    all_words = list(_all_words(q, n))
    return all_words

# EVOLVE-BLOCK-END
