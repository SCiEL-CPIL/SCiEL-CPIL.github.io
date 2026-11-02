"""Improved program: greedy + local search for K_3(6,1) covering code."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 140s safety margin under 150s limit


def _time_left():
    return _DEADLINE - time.time()


def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _ball(word, q, n, radius):
    """Generate all words within Hamming distance <= radius of `word`."""
    if radius == 0:
        yield word
        return
    positions = list(range(n))
    # distance 0
    yield word
    # distance 1
    for i in positions:
        for v in range(q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                yield w
    # distance 2
    for i, j in itertools.combinations(positions, 2):
        for vi in range(q):
            if vi == word[i]:
                continue
            for vj in range(q):
                if vj == word[j]:
                    continue
                w = list(word)
                w[i] = vi
                w[j] = vj
                yield w
    # distance 3
    if radius >= 3:
        for i, j, k in itertools.combinations(positions, 3):
            for vi in range(q):
                if vi == word[i]:
                    continue
                for vj in range(q):
                    if vj == word[j]:
                        continue
                for vk in range(q):
                    if vk == word[k]:
                        continue
                    w = list(word)
                    w[i] = vi
                    w[j] = vj
                    w[k] = vk
                    yield w


def _is_covered(word, codewords, q, n, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _coverage_count(word, uncovered_set, q, n, radius):
    """Count how many words in uncovered_set are within radius of word."""
    count = 0
    for w in _ball(word, q, n, radius):
        if w in uncovered_set:
            count += 1
    return count


def _greedy_cover(q, n, radius):
    """Greedy set cover: pick word covering most uncovered words each step."""
    all_words = set(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    codewords = []
    
    while uncovered:
        if _time_left() < 2.0:
            break
        # Sample candidate words to find a good one (not exhaustive for speed)
        # First try a few random words, then pick the best
        best_word = None
        best_count = -1
        
        # Try some random candidates
        n_samples = min(200, len(uncovered))
        candidates = list(uncovered)
        random.shuffle(candidates)
        candidates = candidates[:n_samples]
        
        for cand in candidates:
            cnt = _coverage_count(cand, uncovered, q, n, radius)
            if cnt > best_count:
                best_count = cnt
                best_word = cand
        
        if best_word is None:
            best_word = candidates[0]
            best_count = _coverage_count(best_word, uncovered, q, n, radius)
        
        # Add the chosen word
        codewords.append(list(best_word))
        # Remove all covered words
        for w in _ball(best_word, q, n, radius):
            uncovered.discard(w)
    
    return codewords


def _local_improve(codewords, q, n, radius, max_iters=2000):
    """Try to reduce codeword count by replacing or removing words."""
    codewords = [list(c) for c in codewords]
    all_words = set(itertools.product(range(q), repeat=n))
    
    def _uncovered_after_removal(cw_idx):
        """Return set of words that would become uncovered if we remove codeword at cw_idx."""
        # Words covered by this codeword
        this_ball = set(_ball(codewords[cw_idx], q, n, radius))
        # For each word in this_ball, check if still covered by another codeword
        newly_uncovered = set()
        for w in this_ball:
            covered = False
            for j, cw in enumerate(codewords):
                if j == cw_idx:
                    continue
                if _hamming_dist(w, cw) <= radius:
                    covered = True
                    break
            if not covered:
                newly_uncovered.add(w)
        return newly_uncovered
    
    def _try_replace(cw_idx, new_word):
        """Try replacing codeword at cw_idx with new_word. Return new list or None."""
        # Check: words covered by old but not new must be covered by others
        old_ball = set(_ball(codewords[cw_idx], q, n, radius))
        new_ball = set(_ball(new_word, q, n, radius))
        
        # Words that old covers but new doesn't
        lost = old_ball - new_ball
        for w in lost:
            covered = False
            for j, cw in enumerate(codewords):
                if j == cw_idx:
                    continue
                if _hamming_dist(w, cw) <= radius:
                    covered = True
                    break
            if not covered:
                return None
        return None  # If we can't reduce count, replacement doesn't help directly
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _time_left() > 2.0:
        improved = False
        iters += 1
        
        # Try removing each codeword
        for i in range(len(codewords)):
            if _time_left() < 2.0:
                break
            newly_uncovered = _uncovered_after_removal(i)
            if not newly_uncovered:
                # Safe to remove
                codewords.pop(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing: for each codeword, try some random new words
        for i in range(len(codewords)):
            if _time_left() < 2.0:
                break
            # Generate candidate replacements from uncovered or random
            # A good replacement should cover words that are only covered by codeword i
            this_ball = set(_ball(codewords[i], q, n, radius))
            
            # Find words only covered by this codeword
            only_here = set()
            for w in this_ball:
                only_here_flag = True
                for j, cw in enumerate(codewords):
                    if j == i:
                        continue
                    if _hamming_dist(w, cw) <= radius:
                        only_here_flag = False
                        break
                if only_here_flag:
                    only_here.add(w)
            
            if not only_here:
                continue
            
            # Try replacing with a word that covers many of only_here
            best_new = None
            best_gain = 0
            for _ in range(50):
                # Pick a random word near only_here
                if only_here:
                    anchor = random.choice(list(only_here))
                else:
                    anchor = tuple(random.randint(0, q-1) for _ in range(n))
                # Perturb
                w = list(anchor)
                for _ in range(random.randint(0, 2)):
                    pos = random.randint(0, n-1)
                    w[pos] = random.randint(0, q-1)
                w = tuple(w)
                
                # Check if this replacement helps
                new_ball = set(_ball(w, q, n, radius))
                # Words we'd lose (covered by old but not new)
                lost = this_ball - new_ball
                # Check if lost words are still covered by others
                valid = True
                for lw in lost:
                    ok = False
                    for j, cw in enumerate(codewords):
                        if j == i:
                            continue
                        if _hamming_dist(lw, cw) <= radius:
                            ok = True
                            break
                    if not ok:
                        valid = False
                        break
                if valid:
                    # Count how many only_here words we now cover
                    gain = len(only_here & new_ball)
                    if gain > best_gain:
                        best_gain = gain
                        best_new = w
            
            if best_new is not None and best_gain > 0:
                codewords[i] = list(best_new)
                improved = True
                break
    
    return codewords


def construct(q, n, radius):
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0
    
    random.seed(42)
    
    best = None
    best_len = float('inf')
    
    # Run multiple greedy + local search iterations with different seeds
    for attempt in range(20):
        if _time_left() < 10.0:
            break
        
        random.seed(attempt * 12345 + 678)
        
        # Greedy cover
        codewords = _greedy_cover(q, n, radius)
        
        if len(codewords) < best_len:
            best_len = len(codewords)
            best = [list(c) for c in codewords]
        
        # Local improvement
        if _time_left() > 5.0:
            improved = _local_improve(codewords, q, n, radius, max_iters=500)
            if len(improved) < best_len:
                best_len = len(improved)
                best = [list(c) for c in improved]
        
        # Additional local search on best
        if best is not None and _time_left() > 5.0:
            improved = _local_improve(best, q, n, radius, max_iters=300)
            if len(improved) < best_len:
                best_len = len(improved)
                best = [list(c) for c in improved]
        
        if best_len <= 73:
            break
    
    if best is None:
        # Fallback: simple direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best
# EVOLVE-BLOCK-END
