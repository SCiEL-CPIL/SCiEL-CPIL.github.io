"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Deadline guard: evaluator kills at 150s; use 110s to be safe.
    deadline = time.time() + 110.0
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: direct product of optimal small code
        # For ternary, length 4, radius 1: tetracode (9 words)
        # General fallback: greedy set cover
        return _greedy_cover(q, n, radius, deadline)
    
    # Known best: 486 words. We'll try to find <= 486.
    # Strategy:
    # 1. Start with a good initial covering (e.g., direct sum of tetracode + all tails = 9*81=729 words)
    # 2. Use greedy removal to reduce the set
    # 3. Use local search to swap words for better coverage
    
    # Step 1: Initial covering
    # The tetracode is a perfect code of length 4, radius 1 over GF(3)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Direct sum: 9 * 3^4 = 729 words
    initial_cover = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Step 2: Greedy removal
    # For each word in the cover, check if removing it still covers all 3^8 = 6561 words.
    # A word w covers all words at distance <= 1 from w, which is 1 + 8*2 = 17 words.
    # We need to check if every word is still covered by some other word in the cover.
    
    # To speed this up, we'll use a coverage count array.
    # coverage_count[x] = number of words in the cover that cover x.
    # A word x is covered if coverage_count[x] >= 1.
    
    def _coverage_count(cover, q, n, radius):
        # Returns an array of size q^n where each entry is the count of covering words.
        total = q ** n
        counts = np.zeros(total, dtype=np.int32)
        
        # For each word in the cover, add 1 to all words within radius.
        # For radius 1, that's the word itself + all single-position changes.
        for w in cover:
            idx = _encode(w, q, n)
            counts[idx] += 1
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        w2 = w[:pos] + [val] + w[pos+1:]
                        idx2 = _encode(w2, q, n)
                        counts[idx2] += 1
        return counts
    
    def _encode(w, q, n):
        # Encode a word to an integer index.
        idx = 0
        for i, d in enumerate(w):
            idx = idx * q + d
        return idx
    
    # Compute initial coverage counts
    counts = _coverage_count(initial_cover, q, n, radius)
    
    # Greedy removal: try to remove words that have coverage_count == 1 for some unique word they cover.
    # Actually, we can remove a word if all words it covers are also covered by other words.
    # i.e., for all x in ball(w, radius), counts[x] >= 2.
    
    cover = initial_cover[:]
    cover_set = set(tuple(w) for w in cover)
    
    # Try to remove words in a random order
    random.shuffle(cover)
    
    for i, w in enumerate(cover):
        if time.time() > deadline - 5.0:
            break
        # Check if w can be removed
        can_remove = True
        for x in _ball(w, q, n, radius):
            idx = _encode(x, q, n)
            if counts[idx] <= 1:
                can_remove = False
                break
        if can_remove:
            # Remove w
            for x in _ball(w, q, n, radius):
                idx = _encode(x, q, n)
                counts[idx] -= 1
            cover.pop(i)
            i -= 1
    
    # Step 3: Local search - try to replace pairs of words with a single word if possible,
    # or swap words to improve coverage.
    # Actually, let's try a different approach: use a greedy set cover from scratch, but with a better heuristic.
    
    # Alternative: use the known 486-word code structure if we can find it.
    # The best known code is 486 words. Let's try to construct it.
    # Actually, let's just return what we have if it's <= 486, otherwise try to improve.
    
    if len(cover) <= 486:
        return cover
    
    # Try local search: randomly swap a word with another word to see if we can reduce the set.
    # Actually, let's try a different approach: start with a smaller initial set and use greedy addition.
    
    # For now, return the current cover.
    return cover

def _ball(w, q, n, radius):
    """Return all words within Hamming distance radius of w."""
    if radius == 0:
        return [w]
    if radius == 1:
        result = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    result.append(w[:pos] + [val] + w[pos+1:])
        return result
    # General case (not needed for this task)
    from itertools import combinations, product
    result = [w]
    for r in range(1, radius + 1):
        for positions in combinations(range(n), r):
            for values in product(range(q), repeat=r):
                if all(values[i] != w[positions[i]] for i in range(r)):
                    w2 = w[:]
                    for i, pos in enumerate(positions):
                        w2[pos] = values[i]
                    result.append(w2)
    return result

def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    total = q ** n
    covered = [False] * total
    
    def _encode(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    cover = []
    remaining = [True] * total
    
    while time.time() < deadline - 2.0:
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        # Sample a subset of candidate words for speed
        candidates = [list(map(int, str(i).zfill(n).rjust(n, '0'))) for i in range(0, total, max(1, total // 1000))]
        for w in candidates:
            count = 0
            for x in _ball(w, q, n, radius):
                idx = _encode(x)
                if remaining[idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        if best_count <= 0:
            break
        # Add best_word to cover
        cover.append(best_word)
        for x in _ball(best_word, q, n, radius):
            idx = _encode(x)
            remaining[idx] = False
    
    return cover
# EVOLVE-BLOCK-END
