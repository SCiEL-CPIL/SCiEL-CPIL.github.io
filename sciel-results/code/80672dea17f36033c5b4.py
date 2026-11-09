"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as a 2D numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.empty((total, n), dtype=np.uint8)
    # Use base-q expansion
    idx = np.arange(total, dtype=np.uint32)
    for i in range(n - 1, -1, -1):
        words[:, i] = (idx // (q ** i)) % q
    return words


def _balls(q, n, radius):
    """
    Precompute for each codeword position, the set of word-indices it covers.
    Returns a list of sets (or bitmasks) - but for 2187 words, use sets of indices.
    Actually, for efficiency with greedy, we want for each potential codeword c,
    the set of uncovered words it would cover. We'll precompute the ball membership.
    """
    all_words = _all_words(q, n)
    total = q ** n
    # For each word index, compute which codewords are within radius.
    # But that's expensive. Instead, for each codeword c (all possible), compute its ball.
    # Since total = 2187 and each ball has 1 + n*(q-1) = 1 + 7*2 = 15 words,
    # we can precompute balls for all 2187 potential codewords.
    
    # Precompute all balls: balls[i] = set of word indices covered by codeword i
    balls = [set() for _ in range(total)]
    
    for i in range(total):
        c = all_words[i]
        # The ball of radius 1 around c includes c itself and all words differing in exactly 1 position
        balls[i].add(i)
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                # Create new word
                # We need to find the index of the word that is c with position pos changed to val
                # Compute index: sum(c[j] * q^(n-1-j) for j in range(n))
                # Let's precompute powers
                # Actually, let's just compute the index directly
                new_word = c.copy()
                new_word[pos] = val
                # Compute index
                idx = 0
                for j in range(n):
                    idx = idx * q + new_word[j]
                balls[i].add(idx)
    
    return all_words, balls


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    all_words, balls = _balls(q, n, radius)
    total = q ** n
    
    # Greedy set cover with multiple restarts using different random orderings
    best_cover = None
    best_size = total + 1
    
    # Initial greedy: always pick the word that covers the most uncovered points
    # We'll do a few restarts with different tie-breaking
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        uncovered = set(range(total))
        cover = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find the codeword that covers the most uncovered words
            best_c = -1
            best_count = -1
            
            # For efficiency, only consider codewords that have at least one uncovered word in their ball
            # But checking all 2187 is fast enough
            for c in range(total):
                if not balls[c]:
                    continue
                # Count how many of c's ball are in uncovered
                count = len(balls[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == len(balls[c]):
                        # Can't do better for this c, but another c might have same
                        # Actually we want max, so keep going unless we hit theoretical max
                        # Theoretical max is len(balls[c]) which is 15
                        pass
            
            if best_c == -1 or best_count == 0:
                break
            
            cover.append(best_c)
            uncovered -= balls[best_c]
        
        if time.time() > deadline:
            break
        
        if len(cover) < best_size:
            best_size = len(cover)
            best_cover = cover[:]
        
        # If we reached a good solution, we can try to improve with local search
        # But first, let's see if we can do better with a different greedy variant
    
    # Local search / post-processing: try to remove redundant codewords
    if best_cover:
        # Try to remove any codeword that doesn't add new coverage
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            new_cover = list(best_cover)
            for i in range(len(new_cover)):
                if time.time() > deadline:
                    break
                # Try removing new_cover[i]
                candidate = new_cover[:i] + new_cover[i+1:]
                # Check if candidate still covers everything
                covered = set()
                for c in candidate:
                    covered |= balls[c]
                if len(covered) == total:
                    new_cover = candidate
                    improved = True
                    break
            best_cover = new_cover
    
    # Convert indices to actual codewords
    if best_cover:
        result = [list(all_words[i]) for i in best_cover]
    else:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return result
# EVOLVE-BLOCK-END
