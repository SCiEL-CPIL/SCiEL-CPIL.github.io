"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

# Deadline guard
_DEADLINE = 110.0  # safety margin before 120s limit

def _time_left():
    return _DEADLINE - time.time()

def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time() + 110.0  # 110s budget out of 120s
    
    # For the specific case q=3, n=7, radius=1, we know the optimal is around 186.
    # We'll use a greedy set cover approach with good heuristics.
    
    # First, let's try the direct sum construction as a baseline
    if n >= 4 and q == 3:
        # Tetracode-based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        baseline = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        baseline = []
    
    # Try to improve with greedy set cover
    try:
        improved = _greedy_cover(q, n, radius, baseline)
        if len(improved) < len(baseline):
            return improved
    except Exception:
        pass
    
    return baseline


def _ball_words(word, q, n, radius):
    """Generate all words within Hamming distance <= radius from word."""
    if radius == 0:
        return [tuple(word)]
    if radius == 1:
        result = [tuple(word)]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    new_word = list(word)
                    new_word[i] = v
                    result.append(tuple(new_word))
        return result
    # General case using itertools combinations
    result = set()
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    if val != word[pos]:
                        new_word[pos] = val
                    else:
                        # Skip if same value (would be distance < r)
                        # Actually we need to be careful - let's just generate all and dedupe
                        pass
                result.add(tuple(new_word))
    # Better approach for radius 1 which is the common case
    return list(result)


def _greedy_cover(q, n, radius, initial_codewords=None):
    """Greedy set cover to find a small covering code."""
    total_words = q ** n
    
    # Represent words as integers for efficiency
    def word_to_int(word):
        result = 0
        for v in word:
            result = result * q + v
        return result
    
    def int_to_word(x):
        word = []
        for _ in range(n):
            word.append(x % q)
            x //= q
        return word
    
    # Precompute balls for all possible words? That's q^n * ball_size which might be large.
    # For q=3, n=7, total_words = 2187. Ball size for radius 1 = 1 + 7*2 = 15.
    # So 2187 * 15 = ~33k entries. That's manageable.
    
    # Build a mapping from word to its ball (set of words it covers)
    # Actually, we need: for each candidate center, what words does it cover?
    # And we need to track which words are uncovered.
    
    all_words_int = list(range(total_words))
    
    # Precompute the ball (as a set of ints) for each word
    # ball[i] = set of word ints within distance <= radius from word i
    balls = {}
    for i in range(total_words):
        word = int_to_word(i)
        ball = set()
        if radius == 1:
            ball.add(i)
            for pos in range(n):
                for v in range(q):
                    if v != word[pos]:
                        new_word = word.copy()
                        new_word[pos] = v
                        ball.add(word_to_int(new_word))
        elif radius == 0:
            ball.add(i)
        else:
            # General radius
            for r in range(radius + 1):
                for positions in itertools.combinations(range(n), r):
                    for values in itertools.product(range(q), repeat=r):
                        new_word = word.copy()
                        for pos, val in zip(positions, values):
                            new_word[pos] = val
                        ball.add(word_to_int(new_word))
        balls[i] = ball
    
    # Greedy set cover
    uncovered = set(all_words_int)
    codewords = []
    
    # If we have initial codewords, start with them
    if initial_codewords:
        for cw in initial_codewords:
            cw_int = word_to_int(cw)
            if cw_int in uncovered:
                codewords.append(cw)
                uncovered -= balls[cw_int]
            elif any(cw == existing for existing in codewords):
                pass
            else:
                codewords.append(cw)
                uncovered -= balls[cw_int]
    
    # Continue greedy selection
    while uncovered and _time_left() > 5.0:
        # Find the word whose ball covers the most uncovered words
        best_center = None
        best_count = -1
        for center in list(uncovered):
            count = len(uncovered & balls[center])
            if count > best_count:
                best_count = count
                best_center = center
                if best_count == len(uncovered):
                    break
        
        if best_center is None or best_count <= 0:
            break
        
        codewords.append(int_to_word(best_center))
        uncovered -= balls[best_center]
    
    return codewords


# EVOLVE-BLOCK-END
