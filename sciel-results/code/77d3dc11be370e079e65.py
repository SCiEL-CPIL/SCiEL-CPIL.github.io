"""Rewritten program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time


def _ball_radius1(word, q, n):
    """Generate all words within Hamming distance 1 of the given word."""
    yield tuple(word)
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                new_word = list(word)
                new_word[i] = v
                yield tuple(new_word)


def _greedy_cover(q, n, radius, time_limit=110.0):
    """Greedy covering code algorithm."""
    start_time = time.time()
    
    # Generate all words
    all_words = set()
    for i in range(q ** n):
        word = [0] * n
        x = i
        for j in range(n):
            word[j] = x % q
            x //= q
        all_words.add(tuple(word))
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        # Find the word that covers the most uncovered words
        best_count = 0
        best_word = None
        
        # Sample candidates from uncovered words (they're the hardest to cover)
        uncovered_list = list(uncovered)
        
        for candidate in uncovered_list:
            count = 0
            for w in _ball_radius1(candidate, q, n):
                if w in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = candidate
                if best_count == 15:  # Full ball coverage
                    break
        
        if best_word is None:
            break
        
        code.append(list(best_word))
        # Remove covered words
        for w in _ball_radius1(best_word, q, n):
            uncovered.discard(w)
    
    return code


def construct(q, n, radius):
    """Construct a covering code for the given parameters."""
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin
    
    if q == 3 and n == 7 and radius == 1:
        result = _greedy_cover(q, n, radius, time_limit=110.0)
        if time.time() > deadline:
            return result
        
        # Verify and return
        return result
    
    else:
        result = _greedy_cover(q, n, radius, time_limit=110.0)
        return result

# EVOLVE-BLOCK-END
