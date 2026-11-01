"""Football pool problem K_3(7,1): minimize number of codewords covering all 3^7 words."""
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s wall-clock

    # For the specific task: q=3, n=7, radius=1
    # We use a greedy set-cover approach with efficient coverage tracking
    if q != 3 or n != 7 or radius != 1:
        # Fallback: use the tetracode construction for general cases
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(tail) for tail in itertools.product(range(q), repeat=n)]

    # Efficient greedy construction for K_3(7,1)
    # Each codeword covers 1 + 7*2 = 15 words (itself + 7 positions * 2 alternatives)
    # Total space: 3^7 = 2187 words
    # Lower bound: 2187/15 = 145.8, so at least 146 words needed
    # Best known: 186

    # Precompute all words and their neighbors
    # Represent each word as an integer 0..2186 (base-3 encoding)
    all_words = list(range(2187))
    
    # Precompute coverage: for each word w, the set of words it covers (w and its neighbors)
    # A word covers itself and all words at Hamming distance 1
    def word_to_list(w):
        """Convert integer to base-3 list of length n."""
        result = [0] * n
        for i in range(n - 1, -1, -1):
            result[i] = w % 3
            w //= 3
        return result

    def list_to_word(lst):
        """Convert base-3 list to integer."""
        w = 0
        for v in lst:
            w = w * 3 + v
        return w

    # Precompute neighbor sets for each word
    # neighbors[w] = set of words within Hamming distance radius of w
    # This includes w itself
    neighbors = {}
    for w in all_words:
        lst = word_to_list(w)
        covered = set()
        covered.add(w)
        for pos in range(n):
            for new_val in range(3):
                if new_val != lst[pos]:
                    lst[pos] = new_val
                    covered.add(list_to_word(lst))
                    lst[pos] = w % (3 ** (n - 1 - pos)) // (3 ** (n - 1 - pos))
        # Recompute original value properly
        orig = word_to_list(w)
        for pos in range(n):
            lst[pos] = orig[pos]
        neighbors[w] = covered

    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    uncovered = set(all_words)
    codewords = []
    
    # For efficiency, track count of uncovered words covered by each candidate
    # Initially, every word covers 15 words (including itself)
    # We'll use a priority approach
    
    while len(uncovered) > 0:
        if time.time() > deadline:
            break
        
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        best_covered = None
        
        # Sample-based search for speed: check a random subset plus some heuristics
        # First, try all words that are themselves uncovered (likely good choices)
        candidates = list(uncovered)
        # Shuffle for diversity
        np.random.shuffle(candidates)
        
        # Check first 200 candidates in detail
        for w in candidates[:200]:
            cnt = sum(1 for x in neighbors[w] if x in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_word = w
                best_covered = neighbors[w]
        
        # Also check some random words from the full space
        for _ in range(50):
            w = np.random.randint(0, 2187)
            cnt = sum(1 for x in neighbors[w] if x in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_word = w
                best_covered = neighbors[w]
        
        if best_word is None:
            best_word = candidates[0]
            best_covered = neighbors[best_word]
        
        # Add the best word to our code
        codewords.append(word_to_list(best_word))
        uncovered -= best_covered
    
    # Verify coverage
    if len(uncovered) > 0:
        # Emergency: add all remaining uncovered words
        for w in uncovered:
            codewords.append(word_to_list(w))
    
    # Remove duplicates (shouldn't happen but just in case)
    seen = set()
    unique_codewords = []
    for cw in codewords:
        key = tuple(cw)
        if key not in seen:
            seen.add(key)
            unique_codewords.append(cw)
    
    return unique_codewords
