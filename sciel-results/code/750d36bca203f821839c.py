"""Greedy set-cover with redundancy removal for K_3(7,1)."""
import time
import random
import numpy as np

def construct(q, n, radius):
    """Construct a covering code for the given parameters."""
    # For this task, q=3, n=7, radius=1
    # We'll implement a general approach but optimize for this specific case.
    
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 120s limit
    
    # If n is small enough, enumerate all words
    total_words = q ** n
    if total_words > 1000000:
        # Fallback to a simple construction for large spaces
        # This shouldn't happen for the given task (3^7 = 2187)
        return _fallback_construction(q, n, radius)
    
    # Represent each word as an integer 0..total_words-1
    # Precompute neighbors for each word (words at Hamming distance <= radius)
    # For radius=1, neighbors are the word itself and words differing in exactly one position
    
    # Helper: convert word (tuple of digits) to integer index
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx
    
    def idx_to_word(idx):
        word = []
        for i in range(n):
            word.append(idx % q)
            idx //= q
        return word[::-1]
    
    # Precompute all neighbors for each word
    neighbors = []
    for idx in range(total_words):
        word = idx_to_word(idx)
        nbrs = set()
        nbrs.add(idx)  # distance 0
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                new_word = word[:]
                new_word[pos] = new_val
                nbrs.add(word_to_idx(tuple(new_word)))
        neighbors.append(list(nbrs))
    
    # Greedy set cover:
    # - Maintain a set of uncovered words
    # - At each step, pick the word that covers the most uncovered words
    # - Add it to the code, mark its neighbors as covered
    # - Continue until all words are covered
    
    best_code = None
    best_len = float('inf')
    
    # Multiple random restarts
    num_restarts = 20
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        # Random initial selection to break symmetry
        # Pick a random starting word
        start_word = rng.randint(0, total_words - 1)
        
        while np.any(uncovered):
            # Find the word that covers the most uncovered neighbors
            best_idx = -1
            best_count = -1
            
            # Only consider words that have at least one uncovered neighbor
            # For speed, we can sample if total_words is large, but 2187 is small
            for idx in range(total_words):
                if not uncovered[idx]:
                    # This word is already covered, but it might still be useful
                    # Actually, we should consider all words, not just uncovered ones
                    # Wait: if a word is covered, it can still be a good candidate
                    # Let's reconsider: we want to maximize the number of uncovered neighbors
                    pass
                
                count = np.sum(uncovered[neighbors[idx]])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1 or best_count == 0:
                # This shouldn't happen if uncovered is non-empty
                break
            
            code.append(best_idx)
            # Mark neighbors as covered
            uncovered[neighbors[best_idx]] = False
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
        
        # Try to improve with a different random seed or strategy
        # For now, just do more restarts with different random starts
    
    # Redundancy removal
    if best_code is not None:
        best_code = _remove_redundancy(best_code, neighbors, total_words)
    
    # Convert back to list of words
    if best_code is None:
        # Fallback
        return _fallback_construction(q, n, radius)
    
    result = [idx_to_word(idx) for idx in best_code]
    
    # Verify coverage (optional, for debugging)
    # assert _verify_coverage(result, q, n, radius)
    
    return result


def _remove_redundancy(code, neighbors, total_words):
    """Remove redundant codewords from the code."""
    code = list(code)
    
    # Count how many times each word is covered
    cover_count = np.zeros(total_words, dtype=int)
    for idx in code:
        cover_count[neighbors[idx]] += 1
    
    changed = True
    while changed:
        changed = False
        new_code = []
        for idx in code:
            # Check if all neighbors of idx are covered at least twice
            if np.min(cover_count[neighbors[idx]]) >= 2:
                # This codeword is redundant
                cover_count[neighbors[idx]] -= 1
                changed = True
            else:
                new_code.append(idx)
        code = new_code
    
    return code


def _fallback_construction(q, n, radius):
    """Fallback: use the tetracode product construction."""
    import itertools
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        else:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Very crude fallback
        return [[0] * n]


def _verify_coverage(code, q, n, radius):
    """Verify that every word is within distance radius of some codeword."""
    if not code:
        return n == 0
    
    # Build a boolean array of covered words
    covered = [False] * (q ** n)
    
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx
    
    for cw in code:
        # Mark all words within distance radius
        # For small radius, enumerate
        from itertools import product
        positions = range(n)
        for delta in product(range(q ** radius + 1), repeat=0):  # This won't work generically
            pass
        
        # Simpler: check all words in the space
        # This is O(|code| * q^n * n) which is fine for small spaces
        pass
    
    # Actually, let's just do a direct check
    covered_set = set()
    for cw in code:
        for word_tuple in product(range(q), repeat=n):
            dist = sum(1 for a, b in zip(cw, word_tuple) if a != b)
            if dist <= radius:
                covered_set.add(word_tuple)
    
    total = q ** n
    return len(covered_set) == total

BET: {mechanism: "Greedy set cover with redundancy removal on the 2187-word space should find a code near 186-190, beating the 243 product construction.", q: 0.7, delta: [0.02, 0.15]}
