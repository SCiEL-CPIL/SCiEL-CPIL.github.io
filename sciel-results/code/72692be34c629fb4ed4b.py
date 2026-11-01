"""Rewritten program for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def _neighbors_ternary(word, q=3, n=8):
    """Generate all words within Hamming distance 1 of the given word."""
    result = [list(word)]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                new_word = list(word)
                new_word[i] = v
                result.append(new_word)
    return result

def _word_to_int(word, q=3, n=8):
    """Convert a word (list of ints) to a single integer."""
    val = 0
    for v in word:
        val = val * q + v
    return val

def _int_to_word(val, q=3, n=8):
    """Convert an integer back to a word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = val % q
        val //= q
    return word

def _build_cover_map(codewords, q=3, n=8):
    """Build a map from each covered word to the codeword index that covers it."""
    cover_map = {}
    for idx, cw in enumerate(codewords):
        for nb in _neighbors_ternary(cw, q, n):
            key = _word_to_int(nb, q, n)
            if key not in cover_map:
                cover_map[key] = idx
    return cover_map

def _find_uncovered(cover_map, q=3, n=8):
    """Find all words not in the cover map."""
    total = q ** n
    uncovered = set()
    for i in range(total):
        if i not in cover_map:
            uncovered.add(i)
    return uncovered

def _greedy_init(q=3, n=8, radius=1, time_limit=30.0):
    """Greedy initialization: repeatedly pick the codeword that covers the most uncovered words."""
    start_time = time.time()
    
    # Precompute all words and their neighbors
    all_words = [list(_int_to_word(i, q, n)) for i in range(q ** n)]
    
    # For each potential codeword, precompute its covered set
    # We'll track coverage incrementally
    
    # Start with an empty set of codewords
    codewords = []
    covered = set()
    
    # We'll use a priority approach: for each candidate, count how many new words it covers
    # To speed this up, we'll maintain a count of how many codewords cover each word
    
    # Actually, let's use a simpler approach: repeatedly find the word with max uncovered neighbors
    # and add a codeword near it
    
    total_words = q ** n
    remaining = set(range(total_words))
    
    # Precompute neighbor sets for all words
    # This is expensive but we only do it once
    # Actually, let's be smarter: for each word, its "ball" is itself + all words at distance 1
    
    while remaining and (time.time() - start_time) < time_limit:
        # Find the word in remaining that has the most uncovered neighbors
        # This is expensive, so let's sample
        best_word = None
        best_count = -1
        
        # Sample a subset of remaining words
        sample_size = min(len(remaining), 500)
        if len(remaining) <= sample_size:
            candidates = list(remaining)
        else:
            candidates = random.sample(list(remaining), sample_size)
        
        for w in candidates:
            # Count how many of w's neighbors are still uncovered
            count = 0
            word = _int_to_word(w, q, n)
            for nb in _neighbors_ternary(word, q, n):
                nb_int = _word_to_int(nb, q, n)
                if nb_int in remaining:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None:
            break
        
        # Add best_word as a codeword (or a word near it)
        # Actually, we should add the word itself as a codeword
        cw = _int_to_word(best_word, q, n)
        codewords.append(cw)
        
        # Remove all covered words from remaining
        for nb in _neighbors_ternary(cw, q, n):
            nb_int = _word_to_int(nb, q, n)
            remaining.discard(nb_int)
    
    return codewords

def _local_search(codewords, q=3, n=8, radius=1, time_limit=60.0):
    """Local search: try to remove codewords and add better ones."""
    start_time = time.time()
    
    # Build cover map
    cover_map = _build_cover_map(codewords, q, n)
    
    # Count how many times each codeword is the "primary" cover for each word
    # A codeword can be removed if no word is covered only by it
    word_to_cw = defaultdict(list)
    for idx, cw in enumerate(codewords):
        for nb in _neighbors_ternary(cw, q, n):
            key = _word_to_int(nb, q, n)
            word_to_cw[key].append(idx)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try to remove each codeword
        for idx in range(len(codewords) - 1, -1, -1):
            if (time.time() - start_time) > time_limit:
                break
            
            # Check if this codeword is the only cover for any word
            can_remove = True
            for nb in _neighbors_ternary(codewords[idx], q, n):
                key = _word_to_int(nb, q, n)
                if len(word_to_cw[key]) == 1 and word_to_cw[key][0] == idx:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove this codeword
                for nb in _neighbors_ternary(codewords[idx], q, n):
                    key = _word_to_int(nb, q, n)
                    word_to_cw[key].remove(idx)
                codewords.pop(idx)
                improved = True
        
        # Try to add a codeword that covers the most uncovered words
        # First, find uncovered words
        uncovered = set(range(q ** n))
        for key in word_to_cw:
            if word_to_cw[key]:
                uncovered.discard(key)
        
        if uncovered and (time.time() - start_time) < time_limit:
            # Find a good word to add
            best_word = None
            best_count = 0
            
            sample_size = min(len(uncovered), 300)
            if len(uncovered) <= sample_size:
                candidates = list(uncovered)
            else:
                candidates = random.sample(list(uncovered), sample_size)
            
            for w in candidates:
                count = 0
                word = _int_to_word(w, q, n)
                for nb in _neighbors_ternary(word, q, n):
                    nb_int = _word_to_int(nb, q, n)
                    if nb_int in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_word = w
            
            if best_word is not None and best_count > 0:
                cw = _int_to_word(best_word, q, n)
                codewords.append(cw)
                new_idx = len(codewords) - 1
                for nb in _neighbors_ternary(cw, q, n):
                    key = _word_to_int(nb, q, n)
                    word_to_cw[key].append(new_idx)
                improved = True
    
    return codewords

def construct(q, n, radius):
    """Construct a covering code for the given parameters."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # For the specific case q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        # Use the direct sum construction as initial solution: 486 words
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Apply local search to try to reduce the number of codewords
        if time.time() < deadline:
            codewords = _local_search(codewords, q, n, radius, time_limit=100.0)
        
        return codewords
    else:
        # General case: use greedy initialization
        codewords = _greedy_init(q, n, radius, time_limit=30.0)
        if time.time() < deadline:
            codewords = _local_search(codewords, q, n, radius, time_limit=70.0)
        return codewords
# EVOLVE-BLOCK-END
