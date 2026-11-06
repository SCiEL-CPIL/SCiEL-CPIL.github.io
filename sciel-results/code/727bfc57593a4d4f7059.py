"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _can_add(word, existing, min_dist=6):
    """Check if word can be added to existing set while maintaining min_dist."""
    for w in existing:
        if (word ^ w).bit_count() < min_dist:
            return False
    return True

def _greedy_build(seed, min_dist=6, time_limit=5.0, rng=None):
    """Greedy construction starting from a seed word."""
    if rng is None:
        rng = random.Random(seed)
    
    words = [0]  # Always include zero
    if len(words) > 1:
        # Check seed is valid
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                if (words[i] ^ words[j]).bit_count() < min_dist:
                    return words
    
    start = time.time()
    # Precompute for faster checking
    while time.time() - start < time_limit:
        c = rng.randint(0, (1 << 18) - 1)
        if c in words:
            continue
        if _can_add(c, words, min_dist):
            words.append(c)
    
    return words

def _local_search(words, min_dist=6, time_limit=5.0, rng=None):
    """Try to add more words via local search: flip single bits of existing words."""
    if rng is None:
        rng = random.Random(42)
    
    start = time.time()
    words_set = set(words)
    words_list = list(words)
    
    while time.time() - start < time_limit:
        progress = False
        # Try flipping one bit of each existing word
        for idx, w in enumerate(words_list):
            for bit in range(18):
                candidate = w ^ (1 << bit)
                if candidate in words_set:
                    continue
                if _can_add(candidate, words_set, min_dist):
                    words_set.add(candidate)
                    words_list.append(candidate)
                    progress = True
                    break
            if progress:
                break
        
        if not progress:
            # Try random words
            for _ in range(100):
                c = rng.randint(0, (1 << 18) - 1)
                if c in words_set:
                    continue
                if _can_add(c, words_set, min_dist):
                    words_set.add(c)
                    words_list.append(c)
                    progress = True
                    break
        
        if not progress:
            # Try flipping two bits
            for idx, w in enumerate(words_list):
                found = False
                for b1 in range(18):
                    if found:
                        break
                    for b2 in range(b1+1, 18):
                        candidate = w ^ (1 << b1) ^ (1 << b2)
                        if candidate in words_set:
                            continue
                        if _can_add(candidate, words_set, min_dist):
                            words_set.add(candidate)
                            words_list.append(candidate)
                            progress = True
                            found = True
                            break
                if found:
                    break
            if not progress:
                break
    
    return list(words_set)

def construct():
    """
    Construct the largest possible collection of 18-bit words with pairwise Hamming distance at least 6.
    """
    start_time = time.time()
    total_time_limit = 55.0
    
    # Multiple random restarts with greedy + local search
    best_words = []
    
    # First, try a deterministic seed
    words1 = _greedy_build(seed=0, min_dist=6, time_limit=2.0)
    words1 = _local_search(words1, min_dist=6, time_limit=3.0, rng=random.Random(1))
    
    if len(words1) > len(best_words):
        best_words = words1
    
    # Try several random restarts
    remaining_time = total_time_limit - (time.time() - start_time)
    num_restarts = max(1, int(remaining_time / 5))
    
    for i in range(num_restarts):
        if time.time() - start_time >= total_time_limit - 1.0:
            break
        seed = i * 7919 + 1
        words_i = _greedy_build(seed=seed, min_dist=6, time_limit=2.0)
        words_i = _local_search(words_i, min_dist=6, time_limit=2.0, rng=random.Random(seed))
        
        if len(words_i) > len(best_words):
            best_words = words_i
    
    return best_words
# EVOLVE-BLOCK-END
