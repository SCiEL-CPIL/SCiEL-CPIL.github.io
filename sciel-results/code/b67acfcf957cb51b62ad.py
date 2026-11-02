"""Ternary covering code K_3(7,1) via greedy construction + local search removal."""
import time
import itertools
import numpy as np

# Deadline guard: evaluator kills at 150s; we return well before that.
_DEADLINE = 140.0  # seconds


def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists of equal length."""
    return sum(x != y for x, y in zip(a, b))


def _covers(word, codewords, radius):
    """Check if word is within radius of any codeword."""
    for cw in codewords:
        d = _hamming_dist(word, cw)
        if d <= radius:
            return True
    return False


def _is_covered(all_words, codewords, radius):
    """Check if all words are covered by codewords within radius."""
    for w in all_words:
        if not _covers(w, codewords, radius):
            return False
    return True


def _greedy_construct(q, n, radius):
    """Greedy set cover: repeatedly pick the codeword that covers the most uncovered words."""
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Precompute coverage: for each possible codeword, which words it covers
    # A codeword c covers word w if dist(c, w) <= radius
    # For radius=1, a codeword covers itself and all words at distance 1.
    # Number of words covered by one codeword: 1 + n*(q-1)
    
    uncovered = set(range(N))  # indices of uncovered words
    
    # Map word index to word
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    codewords = []
    
    while uncovered:
        if time.time() > _DEADLINE - 10:
            break
        
        # For each candidate codeword, count how many uncovered words it covers
        # To be efficient, we only need to consider candidates that are "useful"
        # i.e., they cover at least one uncovered word.
        
        # For radius 1: candidate c covers c itself and neighbors.
        # Let's iterate over all possible codewords and pick the best.
        # But 3^7 = 2187 candidates, and for each we check up to 1+7*2=15 words.
        # That's 2187 * 15 = ~32k checks per iteration. With ~200 iterations, ~6.4M total. Fine.
        
        best_cand = None
        best_count = 0
        best_covered = set()
        
        for cand in itertools.product(range(q), repeat=n):
            # Find all uncovered words within radius of cand
            covered_here = set()
            for uidx in uncovered:
                w = all_words[uidx]
                if _hamming_dist(cand, w) <= radius:
                    covered_here.add(uidx)
            if len(covered_here) > best_count:
                best_count = len(covered_here)
                best_cand = cand
                best_covered = covered_here
        
        if best_cand is None:
            break
        
        codewords.append(best_cand)
        uncovered -= best_covered
    
    return codewords, all_words


def _try_remove_redundant(codewords, all_words, radius, time_budget):
    """Try to remove codewords one by one if the code remains valid."""
    codewords = list(codewords)
    removed = 0
    start = time.time()
    
    # We'll iterate multiple passes until no more removals are possible or time runs out
    progress = True
    while progress and time.time() < time_budget:
        progress = False
        i = 0
        while i < len(codewords) and time.time() < time_budget:
            # Try removing codewords[i]
            cw = codewords[i]
            remaining = codewords[:i] + codewords[i+1:]
            
            # Check if all words are still covered
            # For efficiency, only check words that were previously covered ONLY by cw
            # Simple approach: check all words
            valid = True
            for w in all_words:
                covered = False
                for rcw in remaining:
                    if _hamming_dist(w, rcw) <= radius:
                        covered = True
                        break
                if not covered:
                    valid = False
                    break
            
            if valid:
                codewords.pop(i)
                removed += 1
                progress = True
                # Don't increment i, check same position again
            else:
                i += 1
    
    return codewords, removed


def _swap_local_search(codewords, all_words, radius, q, n, time_budget):
    """Try swapping codewords to find a smaller code.
    
    Strategy: For each codeword, try replacing it with another codeword (not in the set)
    that allows further removals. Also try direct removal after perturbations.
    """
    codewords = list(codewords)
    start = time.time()
    best = codewords
    best_size = len(codewords)
    
    # Precompute all possible codewords
    all_cands = list(itertools.product(range(q), repeat=n))
    
    # For each position, try swapping with random candidates
    # Use a simple hill-climbing with restarts
    
    import random
    rng = random.Random(42)
    
    for _ in range(50):  # up to 50 restarts
        if time.time() > time_budget:
            break
        
        current = list(best)
        # Random perturbation: swap 1-3 codewords with random ones
        for _ in range(rng.randint(1, 3)):
            if len(current) == 0:
                break
            idx = rng.randint(0, len(current) - 1)
            new_cw = rng.choice(all_cands)
            # Make sure it's not already in the set
            if new_cw not in current:
                current[idx] = new_cw
        
        # Try to reduce
        current, _ = _try_remove_redundant(current, all_words, radius, time.time() + 5)
        
        if len(current) < best_size:
            best = current
            best_size = len(current)
    
    return best


def construct(q, n, radius):
    start_time = time.time()
    
    # Handle the specific case: q=3, n=7, radius=1
    # But make it general enough
    
    # Greedy construction
    codewords, all_words = _greedy_construct(q, n, radius)
    
    # Try to remove redundant codewords
    time_budget = start_time + (_DEADLINE - 15)  # leave 15s margin
    codewords, removed = _try_remove_redundant(codewords, all_words, radius, time_budget)
    
    # Local search with swaps
    if time.time() < time_budget - 10:
        codewords = _swap_local_search(codewords, all_words, radius, q, n, time_budget)
    
    # Final cleanup: one more pass of removal
    if time.time() < _DEADLINE - 5:
        codewords, _ = _try_remove_redundant(codewords, all_words, radius, _DEADLINE - 3)
    
    # Convert to lists
    return [list(cw) for cw in codewords]
