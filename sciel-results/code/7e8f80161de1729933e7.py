"""Greedy Set Cover with Local Search for K_3(8,1).

This program uses a greedy set cover algorithm to construct a covering code,
followed by a greedy removal phase to eliminate redundant codewords.
This structural change moves away from the fixed algebraic direct sum
(which yields 729 words) to a heuristic construction that is expected to
yield a smaller code (likely < 729), improving the score.
"""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

_DEADLINE = 100.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _precompute_balls(q, n, radius):
    """Precompute the ball of radius r for every word in the space."""
    all_words = list(itertools.product(range(q), repeat=n))
    all_words_np = np.array(all_words, dtype=np.int8)
    n_words = len(all_words_np)
    
    # balls[i] is a boolean array of length n_words indicating which words are within radius of all_words[i]
    balls = np.zeros((n_words, n_words), dtype=bool)
    for i in range(n_words):
        d = (all_words_np != all_words_np[i]).sum(axis=1)
        balls[i] = (d <= radius)
    return all_words, all_words_np, balls


def _greedy_set_cover(q, n, radius, all_words_np, balls, rng, time_limit):
    """Greedy set cover: pick words that cover the most uncovered words."""
    n_words = len(all_words_np)
    uncovered = np.ones(n_words, dtype=bool)
    codewords = []
    covered_count = np.zeros(n_words, dtype=np.int32)
    
    # We need to efficiently find the word that covers the most uncovered words.
    # Since balls is precomputed, we can compute the intersection of balls[i] and uncovered.
    # However, iterating over all n_words (6561) and checking intersection might be slow if done naively.
    # We can maintain a count of how many times each candidate word covers an uncovered word?
    # Actually, simpler: just iterate and pick the best.
    
    while uncovered.any() and _remaining() > time_limit:
        best_idx = -1
        best_count = -1
        
        # Optimization: Only consider words that are not already in codewords?
        # Actually, we can just pick any word. If it's already there, it adds 0 new coverage.
        # So we should prefer words not in codewords, or just rely on count=0 if covered.
        
        # To speed up, we can shuffle the order of candidates slightly or just scan.
        # Scanning 6561 words * 6561 words (for intersection) is ~43M ops.
        # In Python, this is too slow.
        
        # Better approach:
        # Maintain a list of "uncovered words".
        # For each candidate c, count how many words in "uncovered list" are in balls[c].
        
        # Let's try a different heuristic:
        # Randomly pick a word from the uncovered set, then pick a random word within its ball?
        # No, greedy set cover is standard.
        
        # Let's optimize the intersection check.
        # Instead of checking all 6561 candidates, we can just pick a random uncovered word u,
        # and then pick a random word c from balls[u]?
        # This is a "Random Greedy" variant. It is much faster.
        
        # Let's implement Random Greedy:
        # 1. Pick a random uncovered word u.
        # 2. Pick a random codeword candidate c from the ball of u (words within radius of u).
        # 3. Add c to codewords.
        # 4. Update uncovered.
        
        # This is not the *optimal* greedy (max coverage), but it is a valid greedy heuristic
        # and much faster. It often yields good results for covering codes.
        
        if not uncovered.any():
            break
            
        # Pick a random uncovered word index
        uncov_indices = np.nonzero(uncovered)[0]
        u_idx = uncov_indices[rng.randrange(len(uncov_indices))]
        
        # Pick a random word c from the ball of u_idx
        # balls[u_idx] is a boolean array. We need indices where it is True.
        c_candidates = np.nonzero(balls[u_idx])[0]
        c_idx = c_candidates[rng.randrange(len(c_candidates))]
        
        c = tuple(all_words_np[c_idx])
        codewords.append(c)
        
        # Update uncovered
        uncovered &= ~balls[c_idx]
        
    return codewords


def _greedy_remove(codewords, q, n, radius, all_words_np, balls, time_limit):
    """Iteratively remove redundant codewords."""
    codewords = list(codewords)
    n_words = len(all_words_np)
    
    # Map codeword tuple to index in balls
    # We need to know which ball corresponds to which codeword.
    # Since all_words_np contains all words, we can find the index.
    # But we have duplicates? No, codewords should be distinct.
    
    # To speed up, we can maintain a set of covered words for each codeword?
    # Or just recompute coverage counts.
    
    changed = True
    while changed and _remaining() > time_limit:
        changed = False
        if not codewords:
            break
            
        # Compute coverage count for each word in the space
        cov = np.zeros(n_words, dtype=np.int32)
        cw_indices = []
        
        # Map codewords to their indices in all_words_np
        # Since all_words_np is sorted (itertools.product), we can use a dict or just search.
        # Better: precompute a mapping from tuple to index.
        # We don't have that here, so we'll create it once if needed.
        # For now, let's just use the index in the codewords list.
        
        # Actually, we can just use the fact that codewords are tuples from all_words_np.
        # We can find the index by searching.
        
        # To make this fast, let's precompute a dictionary mapping tuple to index in all_words_np.
        # This is expensive to build every time, but we can do it once in construct.
        
        # For now, let's assume we have a way to get the index.
        # Let's pass a 'word_to_idx' dict.
        
        # Wait, I didn't pass word_to_idx. Let's modify the function signature or compute it.
        # Computing it for 6561 words is fast.
        
        # Let's refactor to pass word_to_idx.
        
        # For now, let's just use a simple loop to find indices.
        # 6561 words, 6561 candidates. 43M lookups. Too slow in Python.
        
        # Let's change strategy:
        # Instead of removing one by one, we can just run the greedy set cover again
        # with a constraint that we prefer existing codewords?
        
        # Simpler: Just return the codewords from greedy set cover.
        # Greedy set cover is already minimal in the sense that it stops when all are covered.
        # It doesn't remove redundant ones added later.
        
        # Let's implement a simpler removal:
        # Shuffle codewords.
        # For each c in shuffled list:
        #   Check if c is redundant.
        #   If yes, remove it.
        
        rng_local = random.Random()
        for _ in range(10): # Try a few times
            if _remaining() < time_limit:
                break
            rng_local.shuffle(codewords)
            
            # Rebuild coverage
            cov = np.zeros(n_words, dtype=np.int32)
            for c in codewords:
                # Find index of c in all_words_np
                # We need a fast way.
                # Let's assume we have a helper.
                pass
        
    return codewords


def construct(q, n, radius):
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = 100.0

    rng = random.Random(12345)

    # Precompute
    all_words, all_words_np, balls = _precompute_balls(q, n, radius)
    
    # Create a mapping from tuple to index for fast lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    best_codewords = []
    best_len = float('inf')
    
    # Run Random Greedy Set Cover multiple times with different seeds
    for seed in range(10):
        if _remaining() < 10.0:
            break
        
        rng_local = random.Random(seed)
        codewords = _greedy_set_cover(q, n, radius, all_words_np, balls, rng_local, 5.0)
        
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = list(codewords)
            
    # Try to improve with Greedy Removal
    # We need a fast removal function.
    # Let's implement a simple one using the precomputed balls and word_to_idx.
    
    def _greedy_remove_fast(codewords):
        codewords = list(codewords)
        n_words = len(all_words_np)
        
        # Map codewords to indices
        cw_indices = [word_to_idx[c] for c in codewords]
        
        changed = True
        while changed and _remaining() > 5.0:
            changed = False
            # Compute coverage
            cov = np.zeros(n_words, dtype=np.int32)
            for idx in cw_indices:
                cov += balls[idx].astype(np.int32)
            
            # Try to remove a redundant codeword
            # A codeword at index i is redundant if for all j in balls[i], cov[j] >= 2
            # We can check this by: min(cov[balls[i]]) >= 2
            
            for k, idx in enumerate(cw_indices):
                ball_mask = balls[idx]
                if cov[ball_mask].min() >= 2:
                    # Remove it
                    codewords.pop(k)
                    cw_indices.pop(k)
                    changed = True
                    break
        return codewords
    
    if best_codewords:
        improved = _greedy_remove_fast(best_codewords)
        if len(improved) < best_len:
            best_codewords = improved
            best_len = len(improved)
            
    # Final validation
    if best_codewords:
        covered = np.zeros(len(all_words_np), dtype=bool)
        for c in best_codewords:
            idx = word_to_idx[c]
            covered |= balls[idx]
        if covered.all():
            return [list(w) for w in best_codewords]
            
    # Fallback: Direct Sum (729 words)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    fallback = [tuple(h + list(tail))
                for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]
    return [list(w) for w in fallback]
# EVOLVE-BLOCK-END
