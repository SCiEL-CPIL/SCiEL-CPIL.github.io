"""Structural redesign: greedy set-cover followed by swap-based local search with simulated annealing.

The seed uses a direct-sum construction (9 * 3**(n-4) words). For n=6 that is 81 words.
We instead:
1. Build an initial covering via a greedy set-cover heuristic (fast, usually near-optimal).
2. Improve it with swap-based local search: try removing one codeword and adding another
   (or just removing if still covered) to reduce the count. Accept improvements greedily;
   also accept equal-size swaps with a simulated-annealing temperature to escape local minima.
3. Use a deadline guard to return the best valid code found before time runs out.
"""
import time
import random
import numpy as np

# Deadline: leave a safety margin (10 s) before the 150 s kill.
DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _hamming_dist(a, b):
    return int(np.sum(a != b))


def _all_words(q, n):
    """Generate all q^n words as a numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.uint8)
    for i in range(total):
        v = i
        for j in range(n):
            words[i, j] = v % q
            v //= q
    return words


def _cover_mask(codeword, all_words, q, n, radius):
    """Return a boolean mask of all_words within Hamming distance <= radius of codeword."""
    # Compute Hamming distances from codeword to all words
    diffs = (all_words != codeword).sum(axis=1)
    return diffs <= radius


def _greedy_cover(q, n, radius, start_time=None):
    """Greedy set-cover: repeatedly pick the word covering the most uncovered words."""
    if start_time is None:
        start_time = time.time()
    
    all_words = _all_words(q, n)
    total = len(all_words)
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    # Precompute nothing heavy; just iterate
    while uncovered.any():
        best_idx = -1
        best_count = 0
        # Sample candidate positions to speed up (for large spaces)
        # For q=3, n=6, total=729, we can afford full scan each iteration
        for i in range(total):
            if not uncovered[i]:
                continue
            # Count how many currently-uncovered words this candidate covers
            dists = (all_words != all_words[i]).sum(axis=1)
            mask = dists <= radius
            count = np.sum(uncovered & mask)
            if count > best_count:
                best_count = count
                best_idx = i
            if best_count == np.sum(uncovered):
                break  # perfect cover from here
        
        if best_idx == -1:
            break
        
        codewords.append(all_words[best_idx].tolist())
        dists = (all_words != all_words[best_idx]).sum(axis=1)
        uncovered &= ~(dists <= radius)
        
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
    
    return codewords


def _is_covered(codewords, q, n, radius):
    """Check if codewords form a valid covering of the entire space."""
    if not codewords:
        return False
    all_words = _all_words(q, n)
    cd = np.array(codewords, dtype=np.uint8)
    # For each word, check if any codeword is within radius
    # Vectorized: compute distances from each word to each codeword
    # Shape: (total, len(codewords))
    diffs = np.abs(all_words[:, np.newaxis, :] - cd[np.newaxis, :, :]).sum(axis=2)
    return np.all(diffs.min(axis=1) <= radius)


def _swap_local_search(codewords, q, n, radius, start_time=None):
    """Swap-based local search with simulated annealing for equal-size moves."""
    if start_time is None:
        start_time = time.time()
    
    all_words = _all_words(q, n)
    total = len(all_words)
    cd = np.array(codewords, dtype=np.uint8)
    current_size = len(codewords)
    best_codewords = [c.tolist() for c in codewords]
    best_size = current_size
    
    # Precompute cover masks for each codeword
    cover_masks = []
    for c in codewords:
        dists = (all_words != c).sum(axis=1)
        cover_masks.append(dists <= radius)
    
    # Compute current coverage
    covered = np.any(np.stack(cover_masks), axis=0)
    assert covered.all(), "Initial code must be a valid cover"
    
    temperature = 0.1  # SA temperature for equal-size swaps
    cooling = 0.99
    
    while time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        improved = False
        
        # Try removals: for each codeword, check if removing it still leaves full coverage
        for i in range(len(cover_masks)):
            if len(cover_masks) <= 1:
                break
            # If this codeword's unique coverage is empty, we can remove it
            others = np.any([cover_masks[j] for j in range(len(cover_masks)) if j != i], axis=0)
            if others.all():
                # Safe to remove
                cover_masks.pop(i)
                codewords.pop(i)
                current_size -= 1
                improved = True
                # Recompute covered
                if cover_masks:
                    covered = np.any(np.stack(cover_masks), axis=0)
                else:
                    covered = np.zeros(total, dtype=bool)
                break
        
        if improved:
            if current_size < best_size:
                best_size = current_size
                best_codewords = [c.tolist() for c in codewords]
            continue
        
        # Try swaps: remove codeword i, add a random word w
        if len(cover_masks) > 0:
            # Pick a random codeword to potentially remove
            i = random.randint(0, len(cover_masks) - 1)
            # Pick a random candidate word to add
            w_idx = random.randint(0, total - 1)
            w = all_words[w_idx]
            
            # Check: after removing i and adding w, is coverage maintained?
            # Coverage without i:
            if len(cover_masks) > 1:
                cov_without_i = np.any([cover_masks[j] for j in range(len(cover_masks)) if j != i], axis=0)
            else:
                cov_without_i = np.zeros(total, dtype=bool)
            
            # New coverage with w added
            dists_w = (all_words != w).sum(axis=1)
            new_cover = cov_without_i | (dists_w <= radius)
            
            if new_cover.all():
                # Valid swap: replace i with w
                cover_masks[i] = dists_w <= radius
                codewords[i] = w
                improved = True
                # Recompute covered
                covered = np.any(np.stack(cover_masks), axis=0)
            
            if not improved:
                # Try a few more random candidates
                for _ in range(10):
                    w_idx = random.randint(0, total - 1)
                    w = all_words[w_idx]
                    dists_w = (all_words != w).sum(axis=1)
                    new_cover = cov_without_i | (dists_w <= radius)
                    if new_cover.all():
                        cover_masks[i] = dists_w <= radius
                        codewords[i] = w
                        improved = True
                        break
        
        if improved:
            if current_size < best_size:
                best_size = current_size
                best_codewords = [c.tolist() for c in codewords]
        
        # Cool down temperature
        temperature *= cooling
    
    # Return the best valid code found
    return best_codewords


def construct(q, n, radius):
    start_time = time.time()
    
    # Step 1: Greedy cover
    codewords = _greedy_cover(q, n, radius, start_time)
    
    # Verify it's a valid cover
    if not _is_covered(codewords, q, n, radius):
        # Fallback: use the direct-sum construction if greedy failed
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For small n, brute force
            codewords = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Step 2: Swap-based local search
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        codewords = _swap_local_search(codewords, q, n, radius, start_time)
    
    # Final verification
    if not _is_covered(codewords, q, n, radius):
        # Fallback to direct-sum
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            codewords = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    return codewords
