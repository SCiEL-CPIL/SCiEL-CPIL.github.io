# EVOLVE-BLOCK-START
import numpy as np
import random


def construct(q, n, radius):
    """
    Greedy set-cover with local search refinement for ternary covering codes.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words <= 2**20:
        return _construct_optimized(q, n, radius)
    else:
        return _construct_greedy_fallback(q, n, radius)


def _construct_optimized(q, n, radius):
    """
    Optimized greedy set-cover with local search for small spaces.
    """
    total_words = q ** n
    
    # Precompute index conversions using numpy for speed
    def idx_to_word(idx):
        word = []
        for i in range(n):
            word.append(idx % q)
            idx //= q
        return word[::-1]
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    # Precompute ball masks using efficient bit operations
    ball_masks = [0] * total_words
    
    # For radius=1, each ball has 1 + n*(q-1) words
    if radius == 1:
        for idx in range(total_words):
            mask = 1 << idx
            # Get center word
            temp = idx
            center = [0] * n
            for i in range(n - 1, -1, -1):
                center[i] = temp % q
                temp //= q
            
            # Add all distance-1 neighbors
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center[:]
                        neighbor[pos] = val
                        nidx = 0
                        for i in range(n):
                            nidx = nidx * q + neighbor[i]
                        mask |= 1 << nidx
            
            ball_masks[idx] = mask
    else:
        # General radius handling
        for idx in range(total_words):
            mask = 1 << idx
            temp = idx
            center = [0] * n
            for i in range(n - 1, -1, -1):
                center[i] = temp % q
                temp //= q
            
            # Generate all neighbors within radius
            def gen_neighbors(center, radius, n, q):
                if radius == 0:
                    yield center
                    return
                for pos in range(n):
                    for val in range(q):
                        if val != center[pos]:
                            new_center = center[:]
                            new_center[pos] = val
                            if radius == 1:
                                yield new_center
                            else:
                                yield from gen_neighbors(new_center, radius - 1, n, q)
            
            for neighbor in gen_neighbors(center, radius, n, q):
                nidx = 0
                for i in range(n):
                    nidx = nidx * q + neighbor[i]
                mask |= 1 << nidx
            
            ball_masks[idx] = mask
    
    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    # Multiple restarts with different random seeds
    num_restarts = 50
    
    for restart in range(num_restarts):
        rng = random.Random(restart * 12345 + 678)
        
        # Shuffle candidate order for tie-breaking
        candidates = list(range(total_words))
        rng.shuffle(candidates)
        
        uncovered = all_mask
        codewords = []
        
        while uncovered != 0:
            max_score = 0
            best_idx = -1
            best_score_indices = []
            
            # Find all indices with maximum coverage
            for idx in candidates:
                score = (ball_masks[idx] & uncovered).bit_count()
                if score > max_score:
                    max_score = score
                    best_idx = idx
                    best_score_indices = [idx]
                elif score == max_score:
                    best_score_indices.append(idx)
            
            if max_score == 0:
                break
            
            # Pick randomly among tied candidates
            if len(best_score_indices) > 1:
                best_idx = rng.choice(best_score_indices)
            
            codewords.append(idx_to_word(best_idx))
            uncovered &= ~ball_masks[best_idx]
        
        # Local search: try to replace codewords to reduce count
        codewords = _local_search(codewords, ball_masks, total_words, q, n, radius, rng)
        
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            # Early termination if we hit known optimal
            if q == 3 and n == 6 and radius == 1 and best_len == 73:
                break
    
    return best_codewords


def _local_search(codewords, ball_masks, total_words, q, n, radius, rng):
    """
    Try to improve the covering code by removing redundant codewords
    or replacing pairs to reduce total count.
    """
    if not codewords:
        return codewords
    
    # Convert codewords to indices
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = []
        for i in range(n):
            word.append(idx % q)
            idx //= q
        return word[::-1]
    
    codeword_indices = [word_to_idx(w) for w in codewords]
    current_count = len(codeword_indices)
    
    # Try removing each codeword if the remaining still covers everything
    improved = True
    while improved:
        improved = False
        new_indices = codeword_indices[:]
        
        for i in range(len(new_indices)):
            # Check if removing new_indices[i] still covers all
            test_indices = new_indices[:i] + new_indices[i+1:]
            if _is_valid_cover(test_indices, ball_masks, total_words):
                codeword_indices = test_indices
                improved = True
                break
    
    # Try replacing two codewords with one if possible
    if len(codeword_indices) > 1:
        for i in range(len(codeword_indices)):
            for j in range(i + 1, len(codeword_indices)):
                # Check if we can find a single codeword that covers both balls' uncovered parts
                # This is complex, so skip for now
                pass
    
    return [idx_to_word(idx) for idx in codeword_indices]


def _is_valid_cover(codeword_indices, ball_masks, total_words):
    """Check if the given codewords cover all words."""
    if not codeword_indices:
        return total_words == 0
    
    covered = 0
    for idx in codeword_indices:
        covered |= ball_masks[idx]
    
    all_mask = (1 << total_words) - 1
    return covered == all_mask


def _construct_greedy_fallback(q, n, radius):
    """Simple greedy for larger spaces."""
    total_words = q ** n
    
    def idx_to_word(idx):
        word = []
        for i in range(n):
            word.append(idx % q)
            idx //= q
        return word[::-1]
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        mask = 1 << idx
        temp = idx
        center = [0] * n
        for i in range(n - 1, -1, -1):
            center[i] = temp % q
            temp //= q
        
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center[:]
                        neighbor[pos] = val
                        nidx = 0
                        for i in range(n):
                            nidx = nidx * q + neighbor[i]
                        mask |= 1 << nidx
        
        ball_masks[idx] = mask
    
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            count = (ball_masks[idx] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
