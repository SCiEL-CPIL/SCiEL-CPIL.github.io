"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Multi-start greedy set-cover algorithm to find a small covering code.
    Uses bitmask representation and multiple random tie-breaking orders
    to escape local optima of the standard greedy heuristic.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words <= 2**20:
        return _construct_greedy_multi_start(q, n, radius)
    else:
        # Fallback for large spaces (not expected for this task, but safe)
        return _construct_greedy_simple(q, n, radius)


def _construct_greedy_multi_start(q, n, radius):
    """
    Optimized greedy set-cover with multiple restarts for small spaces.
    """
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute ball masks as Python integers (bitmasks)
    ball_masks = [0] * total_words
    
    # Generate neighbors for each center
    # For efficiency, we iterate through all words and compute distance
    # But for radius=1, direct generation is faster.
    # We'll use a generic neighbor generator that is efficient for small n.
    
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        
        if radius >= 2:
            # Add distance 2 neighbors
            # This is O(n^2 * q^2) per center, which is fine for n=6
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == center[pos1]: continue
                        for val2 in range(q):
                            if val2 == center[pos2]: continue
                            # Also consider same position changes? No, distance 2 means 2 positions changed.
                            # We already handled distance 1.
                            # Distance 2: change pos1 and pos2.
                            neighbor = center.copy()
                            neighbor[pos1] = val1
                            neighbor[pos2] = val2
                            nidx = word_to_idx(neighbor)
                            mask |= 1 << nidx
                            
                            # Also cases where one position is changed to a 3rd value?
                            # No, distance is number of differing positions.
                            # If pos1 and pos2 are different from center, distance is 2.
                            # We covered all pairs of positions and all values != center.
                            
        # Note: For radius > 2, this becomes very slow. 
        # Given the task is q=3, n=6, radius=1, the radius>=2 block is likely not executed 
        # or can be optimized if needed. For now, we assume radius is small.
        # If radius is larger, we might need a better generator.
        # But for the specific task (radius=1), the radius>=1 block is sufficient.
        
        ball_masks[idx] = mask
    
    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    # Number of restarts. 50 is safe within time limits.
    num_restarts = 50
    
    for seed in range(num_restarts):
        # Generate a random permutation of indices for tie-breaking
        perm = np.random.RandomState(seed).permutation(total_words)
        
        uncovered = all_mask
        codewords = []
        
        while uncovered != 0:
            # Find max coverage
            max_score = 0
            # We can optimize by skipping if uncovered is small, but 729 is small.
            for idx in range(total_words):
                # If the ball is fully covered, score is 0. 
                # We can check if ball_masks[idx] is subset of covered (i.e., not in uncovered)
                # But bit_count is fast.
                score = (ball_masks[idx] & uncovered).bit_count()
                if score > max_score:
                    max_score = score
            
            if max_score == 0:
                break # Should not happen if code is valid cover, but safety break
                
            # Find the earliest index in 'perm' that achieves max_score
            best_idx = -1
            for idx in perm:
                if (ball_masks[idx] & uncovered).bit_count() == max_score:
                    best_idx = idx
                    break
            
            if best_idx == -1:
                # Fallback to first one found
                for idx in range(total_words):
                    if (ball_masks[idx] & uncovered).bit_count() == max_score:
                        best_idx = idx
                        break
            
            codewords.append(idx_to_word(best_idx))
            uncovered &= ~ball_masks[best_idx]
            
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            # Early termination if we hit a very good bound (e.g., known optimal)
            # For K_3(6,1), optimal is 73.
            if best_len == 73:
                break
                
    return best_codewords

def _construct_greedy_simple(q, n, radius):
    """Fallback simple greedy for larger spaces."""
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        ball_masks[idx] = mask
        
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
