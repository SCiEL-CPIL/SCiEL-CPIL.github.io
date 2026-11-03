# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Multi-start greedy set-cover algorithm to find a small covering code.
    Uses NumPy vectorization for fast scoring and multiple random tie-breaking orders
    to escape local optima of the standard greedy heuristic.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words <= 2**20:
        return _construct_greedy_numpy(q, n, radius)
    else:
        # Fallback for large spaces (not expected for this task, but safe)
        return _construct_greedy_simple(q, n, radius)


def _construct_greedy_numpy(q, n, radius):
    """
    Optimized greedy set-cover with multiple restarts using NumPy for scoring.
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
    
    # Precompute adjacency matrix B where B[i, j] = 1 if dist(i, j) <= radius
    # Using uint8 for memory efficiency and fast dot products
    B = np.zeros((total_words, total_words), dtype=np.uint8)
    
    for i in range(total_words):
        center = idx_to_word(i)
        B[i, i] = 1
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        j = word_to_idx(neighbor)
                        B[i, j] = 1
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == center[pos1]: continue
                        for val2 in range(q):
                            if val2 == center[pos2]: continue
                            neighbor = center.copy()
                            neighbor[pos1] = val1
                            neighbor[pos2] = val2
                            j = word_to_idx(neighbor)
                            B[i, j] = 1
                            
    best_codewords = None
    best_len = float('inf')
    
    # Increase restarts significantly since NumPy makes each restart very fast
    num_restarts = 500
    
    for seed in range(num_restarts):
        rng = np.random.RandomState(seed)
        uncovered = np.ones(total_words, dtype=np.uint8)
        codewords = []
        
        while np.any(uncovered):
            # Calculate score for each candidate: number of uncovered words it covers
            # B is (N, N), uncovered is (N,). Result is (N,)
            scores = B @ uncovered
            
            # Add small random noise to break ties deterministically based on seed
            # This helps escape local optima compared to fixed order
            noise = rng.random(total_words) * 0.1
            scores_noisy = scores + noise
            
            best_idx = np.argmax(scores_noisy)
            
            if scores[best_idx] == 0:
                break # Should not happen if valid cover exists
            
            codewords.append(best_idx)
            uncovered[B[best_idx, :]] = 0
            
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = [idx_to_word(idx) for idx in codewords]
            
            # Early termination if we hit a very good bound
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
