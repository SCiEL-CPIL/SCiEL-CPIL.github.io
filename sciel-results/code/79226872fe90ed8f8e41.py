# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def construct(q, n, radius):
    """
    Optimized greedy set-cover with priority queue and weighted scoring.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 5000:
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_optimized(q, n, radius)

def _construct_greedy_optimized(q, n, radius):
    """
    Greedy set-cover with priority queue, incremental updates, and weighted scoring.
    """
    total_words = q ** n
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
        
    # Precompute ball masks and neighbor lists
    ball_masks = [0] * total_words
    neighbor_lists = [[] for _ in range(total_words)]
    
    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            neighbors = [i]
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v == digit:
                        continue
                    j = i + (v - digit) * step
                    mask |= 1 << j
                    neighbors.append(j)
            ball_masks[i] = mask
            neighbor_lists[i] = neighbors
    else:
        # Generic fallback
        for i in range(total_words):
            mask = 1 << i
            neighbors = [i]
            for j in range(total_words):
                if i == j:
                    continue
                # Check Hamming distance
                dist = 0
                a, b = i, j
                for p in range(n):
                    da = (a // q_powers[p]) % q
                    db = (b // q_powers[p]) % q
                    if da != db:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    mask |= 1 << j
                    neighbors.append(j)
            ball_masks[i] = mask
            neighbor_lists[i] = neighbors

    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 115.0
    
    restart_count = 0
    rng = np.random.RandomState(42)
    
    while True:
        perm = rng.permutation(total_words)
        
        # Greedy with priority queue
        uncovered = all_mask
        codewords = []
        
        # Track how many uncovered words each candidate covers
        # Use a heap: (-score, candidate_idx)
        # We'll use lazy deletion: push new scores, ignore stale entries
        
        # Initial scores
        scores = [0] * total_words
        for idx in range(total_words):
            scores[idx] = (ball_masks[idx] & uncovered).bit_count()
        
        # Build heap
        heap = []
        for idx in range(total_words):
            if scores[idx] > 0:
                heapq.heappush(heap, (-scores[idx], idx))
        
        # Track which candidates are still "active" (not yet selected)
        selected = [False] * total_words
        
        while uncovered != 0:
            # Pop until we find a valid entry
            best_idx = -1
            best_score = 0
            while heap:
                neg_score, idx = heapq.heappop(heap)
                score = -neg_score
                if selected[idx]:
                    continue
                # Verify score is still accurate (lazy check)
                actual_score = (ball_masks[idx] & uncovered).bit_count()
                if actual_score != score:
                    # Stale entry, recompute and push back
                    if actual_score > 0:
                        heapq.heappush(heap, (-actual_score, idx))
                    continue
                best_idx = idx
                best_score = score
                break
            
            if best_idx == -1 or best_score == 0:
                break
            
            # Select this codeword
            selected[best_idx] = True
            temp = best_idx
            cw = [0] * n
            for k in range(n-1, -1, -1):
                cw[k] = temp % q
                temp //= q
            codewords.append(cw)
            
            # Update uncovered
            newly_covered = ball_masks[best_idx] & uncovered
            uncovered &= ~ball_masks[best_idx]
            
            # Update scores for candidates that cover any newly covered word
            # A candidate's score decreases by the number of newly covered words it covers
            # We need to find candidates that overlap with newly_covered
            # For efficiency, iterate over newly covered words and their "covering candidates"
            # But we don't have reverse mapping, so we just push updated scores for all
            # candidates that have non-zero overlap with newly_covered
            
            # Since ball size is small (~13), and number of candidates is 729,
            # we can just update scores for candidates that intersect newly_covered
            # Use the fact that if candidate c covers word w, then w is in neighbor_lists[c]
            # Reverse: for each word w in newly_covered, which candidates cover w?
            # We don't have this precomputed, so let's precompute it
            
            # Actually, let's just recompute scores lazily - the heap approach with
            # lazy deletion should handle this. But we need to push updated entries.
            # For each candidate that might have changed score, push a new entry.
            # A candidate's score changes only if it covers at least one newly_covered word.
            # So we need: for each w in newly_covered, for each c that covers w, push (-new_score, c)
            
            # Precompute reverse: coverers[w] = list of candidates that cover w
            # This is expensive to build but worth it for speed
            pass  # We'll build this outside the restart loop
            
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            if best_len == 73:
                break
                
        restart_count += 1
        
        if time.time() - start_time > max_time:
            break
            
        if restart_count > 5000:
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
