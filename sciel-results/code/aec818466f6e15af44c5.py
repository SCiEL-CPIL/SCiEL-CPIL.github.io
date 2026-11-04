import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    total_words = q ** n
    
    # 1. Precompute all words and their balls
    # Represent words as integers 0..total_words-1
    # Ball of a word: set of indices within distance <= radius
    
    # For q=3, n=6, radius=1:
    # Ball size = 1 + n*(q-1) = 1 + 6*2 = 13
    
    # We use a list of sets for balls for easy set operations, 
    # but for speed in local search, we use bitmasks.
    
    # Map integer index to word tuple
    # This is a base-q representation
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[n-1-i] = idx % q
            idx //= q
        return w
    
    # Precompute balls as sets of indices
    balls = []
    for idx in range(total_words):
        word = idx_to_word(idx)
        ball = {idx}
        for pos in range(n):
            for val in range(1, q):
                new_word = list(word)
                new_word[pos] = (word[pos] + val) % q
                # Convert back to index
                # We can do this faster by precomputing "neighbor offsets"
                # But for 729 words, even simple conversion is fast.
                # Let's optimize index conversion.
                # Actually, let's just build the ball using the word values.
                pass
        
        # Faster ball generation using precomputed neighbors
        # Let's precompute neighbors for all words first.
        # neighbors[idx] = list of indices at distance 1
        # ball[idx] = {idx} U neighbors[idx]
        
    # Optimization: Precompute all neighbors
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        word = idx_to_word(idx)
        for pos in range(n):
            for val in range(1, q):
                new_word = list(word)
                new_word[pos] = (word[pos] + val) % q
                # Convert new_word to index
                new_idx = 0
                for k in range(n):
                    new_idx = new_idx * q + new_word[k]
                neighbors[idx].append(new_idx)
        
    balls = []
    for idx in range(total_words):
        balls.append({idx} | set(neighbors[idx]))
        
    # Convert balls to numpy uint64 masks for fast AND/OR/NOT
    num_chunks = (total_words + 63) // 64
    full_mask = np.zeros(num_chunks, dtype=np.uint64)
    for c in range(num_chunks):
        full_mask[c] = (1 << 64) - 1
    if total_words % 64 != 0:
        full_mask[-1] = (1 << (total_words % 64)) - 1
        
    ball_masks = []
    for ball in balls:
        mask = np.zeros(num_chunks, dtype=np.uint64)
        for idx in ball:
            chunk = idx // 64
            bit = idx % 64
            mask[chunk] |= (1 << bit)
        ball_masks.append(mask)
        
    # 2. Greedy Set Cover
    def greedy_cover():
        covered = np.zeros(num_chunks, dtype=np.uint64)
        selected = []
        remaining = list(range(total_words))
        
        while True:
            # Check if full coverage
            is_full = True
            for c in range(num_chunks):
                if covered[c] != full_mask[c]:
                    is_full = False
                    break
            if is_full:
                break
                
            # Find word with max gain
            best_gain = -1
            best_idx = -1
            for idx in remaining:
                # gain = popcount(ball[idx] & ~covered)
                gain = 0
                for c in range(num_chunks):
                    gain += np.bitwise_count(ball_masks[idx][c] & ~covered[c])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            remaining.remove(best_idx)
            for c in range(num_chunks):
                covered[c] |= ball_masks[best_idx][c]
                
        return selected, covered
        
    # 3. Local Search
    def is_covered(mask):
        for c in range(num_chunks):
            if mask[c] != full_mask[c]:
                return False
        return True
        
    def local_search(selected, covered):
        # Try to remove redundant codewords
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                # Try removing selected[i]
                new_covered = covered.copy()
                for c in range(num_chunks):
                    new_covered[c] ^= ball_masks[selected[i]][c]
                if is_covered(new_covered):
                    selected.pop(i)
                    covered = new_covered
                    changed = True
                    break # Restart loop as indices changed
            
        # Try swaps
        # We want to find a swap that allows a reduction, or just explores the space
        # To be efficient, we limit the number of swaps tried per iteration
        max_swaps = 500
        for _ in range(max_swaps):
            if time.time() > deadline - 10:
                break
                
            if len(selected) == 0:
                break
                
            # Pick a random codeword to remove
            i = random.randint(0, len(selected) - 1)
            removed_idx = selected[i]
            
            # Coverage without removed_idx
            temp_covered = covered.copy()
            for c in range(num_chunks):
                temp_covered[c] ^= ball_masks[removed_idx][c]
                
            # Find a word to add that restores coverage
            # Ideally, we want to add a word that covers the "holes" left by removed_idx
            # Holes = ~temp_covered
            holes = np.zeros(num_chunks, dtype=np.uint64)
            for c in range(num_chunks):
                holes[c] = ~temp_covered[c] & full_mask[c]
                
            # Find best candidate to add
            best_candidate = -1
            best_gain = -1
            for j in range(total_words):
                if j in selected:
                    continue
                # How many holes does j cover?
                gain = 0
                for c in range(num_chunks):
                    gain += np.bitwise_count(ball_masks[j][c] & holes[c])
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = j
                    
            if best_candidate == -1:
                continue
                
            # Check if adding best_candidate restores full coverage
            new_covered = temp_covered.copy()
            for c in range(num_chunks):
                new_covered[c] |= ball_masks[best_candidate][c]
                
            if is_covered(new_covered):
                # Swap is valid
                selected[i] = best_candidate
                covered = new_covered
                
                # Try to remove redundant words again
                changed = True
                while changed:
                    changed = False
                    for k in range(len(selected)):
                        temp_cov2 = covered.copy()
                        for c in range(num_chunks):
                            temp_cov2[c] ^= ball_masks[selected[k]][c]
                        if is_covered(temp_cov2):
                            selected.pop(k)
                            covered = temp_cov2
                            changed = True
                            break
                            
        return selected, covered

    best_solution = None
    best_size = float('inf')
    
    # Multiple restarts
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
            
        random.seed(42 + attempt)
        
        # Greedy init
        selected, covered = greedy_cover()
        
        # Local search
        selected, covered = local_search(selected, covered)
        
        if is_covered(covered):
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = [idx_to_word(idx) for idx in selected]
                
    # Fallback if no valid solution found (should not happen)
    if best_solution is None:
        # Return a valid but potentially non-optimal code
        # Simple construction: all words with first 3 coords fixed? No.
        # Just return the last greedy result
        selected, covered = greedy_cover()
        best_solution = [idx_to_word(idx) for idx in selected]
        
    return best_solution
