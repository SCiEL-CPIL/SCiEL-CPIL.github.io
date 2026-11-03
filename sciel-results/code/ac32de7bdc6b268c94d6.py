import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But the function signature is generic. We assume the evaluator passes these.
    # We will optimize for the specific case q=3, n=7, radius=1 but keep it general enough.
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: simple greedy
        # For now, focus on the specific task instance as per instructions
        pass

    total_words = q ** n
    # Precompute neighbors as bitmasks
    # Map word tuple to index
    # Index = sum(digit * q^i)
    
    # Precompute neighbors masks
    # neighbors_mask[i] is an integer where bit j is 1 if word j is within distance 1 of word i
    neighbors_mask = [0] * total_words
    
    # Helper to convert index to tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    # Helper to convert tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 0
        # Distance 0
        mask |= (1 << i)
        # Distance 1
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
        neighbors_mask[i] = mask

    # Greedy construction with bitmasks
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            # To add randomness, we can pick from top few or just randomize iteration order
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            # Optimization: Only consider words that are currently uncovered?
            # If a word is covered, it might still have high gain for others.
            # But usually, the best codeword is one that covers many uncovered words.
            # We can iterate all, but it's 2187 * 250 steps.
            
            # Faster: iterate only through set bits of uncovered?
            # No, a codeword must be a valid word. It doesn't have to be uncovered.
            # However, if w is covered by c, S(w) is not necessarily subset of S(c).
            # Let's just iterate all candidates but break early if we find gain 0? No.
            
            # We can optimize by only checking words that have at least one neighbor in uncovered.
            # This is equivalent to checking all, but we can skip if (neighbors_mask[i] & uncovered) == 0.
            
            gains = []
            for i in candidates:
                if (neighbors_mask[i] & uncovered) == 0:
                    continue
                gain = (neighbors_mask[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break # Should not happen if uncovered is not 0 and masks are correct
                
            code.append(best_idx)
            uncovered &= ~neighbors_mask[best_idx]
            
        return code

    def is_cover(code):
        # Check if code covers all words
        if not code:
            return total_words == 0
        covered = 0
        for c in code:
            covered |= neighbors_mask[c]
        return (covered == (1 << total_words) - 1)

    def reduce_code(code):
        # Remove redundant codewords
        # Maintain cover counts
        # Since total_words is small, we can use an array of counts
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            # Add 1 to all neighbors of c
            # This is slow if done naively. 
            # Better: use the bitmask to update?
            # Let's stick to a simpler check: 
            # Try removing each c. If valid, remove.
            # To check validity fast, we can maintain the union of masks?
            # No, removal requires knowing if any word becomes uncovered.
            
            # Efficient removal check:
            # Precompute for each codeword c, the set of words that are ONLY covered by c?
            # This is complex.
            
            # Alternative: 
            # Maintain an array `count[w]`.
            # When adding c, increment count[w] for w in neighbors[c].
            # When removing c, decrement count[w] for w in neighbors[c].
            # If min(count) > 0, removal is safe.
            
            pass
        
        # Let's implement the count-based reduction
        counts = np.zeros(total_words, dtype=np.int8)
        # Initialize counts
        for c in code:
            # Increment neighbors
            mask = neighbors_mask[c]
            # Iterate set bits? Or just loop through all neighbors?
            # Neighbors size is 15.
            # We need the list of neighbor indices, not just mask.
            pass
        
        # Let's precompute neighbor indices lists for faster count updates
        neighbor_indices = [[] for _ in range(total_words)]
        for i in range(total_words):
            mask = neighbors_mask[i]
            # Extract indices
            # Since n=7, q=3, max idx 2186.
            # We can just loop 0 to total_words-1 and check bit? 2187 * 2187 is 4.7M. Fine.
            # Or better, during precomputation, store the list.
            pass

    # Re-do precomputation to store neighbor indices
    neighbor_indices = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        neighbor_indices[i] = idxs

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Try removing c
                for w in neighbor_indices[c]:
                    counts[w] -= 1
                
                # Check if any count is 0
                if np.any(counts == 0):
                    # Restore
                    for w in neighbor_indices[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    # Keep removed
                    changed = True
            
            code = new_code
        return code

    # Main loop with time management
    deadline = time.time() + 110  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # Optional: Try further local search if time permits
            # For now, just collect results
            
        seed += 1
        
        # If we hit 186, we can stop? No, lower bound is 156. 
        # But 186 is best known. If we find 186, it's a great result.
        if best_size <= 186:
            break

    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
