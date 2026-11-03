import itertools
import time
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks
    # mask[i] is a bitmask of indices of words within Hamming distance <= radius of word i
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Lazy Greedy Set Cover
    # Uses a priority queue to efficiently find the codeword with maximum marginal gain.
    # "Lazy" evaluation: items in the heap may have stale gains. We pop until we find
    # one whose stored gain matches the current marginal gain.
    
    def lazy_greedy_cover():
        uncovered = full_mask
        code = []
        
        # Heap stores (-gain, codeword_index)
        # Initialize with all codewords and their initial gains
        heap = []
        for i in range(total_space):
            gain = cover_masks[i].bit_count()
            heapq.heappush(heap, (-gain, i))
        
        # To avoid re-pushing every item, we just pop and check.
        # Since total_space is 2187, the heap size is manageable.
        # However, standard lazy greedy re-inserts the updated gain.
        
        while uncovered:
            # Pop until we find a valid candidate
            valid_gain = -1
            valid_idx = -1
            while heap:
                neg_gain, i = heapq.heappop(heap)
                gain = -neg_gain
                # Compute current marginal gain
                current_gain = (cover_masks[i] & uncovered).bit_count()
                if current_gain == gain:
                    valid_gain = gain
                    valid_idx = i
                    break
                else:
                    # Gain has changed, re-insert with updated gain
                    if current_gain > 0:
                        heapq.heappush(heap, (-current_gain, i))
            
            if valid_idx == -1:
                # No more gain possible, but uncovered is not empty?
                # This implies the cover is not complete or we ran out of candidates.
                # For a valid covering problem, this shouldn't happen if we have enough words.
                # But if it does, we break to avoid infinite loop.
                break
                
            code.append(valid_idx)
            uncovered &= ~cover_masks[valid_idx]
            
            # In a full lazy greedy, we would update gains for all i in the heap.
            # But we use the "pop and check" strategy which is simpler but might
            # result in more heap operations. Given N=2187, this is fast enough.
            # A more optimized version would update a dictionary of gains and only
            # push when gain increases, but gains only decrease here.
            
        return code

    # Local Search: Remove redundant codewords using cover_count
    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        
        # Build cover_count array: cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_masks[idx]
                
                # Check if all words covered by idx are covered by at least one other codeword
                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                
                if removable:
                    # Decrement cover_count and remove
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    improved = True
                    break
        
        return current_code

    # Swap-based local search: Try to replace one codeword with another to reduce total size?
    # Actually, reduction is the main goal. Swapping might help if a different codeword
    # allows further reductions.
    # Let's stick to reduction for now, but add a "swap" heuristic if time permits.
    
    def local_search_swap(code):
        """
        Try to swap codeword i with codeword j if it leads to a smaller set after reduction.
        This is expensive, so we only do it if we have time.
        """
        if not code:
            return code
        
        current_code = code[:]
        best_code = current_code[:]
        
        # Simple heuristic: for each codeword, try swapping with a random other codeword
        # and see if reduction improves.
        # This is O(N^2 * ReduceCost). ReduceCost is ~N * MaskSize.
        # With N~200, this is 40000 * 200 * 200 = 1.6e9 operations. Too slow.
        
        # Alternative: Random restarts of greedy + reduction is likely more effective.
        return current_code

    # Initial greedy construction using Lazy Greedy
    best_code = lazy_greedy_cover()
    
    # Apply local search
    best_code = local_search_reduce(best_code)
    
    # Randomized restarts
    def lazy_greedy_cover_randomized():
        uncovered = full_mask
        code = []
        
        # Initialize heap
        heap = []
        for i in range(total_space):
            gain = cover_masks[i].bit_count()
            heapq.heappush(heap, (-gain, i))
        
        while uncovered:
            valid_gain = -1
            valid_idx = -1
            candidates = []
            
            # To allow randomization, we can collect all candidates with max gain
            # But lazy heap pops one by one.
            # Simple approach: pop max gain, if tie, pick randomly from ties?
            # Heap doesn't easily support tie-breaking.
            # Alternative: Use standard greedy but with random tie-breaking.
            
            # Let's fallback to standard greedy with random tie-breaking for simplicity
            # in the randomized version, as lazy heap tie-breaking is tricky.
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Run a few restarts if time permits
    if time.time() < deadline - 20.0:
        for _ in range(15): # Try up to 15 restarts
            if time.time() > deadline - 5.0:
                break
            cand_code = lazy_greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]

# EVOLVE-BLOCK-END
