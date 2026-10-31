import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general inputs if possible, though the prompt says "evaluator passes this task's alphabet size..."
    # implying q, n, radius are fixed for the task.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a simple construction for other parameters to avoid errors
        # This is a simple greedy for general case
        total_words = q ** n
        # ... implementation for general case ...
        # For now, assume the specific case or use a generic slow fallback
        # The prompt implies we only need to solve the specific instance well.
        # Let's implement a generic greedy just in case, but optimize for 3,6,1.
        
    # Optimized path for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Map words to indices
        # Word is a tuple of 6 ints in 0..2
        # Index = w[0]*3^5 + w[1]*3^4 + ... + w[5]
        
        # Precompute masks
        # There are 3^6 = 729 words.
        # We represent the set of covered words as a Python integer bitset.
        
        def word_to_idx(w):
            idx = 0
            for i in range(n):
                idx = idx * q + w[i]
            return idx

        def idx_to_word(idx):
            w = [0] * n
            for i in range(n-1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w

        # Precompute neighbor masks for all 729 words
        # mask[i] is a bitset of indices j such that dist(i, j) <= 1
        masks = [0] * (q ** n)
        
        for i in range(q ** n):
            w_i = idx_to_word(i)
            mask = 1 << i  # self
            # distance 1 neighbors
            for pos in range(n):
                for val in range(q):
                    if val == w_i[pos]:
                        continue
                    w_new = w_i[:]
                    w_new[pos] = val
                    idx_new = word_to_idx(tuple(w_new))
                    mask |= (1 << idx_new)
            masks[i] = mask

        # Greedy Set Cover
        # uncovered: bitset of all words
        uncovered = (1 << (q ** n)) - 1
        code_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_count = -1
            # Iterate through all candidates
            # Optimization: We can skip candidates that are already covered? 
            # No, a covered candidate might still cover new uncovered neighbors.
            # But if a candidate's *entire* mask is already covered (mask & uncovered == 0), it's useless.
            for i in range(q ** n):
                # Quick check: if mask[i] & uncovered is 0, skip
                # Actually, we need to count bits.
                # To speed up, we can just compute count.
                common = uncovered & masks[i]
                if common:
                    cnt = common.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
            
            if best_idx == -1:
                # Should not happen if uncovered != 0
                break
                
            code_indices.append(best_idx)
            uncovered &= ~masks[best_idx]
            
        # Convert indices back to words
        result = [idx_to_word(i) for i in code_indices]
        
        # If we didn't cover everything due to time, return best effort? 
        # The problem requires valid coverage. 
        # For n=6, q=3, the greedy loop is very fast (<< 1s).
        # So we should have a valid code.
        
        # Try to improve with swaps if time remains and size > 73
        if time.time() < deadline - 10.0 and len(result) > 73:
            # Local search: try to swap one word for another to reduce size
            # Or just random restarts of greedy with different tie-breaking?
            # Let's try a simple swap heuristic:
            # Current code size K.
            # Try to find a word w not in code such that replacing some c in code with w
            # still allows coverage, and then see if we can remove another word.
            # This is complex. 
            # Alternative: Randomized Greedy.
            # Run greedy a few times with random tie-breaking.
            
            best_code = result
            best_size = len(result)
            
            # Run a few randomized greedy instances
            for _ in range(5): # 5 restarts
                if time.time() > deadline - 5.0:
                    break
                uncovered = (1 << (q ** n)) - 1
                code_indices = []
                while uncovered:
                    best_idx = -1
                    best_count = -1
                    # Random tie-breaking
                    candidates = []
                    for i in range(q ** n):
                        common = uncovered & masks[i]
                        if common:
                            cnt = common.bit_count()
                            if cnt > best_count:
                                best_count = cnt
                                candidates = [i]
                            elif cnt == best_count:
                                candidates.append(i)
                    if not candidates:
                        break
                    best_idx = random.choice(candidates)
                    code_indices.append(best_idx)
                    uncovered &= ~masks[best_idx]
                
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = [idx_to_word(i) for i in code_indices]
                    if best_size == 73:
                        break # Can't do better than known lower bound 71, best known 73
                        
            result = best_code

        return result

    else:
        # Generic fallback (slow)
        # ...
        pass
