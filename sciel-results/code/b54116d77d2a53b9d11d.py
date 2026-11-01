import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Only handle the specific case q=3, n=6, radius=1 efficiently
    # For other cases, fallback to seed-like or simple greedy
    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        # Map word to index for fast lookup
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        idx_to_word = {i: w for i, w in enumerate(all_words)}
        
        # Precompute neighbors
        neighbors = []
        for w in all_words:
            nb = [w]
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        nw = list(w)
                        nw[i] = v
                        nb.append(tuple(nw))
            neighbors.append(nb)
            
        uncovered = set(range(len(all_words)))
        code = []
        
        while uncovered:
            best_idx = -1
            best_cover = 0
            for i in range(len(all_words)):
                cover = 0
                for nb in neighbors[i]:
                    if word_to_idx[nb] in uncovered:
                        cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_idx = i
            code.append(list(idx_to_word[best_idx]))
            for nb in neighbors[best_idx]:
                uncovered.discard(word_to_idx[nb])
            if time.time() > deadline:
                break
        return code

    # Specific optimized path for q=3, n=6, radius=1
    # Total words: 729
    # Represent words as integers 0..728 (base 3)
    
    def decode(x):
        return [int(x // (3**i) % 3) for i in range(6)]
        
    def encode(w):
        x = 0
        for i in range(6):
            x += w[i] * (3**i)
        return x

    all_words = list(range(729))
    N = 729
    
    # Precompute coverage bitsets
    # Cover[i] is an integer where bit j is 1 if word j is covered by codeword i
    covers = [0] * N
    for i in all_words:
        w = decode(i)
        mask = 0
        # Self
        mask |= (1 << i)
        # Neighbors
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                j = encode(nw)
                mask |= (1 << j)
        covers[i] = mask
        
    all_mask = (1 << N) - 1
    
    # Greedy Initialization
    def greedy_init():
        uncovered = all_mask
        code = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(N):
                gain = bin(covers[i] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            code.append(best_idx)
            uncovered &= ~covers[best_idx]
        return code

    def reduce_code(code):
        # Remove redundant codewords
        # Union of all covers
        union_mask = 0
        for c in code:
            union_mask |= covers[c]
            
        new_code = []
        for c in code:
            # Check if c is redundant
            # Union of others
            union_others = 0
            for c2 in code:
                if c2 != c:
                    union_others |= covers[c2]
            
            if (covers[c] & ~union_others) == 0:
                # Redundant
                pass
            else:
                new_code.append(c)
                
        return new_code

    # Initial code
    code = greedy_init()
    code = reduce_code(code)
    
    # Local Search
    # We will try swaps
    # To speed up, we can track the union mask
    current_union = 0
    for c in code:
        current_union |= covers[c]
        
    best_code = list(code)
    best_size = len(code)
    
    # Set of codes in current solution for fast lookup
    code_set = set(code)
    
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() > deadline - 10:
            break
            
        improved = False
        # Try all swaps
        # Optimization: Iterate over c_out in code, c_in in not code
        # If valid, try reduction
        
        for i_out, c_out in enumerate(list(code)):
            if time.time() > deadline - 5:
                break
                
            # Calculate union without c_out
            # This is expensive if done naively.
            # But we can precompute prefix/suffix unions?
            # Or just recompute union_others for each c_out?
            # |code| is ~80. 80 * 80 = 6400 ops.
            
            union_without_out = 0
            for c in code:
                if c != c_out:
                    union_without_out |= covers[c]
            
            # For each c_in not in code
            for c_in in range(N):
                if c_in in code_set:
                    continue
                    
                # New union
                new_union = union_without_out | covers[c_in]
                
                if new_union == all_mask:
                    # Valid swap
                    # Create new code
                    new_code = list(code)
                    new_code.remove(c_out)
                    new_code.append(c_in)
                    
                    # Reduce
                    new_code = reduce_code(new_code)
                    
                    if len(new_code) < best_size:
                        code = new_code
                        code_set = set(code)
                        best_size = len(code)
                        best_code = list(code)
                        improved = True
                        # Update union for next iterations?
                        # We break out of c_in loop to restart with new code?
                        # Or just continue?
                        # If we change code, the indices in 'code' list change.
                        # Better to break and restart the outer loop?
                        # Or just update 'code' and continue?
                        # Since 'code' is modified, the 'for i_out, c_out in enumerate(list(code))'
                        # is iterating over a copy, so it's safe.
                        # But 'union_without_out' is now stale for subsequent c_in in this c_out?
                        # No, c_out is removed.
                        # But other c_outs are also affected?
                        # It's safer to restart the search loop if we find an improvement.
                        break
            
            if improved:
                break
        
        if not improved:
            # Try random perturbations if stuck
            # Pick a random c_out and a random c_in that is valid
            found_swap = False
            for _ in range(100): # Try 100 random swaps
                if time.time() > deadline - 5:
                    break
                c_out = random.choice(code)
                c_in = random.randint(0, N-1)
                if c_in in code_set:
                    continue
                
                union_without_out = 0
                for c in code:
                    if c != c_out:
                        union_without_out |= covers[c]
                
                new_union = union_without_out | covers[c_in]
                if new_union == all_mask:
                    new_code = list(code)
                    new_code.remove(c_out)
                    new_code.append(c_in)
                    new_code = reduce_code(new_code)
                    
                    if len(new_code) <= best_size:
                        code = new_code
                        code_set = set(code)
                        best_size = len(code)
                        best_code = list(code)
                        found_swap = True
                        break
            
            if not found_swap:
                # No improvement, maybe break to save time?
                # Or continue?
                pass
                
    # Convert best_code back to lists
    result = [decode(c) for c in best_code]
    
    return result
