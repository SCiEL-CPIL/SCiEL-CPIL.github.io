import random
import time

def construct(q, n, radius):
    # Start time to track budget
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    total_words = q ** n
    
    # If the space is too large for the bitset heuristic, fall back to simple greedy or direct sum
    # For this specific task (q=3, n=6), total_words=729, which is small.
    if total_words > 10000:
        # Fallback: Direct sum construction from seed
        # This is only for safety if parameters change unexpectedly
        # But the task is fixed.
        # Let's just use a simple greedy without bitsets for larger spaces if needed, 
        # but given the constraints, we expect small n.
        pass 

    # Precompute neighbors for all words
    # Represent each word as an integer index 0 to total_words-1
    # Convert index to tuple of digits
    def index_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(t)
    
    def tuple_to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbor bitsets
    # neighbor_bits[i] is an integer where bit j is 1 if word j is within distance radius of word i
    neighbor_bits = [0] * total_words
    
    # Generate all words
    words = [index_to_tuple(i) for i in range(total_words)]
    
    # For each word, find its neighbors
    for i in range(total_words):
        w_i = words[i]
        mask = 0
        # Iterate over all words to find distance? 
        # For n=6, q=3, 729*729 = 500k comparisons. 
        # Each comparison is n=6 ops. 3M ops. Fast.
        for j in range(total_words):
            w_j = words[j]
            dist = 0
            for k in range(n):
                if w_i[k] != w_j[k]:
                    dist += 1
            if dist <= radius:
                mask |= (1 << j)
        neighbor_bits[i] = mask
        
    # Greedy algorithm with restarts
    best_code = None
    best_size = total_words + 1
    
    num_restarts = 100
    # Adjust restarts based on time
    # Each restart takes approx (total_words * best_size) bit_count ops.
    # For 729 words, ~73 steps. 729*73 = 53k ops.
    # 100 restarts = 5.3M ops. Very fast.
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        uncovered = (1 << total_words) - 1
        used = [False] * total_words
        code = []
        
        # Shuffle candidates to break ties randomly
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        while uncovered:
            best_gain = -1
            best_c = -1
            
            for c in candidates:
                if used[c]:
                    continue
                # Calculate gain
                g = (uncovered & neighbor_bits[c]).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_c = c
                    # If we found a perfect cover for the remaining (unlikely in intermediate steps), 
                    # we can't get better than this specific gain, but we still need to finish.
                    # However, if best_gain is 0, we are stuck (should not happen if radius is valid).
                    if best_gain == 0:
                        break
            
            if best_c == -1:
                # No candidate found? Should not happen.
                break
                
            code.append(best_c)
            used[best_c] = True
            uncovered &= ~neighbor_bits[best_c]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # Early termination if we hit the theoretical lower bound or a known good value?
            # Lower bound for K_3(6,1) is 71. We know 73 is possible.
            if best_size <= 73:
                # We found a very good solution. We can stop or keep searching for 71/72.
                # 71 is lower bound, 73 is upper bound.
                # Let's keep searching a bit more if time permits.
                pass

    if best_code is None:
        # Fallback if something went wrong
        return [words[i] for i in range(total_words)]
        
    # Convert indices back to lists of digits
    result = [list(words[i]) for i in best_code]
    
    return result
