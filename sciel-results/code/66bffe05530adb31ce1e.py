import time
import random
import numpy as np

def construct(q, n, radius):
    # q=3, n=6, radius=1
    # Total words = 3**6 = 729
    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighbors for each word (Hamming distance <= 1)
    # A word is a tuple of n ints
    words = []
    for i in range(total_words):
        # Convert index to base-q tuple
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        words.append(tuple(w))
        
    # Map index to word and word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors.append(neigh)
    
    # Initial Code: Tetracode direct sum
    # Tetracode: 9 words of length 4
    tetracode = []
    for a in range(q):
        for b in range(q):
            c = (a + b) % q
            d = (a + 2 * b) % q
            tetracode.append((a, b, c, d))
            
    # Expand to length 6 by adding 2 more coordinates
    # The seed program used: h + tail where tail is product of range(3) repeat n-4
    # This creates 9 * 3**2 = 81 words.
    initial_code_indices = []
    for h in tetracode:
        for x in range(q):
            for y in range(q):
                full_word = h + (x, y)
                idx = word_to_idx[full_word]
                initial_code_indices.append(idx)
                
    C = initial_code_indices
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Calculate initial cover counts
    for c in C:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    # Helper to check if valid
    def is_valid():
        return np.min(cover_count) >= 1

    # SA Parameters
    start_time = time.time()
    deadline = 110.0 # 10s margin for 120s limit
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    best_C = C.copy()
    best_cover_count = cover_count.copy()
    best_size = len(C)
    
    # We want to minimize size. 
    # If we remove a word and it stays valid, we accept.
    # If we remove a word and it becomes invalid, we might accept based on penalty.
    # Penalty = number of uncovered words.
    
    while time.time() < deadline:
        # Pick a random move
        # 80% Remove, 20% Swap
        if random.random() < 0.8:
            if not C:
                break
            c_old = random.choice(C)
            # Try removing c_old
            # Calculate change in cover_count
            delta_cover = np.zeros(total_words, dtype=np.int32)
            for nb in neighbors[c_old]:
                delta_cover[nb] -= 1
            
            new_cover_count = cover_count + delta_cover
            # Count newly uncovered words
            # Words that were 1 and become 0
            newly_uncovered = np.sum((cover_count == 1) & (new_cover_count == 0))
            
            # Change in cost
            # Cost = Size + Lambda * Uncovered
            # If valid (newly_uncovered == 0), Delta Cost = -1
            # If invalid, Delta Cost = -1 + Lambda * newly_uncovered
            Lambda = 10.0
            delta_cost = -1.0 + Lambda * newly_uncovered
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                cover_count = new_cover_count
                C.remove(c_old)
                
                # Update best if valid
                if newly_uncovered == 0:
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_cover_count = cover_count.copy()
            # else reject, no change
        else:
            if not C:
                break
            c_old = random.choice(C)
            # Pick a random c_new not in C
            # To avoid O(N) check, we can just pick a random index from 0..728
            # and check if it's in C. If yes, retry or pick another.
            # Given C is small (~80), a set is fast.
            C_set = set(C)
            while True:
                c_new = random.randint(0, total_words - 1)
                if c_new not in C_set:
                    break
            
            # Swap c_old with c_new
            # Remove c_old, Add c_new
            delta_cover = np.zeros(total_words, dtype=np.int32)
            for nb in neighbors[c_old]:
                delta_cover[nb] -= 1
            for nb in neighbors[c_new]:
                delta_cover[nb] += 1
            
            new_cover_count = cover_count + delta_cover
            newly_uncovered = np.sum((cover_count == 1) & (new_cover_count == 0))
            
            # Size doesn't change
            delta_cost = Lambda * newly_uncovered
            
            if delta_cost == 0 or random.random() < np.exp(-delta_cost / T):
                cover_count = new_cover_count
                C.remove(c_old)
                C.append(c_new)
                
                if newly_uncovered == 0:
                    # Size is same, but maybe structure is better?
                    # We don't update best unless size decreases, 
                    # but keeping valid state is good.
                    pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return best valid code found
    # Ensure best_C is valid
    # It should be, as we only update best_C when newly_uncovered == 0
    
    # Convert indices back to lists of ints
    result = []
    for idx in best_C:
        result.append(list(words[idx]))
        
    return result
