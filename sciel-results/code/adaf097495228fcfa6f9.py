import itertools
import random
import time

def construct(q, n, radius):
    # Parameters for the specific task
    # q=3, n=6, radius=1
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for general case (not expected to be called with other params in this specific task context, 
        # but good to have a basic constructor)
        # For now, return a trivial covering (all words) if params don't match, 
        # though the task specifies this is for the ternary covering code problem.
        # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
        # So we assume q=3, n=6, radius=1.
        pass

    total_words = q ** n
    # Map word tuple to index
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

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices of words within Hamming distance 1 of word i
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        word = idx_to_word(idx)
        neighbors[idx].append(idx) # distance 0
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    word[pos] = new_val
                    n_idx = word_to_idx(word)
                    neighbors[idx].append(n_idx)
            word[pos] = orig_val
            
    # Initial Code: 81 words from Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = h + list(tail)
            initial_codewords.append(word_to_idx(cw))
            
    # cover_count[w] = number of codewords in C that cover w
    cover_count = [0] * total_words
    C = set(initial_codewords)
    
    for c in C:
        for w in neighbors[c]:
            cover_count[w] += 1

    def is_valid_cover():
        return all(cover_count[w] > 0 for w in range(total_words))

    # Greedy Removal
    def greedy_remove():
        removed_any = True
        while removed_any:
            removed_any = False
            # Try to remove any redundant codeword
            # We iterate through a copy of C to allow modification
            for c in list(C):
                # Check if c is redundant
                # A codeword c is redundant if for all w in neighbors[c], cover_count[w] > 1
                redundant = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    C.remove(c)
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    removed_any = True
                    break # Restart loop as counts changed
        return len(C)

    # Initial Greedy Pass
    greedy_remove()
    
    # Randomized Swap Search
    # We try to swap a codeword c_old with a new word c_new
    # Condition: All words w that are uniquely covered by c_old must be covered by c_new
    # Note: If a word w is covered by c_old and others, it's fine if c_old is removed.
    # So we only check words w where cover_count[w] == 1 and w in neighbors[c_old].
    
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_size = len(C)
    
    # To speed up swap checks, we can precompute the set of "critical" words for each codeword
    # critical[c] = set of w in neighbors[c] such that cover_count[w] == 1
    # But cover_count changes, so we must update this or recompute.
    # Given N=729 and |C|~75, recomputing critical sets is cheap if done locally.
    
    while time.time() < deadline:
        greedy_remove()
        
        if len(C) < best_size:
            best_size = len(C)
        
        # If we reached 73 or lower, we are very good. 
        # The known bound is 73. If we hit 73, we can stop or try for 72 (unlikely/impossible).
        if best_size <= 73:
             # Try a bit more to see if we can go lower, but 73 is the target.
             pass

        # Try swaps
        # Pick a random codeword to replace
        c_old_list = list(C)
        if not c_old_list:
            break
        c_old = random.choice(c_old_list)
        
        # Identify words uniquely covered by c_old
        critical_words = []
        for w in neighbors[c_old]:
            if cover_count[w] == 1:
                critical_words.append(w)
        
        # We need to find a c_new such that all critical_words are in neighbors[c_new]
        # And c_new is not in C (or if it is, it's a no-op swap, which we skip)
        
        # To find valid c_new efficiently:
        # If critical_words is empty, any c_new not in C is valid.
        # If critical_words is not empty, c_new must be within distance 1 of ALL critical words.
        # This is the intersection of the balls B(w) for w in critical_words.
        
        valid_c_new = []
        if not critical_words:
            # Any word not in C is a candidate.
            # To avoid checking all 729, we can just pick a random one not in C.
            # But we want to explore.
            # Let's just pick a random candidate.
            while True:
                c_new = random.randint(0, total_words - 1)
                if c_new not in C:
                    valid_c_new = [c_new]
                    break
        else:
            # Intersection of neighbors of critical words
            # Start with neighbors of the first critical word
            candidates = set(neighbors[critical_words[0]])
            for w in critical_words[1:]:
                candidates &= set(neighbors[w])
                if not candidates:
                    break
            
            # Filter out words already in C
            for c in candidates:
                if c not in C:
                    valid_c_new.append(c)
        
        if valid_c_new:
            c_new = random.choice(valid_c_new)
            
            # Perform Swap
            C.remove(c_old)
            for w in neighbors[c_old]:
                cover_count[w] -= 1
            
            C.add(c_new)
            for w in neighbors[c_new]:
                cover_count[w] += 1
            
            # Check validity just in case (should be valid by construction)
            # if not is_valid_cover():
            #     # Rollback
            #     C.remove(c_new)
            #     for w in neighbors[c_new]:
            #         cover_count[w] -= 1
            #     C.add(c_old)
            #     for w in neighbors[c_old]:
            #         cover_count[w] += 1
            #     continue

        # Occasionally, try to remove multiple redundant codewords if the swap created redundancy
        # The next loop iteration will call greedy_remove()
        
        # If we are stuck at a local minimum (no swaps valid or no removals),
        # we might try a "kick": remove a codeword even if it breaks coverage,
        # then try to repair? 
        # For now, the swap mechanism should be sufficient to escape most local minima 
        # because it allows moving the "center" of a cluster.

    # Final Greedy Pass
    greedy_remove()
    
    # Return list of codewords
    return [idx_to_word(c) for c in C]
