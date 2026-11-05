import itertools
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task
    # q=3, n=6, radius=1
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for general case
        # Trivial covering: all words
        return [[i // (q ** (n - 1 - pos)) % q for pos in range(n)] for i in range(q ** n)]

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

    # Precompute neighbors for all words using numpy for speed
    # neighbors[i] is a list of indices of words within Hamming distance 1 of word i
    neighbors = [[] for _ in range(total_words)]
    neighbors_set = [set() for _ in range(total_words)]
    for idx in range(total_words):
        word = idx_to_word(idx)
        neighbors[idx].append(idx)
        neighbors_set[idx].add(idx)
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    word[pos] = new_val
                    n_idx = word_to_idx(word)
                    neighbors[idx].append(n_idx)
                    neighbors_set[idx].add(n_idx)
            word[pos] = orig_val
            
    # Initial Code: 81 words from Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = h + list(tail)
            initial_codewords.append(word_to_idx(cw))
            
    # cover_count[w] = number of codewords in C that cover w
    cover_count = np.zeros(total_words, dtype=np.int32)
    C = set(initial_codewords)
    
    for c in C:
        for w in neighbors[c]:
            cover_count[w] += 1

    def is_valid_cover():
        return np.all(cover_count > 0)

    # Greedy Removal - optimized to scan once and collect all removable
    def greedy_remove():
        removed_any = True
        while removed_any:
            removed_any = False
            # Collect all redundant codewords in one pass
            redundant_list = []
            for c in C:
                # Check if c is redundant: all neighbors have cover_count > 1
                if np.all(cover_count[neighbors[c]] > 1):
                    redundant_list.append(c)
            # Remove all redundant codewords at once
            for c in redundant_list:
                C.remove(c)
                for w in neighbors[c]:
                    cover_count[w] -= 1
            if redundant_list:
                removed_any = True
        return len(C)

    # Initial Greedy Pass
    greedy_remove()
    
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    best_size = len(C)
    best_C = set(C)
    
    # Simulated Annealing / Randomized Local Search
    # Objective: Minimize |C|
    # Moves:
    # 1. Remove redundant codeword (always accepted)
    # 2. Swap codeword c_old with c_new such that coverage is maintained (always accepted if size decreases, otherwise accepted with probability exp(-delta/T))
    #    Note: A swap that maintains coverage but doesn't change size is a "neutral" move.
    #    A swap that maintains coverage and reduces size is an "improvement".
    #    Since we want to minimize size, we primarily look for removals.
    #    Swaps are used to escape local minima where no single removal is possible.
    
    # We will use a temperature schedule.
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute list of all words for random selection
    all_words = list(range(total_words))
    
    # Helper to check if a swap is valid (maintains coverage)
    def check_swap_valid(c_old, c_new):
        # c_old is in C, c_new is not in C
        # Words uniquely covered by c_old must be covered by c_new
        # i.e., for all w in neighbors[c_old] with cover_count[w] == 1, w must be in neighbors[c_new]
        for w in neighbors[c_old]:
            if cover_count[w] == 1:
                if w not in neighbors[c_new]:
                    return False
        return True
    
    # We can speed up check_swap_valid by precomputing critical words for c_old
    # critical_words[c_old] = [w for w in neighbors[c_old] if cover_count[w] == 1]
    # But cover_count changes.
    # For small N and |C|, recomputing is fine.
    
    while time.time() < deadline:
        # 1. Try to remove redundant codewords
        greedy_remove()
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = set(C)
            
        if best_size <= 73:
            # We hit the known best. Try to find 72 (unlikely) or just stop early if time is short?
            # Let's keep searching a bit more just in case, but prioritize finishing.
            if time.time() > deadline - 10:
                break

        # 2. Try a swap
        if len(C) == 0:
            break
            
        c_old = random.choice(list(C))
        
        # Identify critical words for c_old
        critical_words = []
        for w in neighbors[c_old]:
            if cover_count[w] == 1:
                critical_words.append(w)
                
        valid_c_new = []
        if not critical_words:
            # Any c_new not in C is valid.
            # Pick a random one.
            while True:
                c_new = random.choice(all_words)
                if c_new not in C:
                    valid_c_new = [c_new]
                    break
        else:
            # Intersection of neighbors of critical words
            candidates = set(neighbors[critical_words[0]])
            for w in critical_words[1:]:
                candidates &= set(neighbors[w])
                if not candidates:
                    break
            
            for c in candidates:
                if c not in C:
                    valid_c_new.append(c)
                    
        if valid_c_new:
            c_new = random.choice(valid_c_new)
            
            # Perform swap
            C.remove(c_old)
            for w in neighbors[c_old]:
                cover_count[w] -= 1
            
            C.add(c_new)
            for w in neighbors[c_new]:
                cover_count[w] += 1
            
            # The swap is valid by construction (coverage maintained).
            # Size remains the same.
            # This is a neutral move in terms of size, but changes the structure.
            # We accept it always? Or with probability?
            # In SA, we usually accept worse moves with probability.
            # Here, size is the same, so it's "equal". We accept to explore.
            
            # However, we should ensure we don't just cycle.
            # The greedy_remove at the start of the loop will clean up if the swap created redundancy.
            
        # Cool down
        T = max(T * cooling, T_min)
        
        # If we are stuck (no swaps possible or all swaps lead to immediate rejection if we had a cost function),
        # we might need a "kick" (remove a word even if it breaks coverage, then repair).
        # But our current swap logic requires coverage to be maintained.
        # If no valid c_new exists for a random c_old, we just try another c_old in the next iteration.
        
        # To escape local minima more effectively, we can occasionally try a "double swap" or "kick".
        # For now, let's stick to simple swaps + greedy removal.
        
        # If T is very low, we might want to increase it occasionally (reheating) if stuck?
        # Let's monitor if we've improved recently.
        # Simple heuristic: if no improvement in last X steps, reheat.
        # We don't track steps easily here without extra state.
        # Let's just keep cooling. The random selection of c_old and c_new provides diversity.

    # Final Greedy Pass
    greedy_remove()
    
    if len(C) < best_size:
        best_size = len(C)
        best_C = set(C)
        
    # Return the best found set
    return [idx_to_word(c) for c in best_C]
