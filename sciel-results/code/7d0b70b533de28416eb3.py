import time
import itertools
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: greedy
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = word_to_idx[tuple(new_w)]
                    neighbors[i].append(j)
        
        code = []
        uncovered = [True] * N
        while any(uncovered):
            best_c = -1
            best_count = -1
            for c in range(N):
                count = sum(1 for nb in neighbors[c] if uncovered[nb])
                if count > best_count:
                    best_count = count
                    best_c = c
            code.append(best_c)
            for nb in neighbors[best_c]:
                uncovered[nb] = False
        
        cover_count = [0] * N
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(code)):
                c = code[i]
                if all(cover_count[nb] > 1 for nb in neighbors[c]):
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code.pop(i)
                    improved = True
                    break
        
        return [list(all_words[c]) for c in code]

    # Specific optimization for q=3, n=6, radius=1
    # Precompute neighbors
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # neighbors[i] is a list of indices within distance 1
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                neighbors[i].append(j)
    
    # Use a set for fast membership check in code
    code_set = set()
    cover_count = [0] * N
    code = []
    
    def add_word(idx):
        if idx in code_set:
            return
        code_set.add(idx)
        code.append(idx)
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    def remove_word(idx):
        if idx not in code_set:
            return
        code_set.remove(idx)
        code.remove(idx)
        for nb in neighbors[idx]:
            cover_count[nb] -= 1
            
    def is_covered(idx):
        return cover_count[idx] > 0
        
    def try_remove(idx):
        # Can we remove idx?
        for nb in neighbors[idx]:
            if cover_count[nb] == 1:
                return False
        return True
        
    def do_remove(idx):
        remove_word(idx)
        
    # Greedy Construction
    # We want to minimize the number of words.
    # Start with empty.
    uncovered = [True] * N
    
    while any(uncovered):
        best_c = -1
        best_gain = -1
        # Optimization: Only consider candidates that cover at least one uncovered word
        # For each candidate c, gain = number of uncovered words in neighbors[c]
        for c in range(N):
            gain = 0
            for nb in neighbors[c]:
                if uncovered[nb]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
            if best_gain == len(neighbors[c]):
                break
                
        add_word(best_c)
        for nb in neighbors[best_c]:
            uncovered[nb] = False
            
    # Local Search: Removal
    # Iterate until no removal possible
    improved = True
    while improved:
        improved = False
        for i in range(len(code)):
            if try_remove(code[i]):
                do_remove(code[i])
                improved = True
                break # Restart loop as list changed
                
    # Local Search: Swap to enable removal
    # Strategy: 
    # 1. Find a codeword c.
    # 2. Identify critical words C_c (words covered only by c).
    # 3. If C_c is empty, remove c. (Handled above, but maybe counts changed? No, we only swap if valid).
    # 4. Find a candidate c' that covers all words in C_c.
    # 5. Check if swapping c with c' keeps the code valid (i.e., no word becomes uncovered).
    #    Condition: For all w not in C_c, if w was covered only by c (impossible, C_c is exactly those), 
    #    wait. 
    #    Let S be the set of words covered by c.
    #    C_c = {w in S | cover_count[w] == 1}
    #    If we swap c -> c', we remove c and add c'.
    #    Words in S \ C_c still have cover_count >= 2, so they become cover_count - 1 >= 1. Safe.
    #    Words in C_c had cover_count 1. They must be covered by c'. So c' must cover C_c.
    #    Words outside S: 
    #       If they were covered by c? No, S is all words covered by c.
    #       So words outside S are not affected by removing c.
    #       Adding c' might cover new words, which is fine (increases count).
    #    So the only condition is: c' must cover all words in C_c.
    #    AND: The code must remain valid for words NOT in S.
    #    Wait, does swapping affect words not in S?
    #    Removing c only affects S.
    #    Adding c' affects S' (neighbors of c').
    #    Words in S' \ S gain cover count. No harm.
    #    Words in S \ S' lose cover count (from c). If they were in C_c, they must be in S' (covered by c').
    #    So if c' covers C_c, then for w in S:
    #       If w in C_c: cover_count becomes (1-1) + 1 (from c') = 1. Safe.
    #       If w not in C_c: cover_count becomes (k-1) + (1 if w in S' else 0).
    #          Since k >= 2, k-1 >= 1. So safe.
    #    Therefore, the ONLY condition for a valid swap is that c' covers all words in C_c.
    
    # Implementation:
    # Precompute for each word w, the list of codewords that cover it?
    # Or just iterate.
    
    # To speed up finding c' that covers C_c:
    # C_c is small (usually 1 or 2).
    # If C_c is empty, remove c.
    # If |C_c| == 1, let w be the word. c' must be in neighbors[w].
    # If |C_c| == 2, let w1, w2 be words. c' must be in neighbors[w1] intersection neighbors[w2].
    
    # We will iterate through codewords and try to swap.
    # To avoid getting stuck in loops, we can track visited states?
    # Or just limit iterations.
    
    max_swaps = 5000
    swaps_done = 0
    stuck_counter = 0
    last_code_size = len(code)
    
    while time.time() - start_time < deadline - 5 and swaps_done < max_swaps:
        swapped = False
        # Shuffle code to avoid bias
        random.shuffle(code)
        
        for i in range(len(code)):
            if time.time() - start_time > deadline - 5:
                break
            
            c = code[i]
            criticals = []
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    criticals.append(nb)
            
            if not criticals:
                # Redundant
                do_remove(c)
                swapped = True
                break # Restart inner loop
            
            # Find candidates for c'
            if len(criticals) == 1:
                cands = neighbors[criticals[0]]
            elif len(criticals) == 2:
                # Intersection
                set1 = set(neighbors[criticals[0]])
                set2 = set(neighbors[criticals[1]])
                cands = list(set1.intersection(set2))
            else:
                # Harder. Intersection of 3+ sets.
                # Unlikely to have 3+ criticals for radius 1 in ternary 6?
                # Ball size is 19.
                # If a codeword has 3 criticals, it means 3 words are uniquely covered by it.
                # It's possible.
                # Let's handle generically.
                cands = set(neighbors[criticals[0]])
                for crit in criticals[1:]:
                    cands.intersection_update(set(neighbors[crit]))
                cands = list(cands)
            
            if not cands:
                continue
                
            # Try candidates
            # We prefer candidates that are not already in code.
            # Also, we might prefer candidates that cover many low-count words?
            # For now, just iterate.
            random.shuffle(cands)
            for c_prime in cands:
                if c_prime == c:
                    continue
                if c_prime in code_set:
                    # Swapping with an existing codeword?
                    # If c_prime is in code, then c_prime is already covering criticals?
                    # If c_prime is in code, and c covers criticals uniquely, then c_prime cannot cover criticals?
                    # Wait. If c_prime is in code, and w is in C_c, then cover_count[w] >= 1 from c_prime?
                    # No, C_c means cover_count[w] == 1 and covered by c.
                    # So no other codeword covers w.
                    # Thus c_prime (if in code) cannot cover w.
                    # So c_prime in code implies c_prime does NOT cover C_c.
                    # So we can skip if c_prime in code.
                    continue
                
                # Swap c with c_prime
                # Check validity: c_prime must cover all criticals.
                # We already constructed cands such that they cover criticals.
                # So validity is guaranteed by the construction of cands?
                # Yes, as reasoned above.
                
                # Perform swap
                remove_word(c)
                add_word(c_prime)
                swapped = True
                swaps_done += 1
                break # Break out of cands loop
            
            if swapped:
                break # Break out of code loop
        
        if not swapped:
            stuck_counter += 1
            # If stuck for a while, maybe perturb?
            # Or just break.
            if stuck_counter > 10:
                break
        else:
            stuck_counter = 0
            
    # Final reduction pass
    improved = True
    while improved:
        improved = False
        for i in range(len(code)):
            if try_remove(code[i]):
                do_remove(code[i])
                improved = True
                break

    return [list(all_words[c]) for c in code]
