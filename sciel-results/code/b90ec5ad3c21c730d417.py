import time
import random
import itertools

def construct(q, n, radius):
    # For this task, q=3, n=7, radius=1
    # General interface, but we optimize for the specific case if parameters match
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed or simple greedy for other cases
        # Seed logic for general case
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force or simple greedy for small n
            pass
            
    N = q ** n
    # Precompute neighbors
    # Represent words as tuples of ints
    # For speed, map to int
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute neighbor lists for all N words
    # neighbors[i] is a list of ints
    neighbors = [[] for _ in range(N)]
    all_words_int = list(range(N))
    
    # Generate all words as tuples for neighbor generation
    # It is faster to generate neighbors on the fly for the greedy?
    # No, precomputation is O(N * n * q) which is small.
    
    # Let's precompute using integer arithmetic or tuples?
    # Tuples are hashable.
    # Let's stick to tuples for clarity and correctness, then convert to int for speed if needed.
    # Actually, 2187 is small. Tuples are fine.
    
    words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors indices
    # neighbors[i] = set of indices
    neighbors_set = [set() for _ in range(N)]
    
    for i, w in enumerate(words):
        # Distance 0
        neighbors_set[i].add(i)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neighbors_set[i].add(word_to_idx[new_w])
                
    # Greedy Set Cover
    # We want to minimize the number of codewords.
    
    # Optimization: 
    # 1. Try multiple random orderings or tie-breaks.
    # 2. Use a "repair" phase: if greedy gets stuck or is suboptimal, try local search.
    
    best_code = None
    best_size = N + 1
    
    deadline = time.time() + 110 # 10s margin
    
    # Strategy 1: Standard Greedy with random tie-breaking
    # To diversify, we can shuffle the candidate list or use random priorities.
    
    for trial in range(50): # Try 50 times
        if time.time() > deadline:
            break
            
        uncovered = set(range(N))
        code = []
        
        # Randomize tie-breaking by adding small random noise to coverage count?
        # Or just shuffle the order we check candidates?
        # Checking all candidates every step is O(N^2 * n). 2187^2 * 7 ~ 33M. 
        # 50 trials * 33M = 1.6B. Too slow in Python.
        
        # We need a faster greedy.
        # Instead of checking all N candidates every step, 
        # we can maintain a list of "active" candidates.
        # But a candidate that was bad might become good? No, coverage only decreases.
        # If a candidate covers 0 new words, it will never cover >0 later.
        # So we can discard candidates that cover 0 new words.
        
        # However, calculating "covers 0 new words" requires checking intersection with uncovered.
        
        # Alternative: 
        # Start with all candidates.
        # At each step, pick the one with max intersection.
        # Update uncovered.
        # Remove candidates that have empty intersection with new uncovered?
        # This is expensive to update for all candidates.
        
        # Given N=2187, let's try to optimize the inner loop.
        # Using lists of ints for neighbors might be faster than sets for intersection?
        # `len(uncovered.intersection(neighbors[i]))`
        
        # Let's try a single run first to see how fast it is.
        
        # Optimization: 
        # If we use the direct sum (243) as a starting point and try to remove?
        # The direct sum is a valid cover.
        # We can try to remove codewords one by one if the remaining set still covers all.
        # This is a "reverse greedy" or "deletion" heuristic.
        
        # Let's implement the deletion heuristic on the direct sum.
        # It starts with 243.
        # Check each codeword: if removing it leaves all words covered, remove it.
        # Repeat until no more can be removed.
        
        # This is much faster to implement and might yield a good result.
        
        # Let's combine: 
        # 1. Generate Direct Sum (243).
        # 2. Apply Deletion Heuristic.
        # 3. Apply Greedy from scratch (maybe limited trials).
        # 4. Apply Local Search (Swap) on the result of 2 or 3.
        
        pass # Placeholder

    # Let's implement the Deletion Heuristic properly.
    
    # 1. Construct Direct Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to tuples for hashing
    initial_code_tuples = [tuple(c) for c in initial_code]
    
    # 2. Deletion Heuristic
    # We need to check if a set of codewords covers all words.
    # Efficient check:
    # Precompute the set of covered words for the current code?
    # If we remove one, we need to know if any word was ONLY covered by that one.
    
    # Let's maintain `cover_count[w]` = number of codewords covering w.
    # And `coverers[w]` = list of indices of codewords covering w?
    # Or just `cover_count` is enough to check validity (min count >= 1).
    # But to update `cover_count` efficiently when removing a codeword, 
    # we just decrement the count for all neighbors of that codeword.
    
    # Let's do this.
    
    def get_cover_counts(code_tuples):
        counts = [0] * N
        for c in code_tuples:
            idx = word_to_idx[c]
            for nb in neighbors_set[idx]:
                counts[nb] += 1
        return counts

    def is_valid_cover(counts):
        return all(c >= 1 for c in counts)

    current_code = initial_code_tuples[:]
    counts = get_cover_counts(current_code)
    
    # Deletion Loop
    changed = True
    while changed:
        changed = False
        # Try to remove each codeword
        # To be faster, we can iterate in random order or specific order.
        # If we remove one, counts change.
        # We can iterate through indices.
        
        # Optimization: If a codeword has a neighbor with count 1, we CANNOT remove it.
        # So we only try to remove codewords where ALL neighbors have count > 1.
        
        for i in range(len(current_code)):
            if time.time() > deadline:
                break
                
            c = current_code[i]
            idx = word_to_idx[c]
            
            # Check if safe to remove
            safe = True
            for nb in neighbors_set[idx]:
                if counts[nb] == 1:
                    safe = False
                    break
                    
            if safe:
                # Remove it
                # Update counts
                for nb in neighbors_set[idx]:
                    counts[nb] -= 1
                current_code.pop(i)
                changed = True
                # After removal, previous indices are shifted. 
                # To keep it simple, we can just break and restart the loop?
                # Or iterate in reverse?
                # Or just mark as deleted and filter at end?
                # Let's use a boolean array `active` to avoid popping issues.
                pass # The logic above with pop(i) is risky for indices.
                
    # Let's rewrite the deletion loop with an `active` flag.
    
    active = [True] * len(initial_code_tuples)
    current_code = initial_code_tuples[:]
    counts = get_cover_counts(current_code)
    
    changed = True
    while changed:
        changed = False
        # Shuffle indices to break symmetry?
        indices = list(range(len(current_code)))
        random.shuffle(indices)
        
        for i in indices:
            if not active[i]:
                continue
                
            c = current_code[i]
            idx = word_to_idx[c]
            
            safe = True
            for nb in neighbors_set[idx]:
                if counts[nb] == 1:
                    safe = False
                    break
                    
            if safe:
                active[i] = False
                for nb in neighbors_set[idx]:
                    counts[nb] -= 1
                changed = True

    final_code = [current_code[i] for i in range(len(current_code)) if active[i]]
    
    # Convert back to lists
    result = [list(c) for c in final_code]
    
    # Optional: Try to improve with Local Search (Swap)
    # If we have time, try swapping one codeword with another to reduce count?
    # No, swap keeps count same. We want to reduce count.
    # We can try to add a codeword that covers many, then remove two?
    # That's complex.
    
    # Let's see if we can do better with a pure Greedy from scratch.
    # If the deletion heuristic gets, say, 200, maybe greedy gets 190?
    
    # Let's add a simple Greedy from scratch if time permits.
    
    if time.time() < deadline - 5:
        # Greedy from scratch
        # To make it fast, we only check candidates in the current `uncovered` set?
        # No, a covered codeword might cover many uncovered ones.
        # But we can limit the search space.
        
        # Let's try one Greedy run.
        uncovered = set(range(N))
        g_code = []
        
        # To speed up, we can pre-compute neighbor lists as lists of ints.
        # neighbors_list[i] = list(neighbors_set[i])
        neighbors_list = [list(s) for s in neighbors_set]
        
        while uncovered:
            best_c = -1
            best_count = 0
            # Iterate all candidates
            for c_idx in range(N):
                # Optimization: if c_idx is already in g_code, skip? 
                # No, we shouldn't add duplicates.
                # But if it's already added, its neighbors are covered, so count will be 0.
                # So we can just check all.
                
                count = 0
                for nb in neighbors_list[c_idx]:
                    if nb in uncovered:
                        count += 1
                        if count > best_count:
                            # Early exit? No, we need max.
                            pass
                            
                if count > best_count:
                    best_count = count
                    best_c = c_idx
                    
            if best_c == -1:
                break # Should not happen if uncovered not empty
                
            g_code.append(best_c)
            # Update uncovered
            for nb in neighbors_list[best_c]:
                uncovered.discard(nb)
                
        # Check if this greedy result is better
        if len(g_code) < len(final_code):
            final_code = [words[c] for c in g_code]
            result = [list(c) for c in final_code]

    return result
