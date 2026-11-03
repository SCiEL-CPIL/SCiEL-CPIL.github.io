import itertools
import random
import time
import sys

# Precomputation for q=3, n=7, radius=1
# We can generalize, but the task is specific. 
# The interface is construct(q, n, radius). 
# The evaluator passes q=3, n=7, radius=1.
# We should handle the general case if possible, or at least not crash.
# Given the specific nature of the "best known" bounds, we focus on 3,7,1.
# However, the seed program works for general n. 
# If n is small, we can do exhaustive or specific search.
# If n is large, the direct product is good.
# For 3,7,1, we want to beat 186.

def construct(q, n, radius):
    # If the problem is the specific ternary covering code K_3(7,1)
    if q == 3 and n == 7 and radius == 1:
        return _solve_k3_7_1()
    
    # Fallback for other parameters: use the seed logic or simple greedy
    # The seed logic:
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # This gives 9 * 3^(n-4)
        # For n=7, 9 * 27 = 243. Wait, seed score is 1.0 which is 186?
        # Let's re-read the seed code.
        # Seed: tetracode (9 words) + tails (3^(n-4)).
        # For n=7, 9 * 3^3 = 243.
        # But the prompt says "The best known code has 186 words... Score: ... seed program ... 1.0000".
        # If the seed program produced 243, and the best is 186, the score would be 186/243 < 1.
        # The prompt says "score is seed/value ... 1.0 equals the seed program".
        # This implies the seed program *does* produce 186?
        # Let's check the seed code again.
        # `tetracode` has 9 words.
        # `itertools.product(range(3), repeat=n - 4)`.
        # If n=7, repeat=3. 3^3 = 27.
        # 9 * 27 = 243.
        # Why is the score 1.0?
        # Maybe the "seed program" in the prompt description is NOT the code provided?
        # "CURRENT PROGRAM (score 1.0000): ... tetracode ...".
        # If the current program yields 243, and the baseline is 243, then score is 1.0.
        # But the text says "best known ... 186".
        # If I output 186, my score will be 243/186 = 1.30.
        # If I output 180, my score will be 243/180 = 1.35.
        
        # Wait, the prompt says: "Score: the number of codewords, minimized; it is an upper bound on K_q(n, radius)."
        # "The evaluator measures the raw value (lower is better). The search score is seed/value..."
        # If seed value is 243, and I produce 186, value=186. Score = 243/186 > 1.
        # So I should aim for the smallest possible number of codewords.
        
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Generic fallback
    # Simple greedy
    return _generic_greedy(q, n, radius)


def _solve_k3_7_1():
    q = 3
    n = 7
    radius = 1
    total_words = q ** n  # 2187
    
    # Map word (tuple) to index (int)
    # Base-q representation
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    # Precompute neighbors for each word
    # neighbors[i] is a list of indices within distance <= 1
    neighbors = [None] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neigh.append(word_to_idx(tuple(new_w)))
        neighbors[i] = neigh

    # Precompute coverage bitmasks
    # 2187 integers of 2187 bits. 2187 * 2187 / 8 bytes ~ 600KB. This is fine.
    cover_masks = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        cover_masks[i] = mask

    # Greedy Construction with time guard
    start_time = time.time()
    time_limit = 115.0  # Leave 35s margin for local search
    
    U = (1 << total_words) - 1
    C = []
    counts = [0] * total_words
    in_C = [False] * total_words  # Track which words are in C for O(1) lookup
    
    while U != 0:
        if time.time() - start_time > time_limit:
            break
        
        best_c = -1
        max_covered = 0
        # Iterate through all possible candidates
        for c in range(total_words):
            if in_C[c]:
                continue
            covered = (cover_masks[c] & U).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_c = c
        
        if best_c == -1:
            break
            
        # Add best_c
        C.append(best_c)
        in_C[best_c] = True
        mask = cover_masks[best_c]
        U &= ~mask
        # Update counts
        for j in neighbors[best_c]:
            counts[j] += 1

    # Local Search: Try to reduce size
    # We have a valid cover C.
    # Strategy:
    # 1. Try to remove a word.
    # 2. If removal makes U non-empty, we failed.
    # 3. To escape local minima, we can "swap" a word.
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    best_C = C[:]
    best_size = len(C)
    
    # Helper to check if C is valid and get U
    # We maintain counts and U incrementally.
    
    while time.time() - start_time < time_limit:
        # Try random deletions
        deleted = False
        # Shuffle C to try different orders? Or just pick random indices.
        # We try to delete one word at a time.
        
        # Randomly pick a codeword to try to remove
        # We iterate through a random permutation of C
        perm = C[:]
        random.shuffle(perm)
        
        for c in perm:
            if time.time() - start_time > time_limit:
                break
                
            # Try to remove c
            # Check if any neighbor of c has count 1
            removable = True
            for j in neighbors[c]:
                if counts[j] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove c
                C.remove(c)
                for j in neighbors[c]:
                    counts[j] -= 1
                    # U update: if counts[j] becomes 0, set bit j in U
                    # But we are checking validity, so if we remove and counts[j] > 0 for all j, U remains 0.
                    # Wait, U is 0 if and only if counts[j] > 0 for all j.
                    # Since we verified counts[j] > 1 for all j in neighbors[c] (which is the only place counts decrease),
                    # and counts[j] for j not in neighbors[c] are unchanged (and >0),
                    # the new set is still a valid cover.
                    pass
                
                deleted = True
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C[:]
                
                # If we deleted, we should restart the permutation loop or continue?
                # Continuing is fine, we might delete more.
                # But the `perm` list contains `c` which is now gone. 
                # We are iterating `perm`. If we remove `c` from `C`, subsequent items in `perm` are still in `C`.
                # We should break and restart the shuffle to account for the new state?
                # Or just continue.
                break # Restart the shuffle loop to pick a new random word from the smaller set
        
        if not deleted:
            # Stuck. Try a swap to change configuration.
            # Pick a random c in C.
            # Try to replace it with a c' not in C such that C - {c} + {c'} is a valid cover.
            # This is equivalent to:
            # 1. Remove c.
            # 2. Let U_temp be the uncovered words (those with count 0 after removal).
            # 3. Find c' such that cover_masks[c'] covers all bits in U_temp.
            
            if C:
                c = random.choice(C)
                
                # Temporarily remove c
                for j in neighbors[c]:
                    counts[j] -= 1
                
                # Find uncovered words
                U_temp = 0
                for j in neighbors[c]:
                    if counts[j] == 0:
                        U_temp |= (1 << j)
                
                if U_temp == 0:
                    # This means c was redundant? We should have caught this in the deletion phase.
                    # Maybe the deletion phase failed because of the `break` logic?
                    # If U_temp == 0, we can just keep it removed.
                    C.remove(c)
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C[:]
                    # Don't restore
                else:
                    # We need to add a c' that covers U_temp
                    # c' must be in the set of all words.
                    # We can restrict c' to words that intersect U_temp significantly.
                    # Or just pick a random word from the universe?
                    # Better: Pick a random word from the universe that covers the most of U_temp?
                    # Or just try a few random candidates.
                    
                    found_replacement = False
                    # Try the original c (backtrack)
                    # Or try a random neighbor of the uncovered words?
                    
                    # Heuristic: Pick a random word w from the entire space.
                    # If cover_masks[w] & U_temp == U_temp, then w is a valid replacement.
                    # Probability of a random word covering a specific small set U_temp is decent.
                    
                    # To speed up, we can pick a random word from the neighbors of the uncovered words.
                    # Let's pick a random word from the universe.
                    # If we fail, we backtrack.
                    
                    # Try a few random candidates
                    for _ in range(10):
                        c_prime = random.randint(0, total_words - 1)
                        if c_prime in C:
                            continue
                        if (cover_masks[c_prime] & U_temp) == U_temp:
                            # Valid replacement
                            C.remove(c)
                            C.append(c_prime)
                            # Update counts
                            for j in neighbors[c_prime]:
                                counts[j] += 1
                            found_replacement = True
                            break
                    
                    if not found_replacement:
                        # Backtrack: restore c
                        for j in neighbors[c]:
                            counts[j] += 1
                        # Note: if c was in C, we didn't remove it from C in the "Temporarily remove" step 
                        # because we only modified counts. 
                        # Wait, in the deletion phase, we did `C.remove(c)`.
                        # Here, we did NOT do `C.remove(c)`. We only modified counts.
                        # So C still contains c.
                        # We just need to restore counts.
                        pass

    # Convert best_C indices back to words
    result = [list(idx_to_word(idx)) for idx in best_C]
    return result


def _generic_greedy(q, n, radius):
    # Fallback for other parameters
    # Simple greedy might be slow for large n, but for small n it's fine.
    # If n is large, use the tetracode product if q=3.
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic Greedy
    total = q ** n
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    # Precompute neighbors (distance <= radius)
    # This is expensive for large n/radius.
    # For the specific task, we use the specialized solver.
    # This fallback is just to ensure it doesn't crash.
    
    # Simple random construction if precomputation is too heavy
    # Just return a set of random words? No, must be valid.
    # If we can't compute neighbors, we can't guarantee validity.
    # But the task is specific to 3,7,1.
    
    # Return empty list if unknown? No, must be valid.
    # For safety, if not 3,7,1, return a large set (all words) which is always valid.
    return [list(idx_to_word(i)) for i in range(total)]

# EVOLVE-BLOCK-START
# The above functions are defined.
# We need to make sure `construct` is the entry point.
# EVOLVE-BLOCK-END
