import time
import random
import numpy as np

def construct(q, n, radius):
    # The task is specifically for q=3, n=7, radius=1.
    # We handle the general case by falling back to seed if parameters don't match,
    # but optimize for the specific instance.
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed logic for other params
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Direct sum
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(3), repeat=n)]

    total_words = 3 ** n
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Precompute neighbors for all words
    # Map word tuple to int index
    # We represent words as integers 0..2186
    
    def int_to_tuple(x):
        return tuple(int(x // (3**i) % 3) for i in range(n))
    
    def tuple_to_int(t):
        res = 0
        for i, v in enumerate(t):
            res += v * (3 ** i)
        return res

    # Generate all words
    all_words_int = list(range(total_words))
    
    # Precompute neighbors
    # neighbors[i] is a list of ints
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        t = int_to_tuple(i)
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            for new_val in range(3):
                if new_val == t[pos]:
                    continue
                new_t = list(t)
                new_t[pos] = new_val
                neighbors[i].append(tuple_to_int(tuple(new_t)))
        
    # Initialize code with seed
    # Seed: Direct sum of tetracode and full space of length 3
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    seed_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            seed_words.append(tuple_to_int(tuple(w)))
            
    current_code = list(set(seed_words)) # Ensure distinct
    code_set = set(current_code)
    
    # Coverage counts
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    best_size = len(current_code)
    best_code = list(current_code)
    
    # Greedy Local Search
    # Strategy:
    # 1. Try to remove a word.
    # 2. If removal causes uncovered words U, try to find a replacement word c' that covers U.
    # 3. Choose c' that minimizes "waste" (covers words already well-covered).
    
    while time.time() < deadline:
        improved = False
        # Shuffle order to avoid local minima bias
        indices = list(range(len(current_code)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            c = current_code[idx]
            
            # Check if c can be removed
            # Words that would become uncovered are those with cover_count == 1 and covered by c
            # Note: cover_count includes c
            would_be_uncovered = []
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    would_be_uncovered.append(nb)
            
            if not would_be_uncovered:
                # Safe to remove
                # Update counts
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                current_code.pop(idx)
                code_set.remove(c)
                best_size = len(current_code)
                best_code = list(current_code)
                improved = True
                break # Restart loop as counts changed
                
            else:
                # Need to replace c with some c'
                # c' must cover all words in would_be_uncovered
                # Also, c' should not be in current_code (unless we are just moving, but we are removing c, so c' != c)
                # Ideally c' is not in code_set to keep size same, or we swap.
                # Actually, if we add c', size becomes same.
                
                # Find best c'
                best_c_prime = -1
                best_cost = float('inf')
                
                # Candidate c' must cover all words in would_be_uncovered
                # We iterate through all possible words
                # Optimization: only consider words that are neighbors of some word in would_be_uncovered?
                # Actually, c' must be at distance 0 or 1 from all words in would_be_uncovered.
                # This is a strong constraint.
                
                # Let's just iterate all 2187 words. It's fast enough.
                for c_prime in range(total_words):
                    if c_prime in code_set:
                        continue # We are removing c, so if c_prime is in code, we'd have duplicate or just no change in set size if we swap? 
                                 # If c_prime is in code_set, adding it does nothing to coverage counts of its neighbors (already covered).
                                 # But we are removing c. So effectively we are swapping c with c_prime?
                                 # No, if c_prime is already in code, we can't add it. We can only replace c with a NEW word.
                                 # So c_prime must NOT be in code_set.
                                 # Wait, if c_prime is in code_set, then N(c_prime) is already covered.
                                 # If we remove c, and c_prime is in code, does c_prime cover would_be_uncovered?
                                 # If yes, then we don't need to add anything, we just remove c.
                                 # But we checked "would_be_uncovered" based on current counts.
                                 # If c_prime is in code, its neighbors have count >= 1 from c_prime.
                                 # So if a word w is in would_be_uncovered, cover_count[w] == 1.
                                 # Since c covers w, and c is the only one covering w (count 1), then c_prime (if in code) does NOT cover w.
                                 # So c_prime cannot be in code_set if it is to cover would_be_uncovered.
                                 # Thus, we only need to check c_prime NOT in code_set.
                                 
                    # Check if c_prime covers all words in would_be_uncovered
                    # Efficient check:
                    # Pre-compute for each c_prime, the set of words it covers? No, neighbors list is fine.
                    # Check intersection
                    covers_all = True
                    # Optimization: if len(would_be_uncovered) is large, this is slow.
                    # But would_be_uncovered is usually small.
                    # Let's use a set for would_be_uncovered for O(1) lookup?
                    # No, we need to check if w in neighbors[c_prime].
                    # neighbors[c_prime] is a list.
                    
                    # Faster:
                    # Create a set of neighbors of c_prime?
                    # Or just loop through would_be_uncovered.
                    
                    for w in would_be_uncovered:
                        if w not in neighbors[c_prime]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Calculate cost: 
                    # We want to minimize the number of "extra" covers we add.
                    # "Extra" means covering words that are already covered by >= 2 codewords (excluding c).
                    # After removing c, count of w becomes count[w] - 1.
                    # After adding c', count of w becomes (count[w] - 1) + 1 if w in N(c') else count[w] - 1.
                    # We want to minimize the sum of "redundancy" added.
                    # Cost = sum( max(0, (cover_count[w] - 1) - 1) for w in N(c_prime) )?
                    # Basically, if a word is already covered by >= 2 others (excluding c), covering it again is "waste".
                    # Let's define waste = number of words in N(c_prime) such that (cover_count[w] - 1) >= 2.
                    # i.e., cover_count[w] >= 3.
                    
                    cost = 0
                    for nb in neighbors[c_prime]:
                        if cover_count[nb] >= 3:
                            cost += 1
                            
                    if cost < best_cost:
                        best_cost = cost
                        best_c_prime = c_prime
                        
                if best_c_prime != -1:
                    # Perform swap
                    # Remove c
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code_set.remove(c)
                    current_code.pop(idx)
                    
                    # Add best_c_prime
                    for nb in neighbors[best_c_prime]:
                        cover_count[nb] += 1
                    current_code.append(best_c_prime)
                    code_set.add(best_c_prime)
                    
                    # Size is same, but structure is better.
                    # We don't update best_size here as size didn't change.
                    # But we might enable future deletions.
                    improved = True
                    break # Restart loop
                    
        if not improved:
            # Try random swaps to escape local minima?
            # Or just stop if greedy fails.
            # Let's try a few random moves if time permits.
            # For now, stop.
            pass
            
    # Ensure we return the best code found (smallest size)
    # best_code is updated on deletion.
    # If we only did swaps, best_code might be larger than current_code?
    # No, we only delete if it reduces size.
    # Swaps keep size same.
    # So best_size is the minimum size reached.
    # We should return the code corresponding to best_size.
    # If we only did swaps after reaching best_size, current_code has size best_size.
    # So returning current_code is safe if we track size correctly.
    
    # Re-verify best_code
    # If best_size < len(current_code), return best_code.
    # Else return current_code.
    
    if len(current_code) <= best_size:
        best_code = current_code
        best_size = len(current_code)
        
    # Convert back to lists of ints
    result = [list(int_to_tuple(c)) for c in best_code]
    
    return result

