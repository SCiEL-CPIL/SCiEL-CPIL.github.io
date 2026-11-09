import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit (or 120s if strict)
    
    # Specific parameters for this task: q=3, n=8, radius=1
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for other parameters
        total_words = q ** n
        if total_words > 10000:
            if q == 3 and n >= 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Generic small case fallback
        centers = []
        covered = set()
        all_words = set(itertools.product(range(q), repeat=n))
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while covered != all_words and time.time() < deadline:
            best_center = None
            max_new = 0
            for c in candidates:
                neigh = [c]
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_c = list(c)
                        new_c[i] = val
                        neigh.append(tuple(new_c))
                new_covered = [w for w in neigh if w not in covered]
                if len(new_covered) > max_new:
                    max_new = len(new_covered)
                    best_center = c
                    best_neigh = neigh
            if best_center is None or max_new == 0:
                break
            centers.append(list(best_center))
            for w in best_neigh:
                covered.add(w)
            candidates.remove(best_center)
        return centers

    # --- Optimized Path for q=3, n=8, radius=1 ---
    
    total_words = 3 ** 8  # 6561
    
    # 1. Pre-compute mapping
    # We use a list of indices for words.
    # word_to_idx: tuple -> int
    # idx_to_word: int -> tuple
    word_to_idx = {}
    idx_to_word = [None] * total_words
    
    for word in itertools.product(range(3), repeat=8):
        idx = 0
        for digit in word:
            idx = idx * 3 + digit
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        
    # 2. Pre-compute neighbors
    # neighbors[i] is a list of indices covered by center i (including i)
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        center = idx_to_word[i]
        neighbors[i].append(i)
        for pos in range(8):
            old_val = center[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                n_idx = word_to_idx[tuple(neighbor)]
                neighbors[i].append(n_idx)
                
    # 3. Initial Code Construction: The 486-word code
    # The optimal code for K_3(8,1) is often constructed as a direct sum of smaller codes
    # or via grid structures.
    # A known optimal code of size 486 can be generated.
    # One common construction for q=3, n=4, r=1 is size 9? No, K_3(4,1)=9 is not right?
    # K_3(4,1) = 9? Let's check. 3^4=81. 9*13 = 117 > 81. 
    # Actually, K_3(4,1) is 9? No.
    # Let's use the product construction if we know components.
    # K_3(8,1) = K_3(4,1) * K_3(4,1)?
    # If K_3(4,1) = 9, then 9*9 = 81? No, 486 is much larger.
    # Wait, 486 = 2 * 3^5 = 2 * 243.
    # Or 486 = 6 * 81.
    # Let's just generate a valid code using a fast greedy first to ensure we have a valid start,
    # then try to optimize from there. The prompt says "best known code has 486".
    # If I can't hardcode 486 easily, I'll use a fast greedy to get a valid code (likely ~500-600)
    # and then ILS will try to shrink it.
    # Actually, hardcoding the structure is better if possible.
    # The code of size 486 for q=3, n=8, r=1 is related to the hexacode or similar?
    # Let's try a greedy initialization but make it very fast.
    
    # Fast Greedy Initialization
    # We want a valid code quickly.
    # Use a simple heuristic: pick centers that cover the most uncovered points.
    # To speed up, we can just pick random centers until covered? No, that's huge.
    # Greedy is O(N^2) roughly. 6561^2 is 43M. In Python, this might take a few seconds.
    # Let's do a randomized greedy with early stopping if we hit a good size.
    
    def fast_greedy_init():
        uncovered = set(range(total_words))
        code_indices = []
        
        # Pre-compute masks as sets for faster intersection? 
        # Or just use the neighbors list.
        # To speed up greedy:
        # Maintain a count of how many uncovered points each candidate covers.
        # This is expensive to update.
        # Alternative: Just pick the first one that covers the most?
        # Let's stick to the previous greedy but maybe limit it?
        # Actually, the previous program's greedy took time.
        # Let's try a simpler initialization:
        # The 486 code can be constructed.
        # Reference: "A ternary covering code of length 8 and radius 1 with 486 codewords"
        # Construction:
        # It is a product of two codes?
        # If we take the code of size 9 for n=4? 
        # K_3(4,1) = 9? 
        # Let's check K_3(4,1).
        # Lower bound: 81/13 = 6.2. Upper bound?
        # A code of size 9 exists? 
        # If we take all words with sum 0 mod 3? No.
        
        # Let's just run the greedy from the previous program but optimized?
        # Or better: Start with a random valid code generated by a fast heuristic.
        # Heuristic: 
        # 1. Start with empty code.
        # 2. While not covered:
        #    Pick a random uncovered word w.
        #    Pick a random center c from the neighbors of w that is not in code?
        #    Or pick the center c that covers the most uncovered neighbors of w?
        
        # Let's use a "randomized greedy" that is fast.
        # We will simulate the previous greedy but with numpy? 
        # No, let's just use the previous greedy logic but maybe it's fast enough?
        # The previous program did 50 restarts. One greedy run might take 1-2s.
        # So we have ~100s for ILS.
        
        # Let's implement a fast greedy:
        # We maintain a list of uncovered indices.
        # At each step, we pick the center that covers the most uncovered indices.
        # To avoid O(N) scan of all N centers every time, we can use a priority queue?
        # Or just scan. 6561 centers. 486 steps. 6561 * 486 ~ 3M operations. 
        # Each operation involves checking intersection with uncovered set.
        # If uncovered is a set, intersection is fast.
        
        uncovered_set = set(range(total_words))
        code = []
        
        # To speed up, we can pre-compute neighbors as sets?
        # neighbors_sets = [set(neighbors[i]) for i in range(total_words)]
        # This takes memory but speeds up intersection.
        
        while uncovered_set:
            if time.time() > deadline - 50: # Leave time for ILS
                break
            best_center = -1
            max_covered = 0
            # Randomize order to avoid bias?
            # Just scan all.
            for c in range(total_words):
                # If c is already in code, skip? 
                # No, if c is in code, its neighbors are already covered, so intersection with uncovered is 0.
                # So we can skip centers already in code.
                if c in code_set: continue
                
                covered_by_c = len(neighbors_sets[c] & uncovered_set)
                if covered_by_c > max_covered:
                    max_covered = covered_by_c
                    best_center = c
            
            if best_center == -1:
                break
                
            code.append(best_center)
            code_set.add(best_center)
            uncovered_set -= neighbors_sets[best_center]
            
        return code

    # Pre-compute neighbors as sets for the greedy init
    neighbors_sets = [set(neighbors[i]) for i in range(total_words)]
    
    # Run one greedy init to get a valid code
    code_indices = fast_greedy_init()
    
    # If greedy failed to cover everything (timeout), we need a valid code.
    # Check if valid
    covered_union = set()
    for c in code_indices:
        covered_union |= neighbors_sets[c]
        
    if covered_union != set(range(total_words)):
        # Fallback: Tetracode sum (729 words) is always valid.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = tuple(h + list(tail))
                code_indices.append(word_to_idx[w])
        
    # Now we have a valid code. Let's try to minimize it using ILS.
    # State: list of center indices.
    # We need to be able to check validity and repair.
    
    def is_valid(code):
        covered = set()
        for c in code:
            covered |= neighbors_sets[c]
        return len(covered) == total_words
        
    def repair(code):
        """
        Attempts to repair the code by adding centers to cover any uncovered words.
        Returns True if repaired, False if failed (should not happen if we add enough).
        """
        covered = set()
        for c in code:
            covered |= neighbors_sets[c]
            
        uncovered = set(range(total_words)) - covered
        
        while uncovered:
            if time.time() > deadline - 10:
                return False
            
            # Pick an uncovered word
            w = random.choice(list(uncovered))
            
            # Find a center that covers w and is not in code
            # To minimize size increase, pick a center that covers the most other uncovered words?
            # Or just pick a random neighbor of w?
            # Greedy repair: pick center c in neighbors[w] that covers most uncovered.
            best_c = -1
            max_cov = 0
            for c in neighbors[w]:
                if c in code_set:
                    continue
                cov = len(neighbors_sets[c] & uncovered)
                if cov > max_cov:
                    max_cov = cov
                    best_c = c
            
            if best_c == -1:
                # All neighbors are in code? That means w is covered by a center in code?
                # But w is uncovered. Contradiction?
                # Unless we removed a center that was the *only* one covering w, but we didn't add it back?
                # Wait, if c is in code, it covers w. So w would be in covered.
                # So if w is uncovered, no center in code covers w.
                # So best_c should not be -1 unless neighbors[w] is empty (impossible).
                # Maybe all neighbors are in code? If c is in code, it covers w.
                # So w would be covered.
                # So this case is impossible.
                break
                
            code.append(best_c)
            code_set.add(best_c)
            uncovered -= neighbors_sets[best_c]
            
        return True

    code_set = set(code_indices)
    
    # ILS Loop
    best_code = list(code_indices)
    best_len = len(best_code)
    
    current_code = list(best_code)
    current_set = set(current_code)
    
    # We need to update current_set when we modify current_code.
    
    for iteration in range(10000): # Upper bound on iterations
        if time.time() > deadline - 5:
            break
            
        # 1. Perturbation: Remove a random center
        if not current_code:
            break
            
        c_to_remove = random.choice(current_code)
        current_code.remove(c_to_remove)
        current_set.remove(c_to_remove)
        
        # 2. Check if still valid
        # Instead of full check, we can just try to repair.
        # If it's valid, repair does nothing.
        # If not, repair adds centers back (maybe different ones).
        
        # To make repair faster, let's check validity first?
        # Validity check is O(Size * 17).
        # Repair is similar.
        
        # Let's just call repair.
        # But repair modifies current_code in place.
        # We need to know if it added anything.
        
        # Actually, a better ILS step:
        # Try to remove c.
        # If valid, keep it removed.
        # If not valid, we must add *some* center back to cover the holes.
        # The greedy repair might add 1 or more centers.
        # If it adds 1 center, the size stays same (removed 1, added 1).
        # If it adds >1, size increases. We should probably reject this move?
        # Or accept it with low probability?
        # Standard ILS: Accept if better or equal. If worse, accept with prob exp(-delta/T).
        # Here, "worse" means larger size.
        
        # Let's implement a simple check:
        # Identify uncovered words after removal.
        # If none, accept removal.
        # If some, find if there is a single center that covers all uncovered words.
        # If yes, swap (remove c, add d). Size unchanged.
        # If no, reject (restore c).
        
        # This "Swap" heuristic is very effective for set cover minimization.
        
        # Identify uncovered
        covered = set()
        for c in current_code:
            covered |= neighbors_sets[c]
            
        uncovered = set(range(total_words)) - covered
        
        if not uncovered:
            # Valid! We successfully removed a center.
            # Keep it removed.
            pass
        else:
            # Try to find a single center that covers all uncovered words
            # The candidate must be in the intersection of neighbors of all uncovered words?
            # Or just scan all centers?
            # Scan all centers is O(N).
            # Find c' such that uncovered is subset of neighbors[c']
            
            found_swap = False
            # Optimization: The candidate must cover all uncovered words.
            # So c' must be in neighbors[w] for all w in uncovered.
            # Intersect neighbors of all uncovered words?
            # If uncovered is large, intersection is small.
            # If uncovered is small (e.g., 1 or 2), intersection is larger.
            
            # Let's just scan all centers.
            for c_prime in range(total_words):
                if c_prime in current_set:
                    continue
                if neighbors_sets[c_prime] >= uncovered:
                    # Found a swap
                    current_code.append(c_prime)
                    current_set.add(c_prime)
                    found_swap = True
                    break
            
            if not found_swap:
                # Restore the removed center
                current_code.append(c_to_remove)
                current_set.add(c_to_remove)
                # Reject this move
                continue
        
        # Update best if improved
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = list(current_code)
            
        # If we hit 486, we can stop? 
        # 486 is the known best. We might not beat it, but we can match it.
        # If we match 486, we are good.
        if best_len <= 486:
            # We can continue to see if we can go lower, but 486 is likely optimal or close.
            # The lower bound is 403.
            # Let's keep running to see if we can beat 486.
            pass

    # Convert best_code to result
    result = [list(idx_to_word[idx]) for idx in best_code]
    
    # Final safety check: ensure distinctness and validity
    # Distinctness is guaranteed by set logic.
    # Validity:
    covered_final = set()
    for w in result:
        idx = word_to_idx[tuple(w)]
        covered_final |= neighbors_sets[idx]
        
    if len(covered_final) != total_words:
        # Fallback to tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return result
