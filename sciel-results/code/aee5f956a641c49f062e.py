import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s, but task says 120s. Use 110s to be safe.
    deadline = start_time + 110
    
    # 1. Precomputation
    # Map words to integers 0..q^n-1
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    total_words = q ** n
    all_words = list(range(total_words))
    
    # Precompute balls (radius-1 neighborhoods) as sets of integers
    # For q=3, n=6, radius=1, ball size is 1 + n*(q-1) = 1 + 6*2 = 13
    balls = {}
    for w in all_words:
        b = {w}
        w_list = int_to_list(w, n)
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.add(list_to_int(new_word))
        balls[w] = b

    # Precompute reverse mapping: for each word, which codewords cover it?
    # This is the same as balls, but we might need it for "unique coverer" logic.
    # Actually, for efficiency, we'll maintain a coverage count array dynamically.
    
    # 2. Initial Construction: Direct Sum of Tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
            
    code = set(initial_codewords)
    
    # Helper: Compute coverage counts and unique coverers
    # Returns: 
    #   coverage: dict word -> count
    #   unique_coverer: dict word -> codeword (or None if >1)
    def compute_coverage():
        coverage = {w: 0 for w in all_words}
        unique_coverer = {w: None for w in all_words}
        
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                elif coverage[w] > 1:
                    unique_coverer[w] = None
        
        return coverage, unique_coverer

    # Helper: Check if code covers all words
    def is_valid():
        covered = set()
        for c in code:
            covered |= balls[c]
        return len(covered) == total_words

    # 3. Greedy Removal Phase
    # Remove any codeword that is not the unique coverer of any word
    def greedy_remove():
        nonlocal code
        if time.time() > deadline:
            return False
        
        coverage, unique_coverer = compute_coverage()
        
        essential = set()
        for w in all_words:
            if unique_coverer[w] is not None:
                essential.add(unique_coverer[w])
                
        removable = code - essential
        if removable:
            code -= removable
            return True
        return False

    # Run greedy removal until stable
    while greedy_remove():
        pass

    # 4. Local Search: Swap and Repair
    # We want to find a codeword c that can be removed, and potentially replaced by c'
    # such that coverage is maintained and size decreases or stays same.
    
    def try_swap_or_remove():
        nonlocal code
        if time.time() > deadline:
            return False
            
        coverage, unique_coverer = compute_coverage()
        
        # Identify codewords that have unique words they cover
        # If we remove c, all words in U_c become uncovered.
        # We need to find a new codeword c' that covers all of U_c.
        
        unique_groups = {}
        for w in all_words:
            uc = unique_coverer[w]
            if uc is not None:
                if uc not in unique_groups:
                    unique_groups[uc] = []
                unique_groups[uc].append(w)
        
        # Sort candidates by size of unique group (smaller is easier to cover)
        candidates = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        for c, U_c in candidates:
            if not U_c:
                continue
                
            # If U_c is empty, c is redundant (should have been removed by greedy)
            # But if U_c is not empty, we must cover these words if we remove c.
            
            # Try to find a codeword c' (not in code) that covers all words in U_c
            # To be efficient, we only need to check c' that are "close" to U_c
            # Actually, any c' whose ball intersects U_c is a candidate.
            # But we need B(c') to cover ALL of U_c.
            
            # Optimization: 
            # 1. If we just remove c, uncovered = U_c.
            # 2. We can add a new c'.
            # Net change: -1 (remove c) + 1 (add c') = 0.
            # We only accept if we can remove c and NOT add c' (i.e., U_c is empty, handled by greedy)
            # OR if we remove c and add c' but c' covers more than just U_c, effectively "cleaning up" redundancy elsewhere?
            # No, size stays same. We want size to decrease.
            
            # Strategy: 
            # 1. Try to remove c without replacement. (Fails if U_c not empty)
            # 2. Try to swap c with c'. If we do this, size is same. 
            #    Why swap? To reach a state where another codeword becomes removable.
            #    This is a hill-climbing move on the "removability" landscape.
            
            # Let's try to find c' such that B(c') covers U_c.
            # We don't need to check all 729 words. 
            # The intersection of balls of all words in U_c is the set of valid c'.
            # valid_cs = intersection of balls[w] for w in U_c
            # If c is in valid_cs, it's a no-op.
            # We want c' not in code.
            
            valid_cs = set(balls[U_c[0]])
            for w in U_c[1:]:
                valid_cs &= balls[w]
                
            # Exclude current code
            candidate_new = valid_cs - code
            
            if not candidate_new:
                continue
                
            # Pick a random candidate to swap with
            # Or pick one that maximizes "potential" (e.g., covers many other unique words of other codewords?)
            # For now, random selection to explore.
            c_new = random.choice(list(candidate_new))
            
            # Perform swap
            code.remove(c)
            code.add(c_new)
            
            # Immediately try to remove redundant words
            greedy_remove()
            
            # If we made progress (size decreased), return True
            # If size stayed same, we might still want to continue to find a decrease.
            # But to avoid infinite loops of swaps, we should track best size.
            # For this simple version, we return True to signal a move was made.
            return True
            
        return False

    def try_remove_and_repair():
        nonlocal code
        if time.time() > deadline:
            return False
            
        # This is more aggressive.
        # Pick a random codeword c.
        # Remove it.
        # Find all uncovered words U.
        # Find a single codeword c' that covers ALL of U.
        # If found, add c'. Size decreases by 1.
        
        if not code:
            return False
            
        c = random.choice(list(code))
        code.remove(c)
        
        # Calculate uncovered
        covered = set()
        for c2 in code:
            covered |= balls[c2]
        uncovered = set(all_words) - covered
        
        if not uncovered:
            # c was redundant
            return True
            
        # Find c' that covers all uncovered
        # valid_cs = intersection of balls[w] for w in uncovered
        valid_cs = set(balls[list(uncovered)[0]])
        for w in list(uncovered)[1:]:
            valid_cs &= balls[w]
            
        candidate_new = valid_cs - code
        
        if candidate_new:
            c_new = random.choice(list(candidate_new))
            code.add(c_new)
            # Clean up any new redundancies
            greedy_remove()
            return True
        else:
            # Couldn't repair, put c back
            code.add(c)
            return False

    # Main Loop
    best_size = len(code)
    
    # Initial polish
    greedy_remove()
    best_size = len(code)
    
    while time.time() < deadline:
        made_progress = False
        
        # 1. Try to remove and repair (best for size reduction)
        if try_remove_and_repair():
            if len(code) < best_size:
                best_size = len(code)
                # Reset random seed or just continue? 
                # We made a significant improvement.
            made_progress = True
            continue
        
        # 2. Try swap to enable future removals
        if try_swap_or_remove():
            made_progress = True
            # Don't update best_size here unless it actually decreased
            if len(code) < best_size:
                best_size = len(code)
            continue
            
        # 3. If stuck, perturb
        if random.random() < 0.1:
            # Remove 1-2 random codewords
            n_remove = random.randint(1, 2)
            to_remove = random.sample(list(code), min(n_remove, len(code)))
            for c in to_remove:
                code.remove(c)
                
            # Greedy repair
            for _ in range(5):
                if time.time() > deadline:
                    break
                covered = set()
                for c in code:
                    covered |= balls[c]
                uncovered = set(all_words) - covered
                if not uncovered:
                    break
                    
                # Find best codeword to add
                best_c = None
                best_count = 0
                # Sample a subset of candidates to speed up
                candidates = random.sample(all_words, min(100, total_words))
                for w in candidates:
                    if w in code:
                        continue
                    count = len(balls[w] & uncovered)
                    if count > best_count:
                        best_count = count
                        best_c = w
                        
                if best_c is not None and best_count > 0:
                    code.add(best_c)
                else:
                    break
            
            # Polish
            greedy_remove()
            
            if len(code) < best_size:
                best_size = len(code)
            made_progress = True
            continue
            
        if not made_progress:
            # No progress, try a few more swaps before breaking?
            # Or break if we've been stuck for a while?
            # For simplicity, break if no progress in a batch.
            break

    # Final verification and cleanup
    greedy_remove()
    
    result = [int_to_list(c, n) for c in code]
    return result
