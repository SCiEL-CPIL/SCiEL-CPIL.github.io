"""Local-search minimization for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    total = q ** n

    # Precompute digits for all words
    all_digits = np.zeros((total, n), dtype=np.int8)
    for idx in range(total):
        x = idx
        for i in range(n - 1, -1, -1):
            all_digits[idx, i] = x % q
            x //= q

    # Precompute ball membership: ball[c] = set of words covered by codeword c
    # For radius 1: c itself + words differing in exactly 1 position
    # We'll build a coverage matrix: cover_matrix[w, c] = 1 if w is in ball(c)
    # Size: total x total = 2187 x 2187 ~ 4.7M entries, fine for numpy bool
    
    # Actually, let's use a different representation for speed:
    # For each word w, compute which codewords cover it.
    # But for local search, we need: for a given code C, for each c in C,
    # how many words are ONLY covered by c (i.e., coverage count == 1 and c is the only one).
    
    # Let's build the ball as a list of sets for each codeword.
    # But 2187 sets of ~19 elements each is fine.
    
    # Precompute balls as lists of word indices
    balls = []
    for c in range(total):
        covered = [c]
        digits = all_digits[c].copy()
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                # Convert digits back to int
                w = 0
                for j in range(n - 1, -1, -1):
                    w = w * q + digits[j]
                covered.append(w)
            digits[i] = orig
        balls.append(covered)
    
    # Convert to sets for fast operations
    ball_sets = [set(b) for b in balls]
    
    # Initial code: direct sum of tetracode
    # Tetracode: all (a, b, (a+b)%3, (a+2b)%3) for a,b in {0,1,2}
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    # Initial code: tetracode ⊗ {0,1,2}^{n-4}
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Convert to integer representation
    def digits_to_int(digits, q):
        x = 0
        for i in range(len(digits) - 1, -1, -1):
            x = x * q + digits[i]
        return x
    
    code_ints = [digits_to_int(c, q) for c in initial_code]
    code_set = set(code_ints)
    
    # Verify initial code is valid
    covered_words = set()
    for c in code_ints:
        covered_words.update(ball_sets[c])
    
    if len(covered_words) != total:
        # Fallback: greedy construction if direct sum is invalid (shouldn't happen)
        # But for safety, let's use greedy
        uncovered = set(range(total))
        code_ints = []
        while len(uncovered) > 0:
            if time.time() > deadline:
                break
            best_c = -1
            best_count = 0
            for c in range(total):
                cnt = len(ball_sets[c] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c < 0 or best_count == 0:
                break
            code_ints.append(best_c)
            uncovered.difference_update(ball_sets[best_c])
        
        if len(uncovered) > 0:
            # Return what we have, but this might be invalid
            # For safety, return the direct sum which is always valid
            return initial_code
    
    # Local search: try to remove one codeword and add another to reduce size
    # Actually, we want to REDUCE the number of codewords.
    # Strategy: for each codeword c in code, try removing it.
    # If the remaining code still covers everything, we've reduced size by 1.
    # If not, we need to find a replacement c' such that:
    #   (code - {c} + {c'}) is valid AND |code - {c} + {c'}| < |code|
    # But that's just size same. We want to eventually remove more.
    
    # Better strategy: iterative improvement
    # 1. Try to remove each codeword. If valid, remove it.
    # 2. Repeat until no more removals possible.
    # 3. Then try swaps: remove c, add c', then try to remove more.
    
    # Step 1: Greedy removal
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(list(code_ints)):
            if time.time() > deadline:
                break
            # Try removing c
            new_code = [x for x in code_ints if x != c]
            if len(new_code) == len(code_ints):
                continue  # c not in code (shouldn't happen)
            
            # Check if new_code covers everything
            # Use a count array for efficiency
            counts = np.zeros(total, dtype=np.int32)
            for cw in new_code:
                for w in ball_sets[cw]:
                    counts[w] += 1
            
            if np.all(counts >= 1):
                # Valid! Remove c
                code_ints = new_code
                changed = True
                break  # Restart loop since code changed
        
        if time.time() > deadline:
            break
    
    # Step 2: Try swaps to enable further removals
    # For each pair (c_out, c_in) where c_out in code, c_in not in code:
    #   new_code = (code - {c_out}) ∪ {c_in}
    #   Check if valid. If valid, and if we can then remove more, accept.
    # This is expensive. Let's do a simpler version:
    # For each c_out in code, find the best c_in (not in code) such that
    # (code - {c_out} + {c_in}) is valid. Then accept the swap if it enables
    # subsequent removals.
    
    # Actually, let's just try all swaps and accept any valid swap that
    # doesn't increase size (it keeps size same). Then try removals again.
    # But this might cycle. Let's add a simple heuristic: prefer swaps where
    # the new codeword covers more "critically covered" words.
    
    # Simpler: just try removals repeatedly, then try a few random swaps.
    
    # Let's do a more systematic local search:
    # Maintain code as a list.
    # Repeat:
    #   1. Try to remove each codeword. If valid, remove.
    #   2. For each codeword c in code, try replacing it with each c' not in code.
    #      If the new code is valid, and if the new code allows at least one
    #      additional removal, accept the swap.
    #   3. If no improvements, stop.
    
    # To make this efficient, precompute for each codeword c the set of words
    # that are ONLY covered by c (given the current code).
    
    # Let's implement a simpler version first:
    # Just do repeated removals.
    
    # Actually, the direct sum code for n=7, q=3, radius=1 has size 9 * 3^3 = 243.
    # The best known is 186. So we need to remove about 57 codewords.
    # Simple removals might not be enough. We need swaps.
    
    # Let's try a different approach: use the fact that we can compute
    # coverage counts efficiently.
    
    # Compute coverage counts for current code
    def compute_counts(code):
        counts = np.zeros(total, dtype=np.int32)
        for cw in code:
            for w in ball_sets[cw]:
                counts[w] += 1
        return counts
    
    counts = compute_counts(code_ints)
    
    # A word is "critical" if its count is exactly 1.
    # If we remove a codeword c, the words that become uncovered are those
    # with count 1 that are only covered by c.
    
    # For each codeword c in code, compute the number of words that would
    # become uncovered if c is removed.
    def words_only_covered_by(c, counts, code):
        # Words w in ball(c) such that counts[w] == 1
        # But we need to be careful: if multiple codewords cover w, counts[w] > 1.
        # So words with counts[w] == 1 are only covered by one codeword.
        # If c is the one covering them, removing c uncovers them.
        uncovered_if_removed = 0
        for w in ball_sets[c]:
            if counts[w] == 1:
                uncovered_if_removed += 1
        return uncovered_if_removed
    
    # Try to remove codewords with 0 words_only_covered_by
    # But we need to update counts after each removal.
    
    # Let's do a proper local search with swaps.
    
    # For efficiency, let's use a different representation.
    # Instead of ball_sets, let's precompute for each word w, the list of
    # codewords that cover it.
    
    # cover_list[w] = list of c such that w in ball(c)
    cover_list = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            cover_list[w].append(c)
    
    # Now, for a given code C, the coverage count of w is |{c in C : w in ball(c)}|
    # = |cover_list[w] ∩ C|
    
    # This is still expensive to compute for all w.
    
    # Let's just use the count array approach and update incrementally.
    
    # Current code
    code = list(code_ints)
    counts = compute_counts(code)
    
    # Try removals
    improved = True
    while improved:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            # Check if removing c is valid
            # Words that become uncovered: those w in ball(c) with counts[w] == 1
            # and c is the only codeword covering w.
            # Since counts[w] == 1, and w in ball(c), c is the only one.
            can_remove = True
            for w in ball_sets[c]:
                if counts[w] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                code.pop(i)
                # Update counts
                for w in ball_sets[c]:
                    counts[w] -= 1
                improved = True
                break  # Restart
    
        if time.time() > deadline:
            break
    
    # Now try swaps
    # For each c in code, try replacing with c' not in code.
    # Accept if the new code is valid.
    # To avoid cycles, only accept if the new code has a lower "potential"
    # or if it enables further removals.
    
    # Let's try a simple heuristic: for each c in code, find the c' not in code
    # that maximizes the number of words with counts[w] == 1 that are covered by c'.
    # This might allow subsequent removals.
    
    for iteration in range(50):  # Limit iterations
        if time.time() > deadline:
            break
        
        best_swap = None
        best_score = -1
        
        for i, c in enumerate(code):
            # Words that are critically covered by c (counts[w] == 1 and w in ball(c))
            critical_words = [w for w in ball_sets[c] if counts[w] == 1]
            
            if not critical_words:
                continue
            
            # Try replacing c with c' not in code
            # We want c' to cover as many critical_words as possible
            # But c' must not be in code.
            
            # For efficiency, only try a subset of c'
            # Let's try c' that are close to c in some sense
            # Or just try all c' not in code (expensive but total is 2187)
            
            for c_prime in range(total):
                if c_prime in code_set:
                    continue
                if c_prime == c:
                    continue
                
                # Check if (code - {c} + {c'}) is valid
                # Words that become uncovered: critical_words not covered by c'
                # Words that become over-covered: words in ball(c') with counts[w] >= 1
                #   (they remain covered, no problem)
                # Words that were only covered by c and not by c' become uncovered.
                
                uncovered = 0
                for w in critical_words:
                    if w not in ball_sets[c_prime]:
                        uncovered += 1
                
                if uncovered == 0:
                    # Valid swap!
                    # Score: number of critical_words covered by c'
                    # (higher is better, as it might enable more removals)
                    score = len(critical_words)
                    if score > best_score:
                        best_score = score
                        best_swap = (i, c, c_prime)
            
            if time.time() > deadline:
                break
        
        if best_swap is not None:
            i, c_out, c_in = best_swap
            # Perform swap
            code[i] = c_in
            code_set.discard(c_out)
            code_set.add(c_in)
            
            # Update counts
            for w in ball_sets[c_out]:
                counts[w] -= 1
            for w in ball_sets[c_in]:
                counts[w] += 1
            
            # Try removals again
            improved = True
            while improved:
                improved = False
                for j in range(len(code)):
                    if time.time() > deadline:
                        break
                    c = code[j]
                    can_remove = True
                    for w in ball_sets[c]:
                        if counts[w] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        code.pop(j)
                        code_set.discard(c)
                        for w in ball_sets[c]:
                            counts[w] -= 1
                        improved = True
                        break
                if time.time() > deadline:
                    break
        else:
            # No valid swap found
            break
    
    # Convert back to list of lists
    result = [all_digits[c].tolist() for c in code]
    
    # Verify validity
    final_covered = set()
    for c in code:
        final_covered.update(ball_sets[c])
    
    if len(final_covered) != total:
        # Fallback to initial code
        return initial_code
    
    return result

# EVOLVE-BLOCK-END
