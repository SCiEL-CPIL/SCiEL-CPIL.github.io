import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases (not expected in this specific task instance but good for robustness)
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    best_size = total
    best_set = None
    
    def greedy_construct(seed_offset=0, rng=None):
        """Vectorized greedy set cover - hyperedge coloring formulation.
        
        Vertices = 6561 words, hyperedges = 17-word radius-1 balls.
        Select hyperedges to minimize count while covering all vertices.
        """
        if rng is None:
            rng = np.random.RandomState(seed_offset)
        
        uncovered_mask = np.ones(total, dtype=np.int32)  # 1 if uncovered
        selected = []
        
        while np.any(uncovered_mask):
            # Vectorized: count uncovered vertices in each ball
            # uncovered_mask[balls] has shape (total, ball_size), sum over axis=1
            counts = uncovered_mask[balls].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Random tie-breaking among candidates with max count
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update: mark words in this ball as covered
            ball_words = balls[best_idx]
            uncovered_mask[ball_words] = 0
            
            if time.time() > deadline - 8.0:
                return None  # timeout
        
        return selected
    
    def compute_cover_count(selected):
        """Compute coverage count for each word given selected balls."""
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        return cover_count
    
    def prune_set(selected, cover_count):
        """Remove redundant codewords - vectorized where possible."""
        changed = True
        while changed and time.time() < deadline - 4.0:
            changed = False
            new_selected = []
            for c in selected:
                # If all words in ball(c) are covered by at least 2 codewords, c is redundant
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count
    
    def local_search_swap(selected, cover_count, max_rounds=30):
        """Try 1-for-1 swaps and 2-for-1 swaps to reduce size.
        
        1-for-1: Remove c, if critical set U is covered by some c', swap.
        2-for-1: Remove c1, c2, if critical set U is covered by some c', replace with c'.
        """
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5.0:
            improved = False
            rounds += 1
            
            # --- 2-for-1 Swaps ---
            # Try to replace two codewords with one
            # This is more expensive, so we might limit it or do it less frequently
            # We'll iterate through pairs
            n_sel = len(selected)
            for i in range(n_sel):
                if time.time() > deadline - 5.0:
                    break
                c1 = selected[i]
                ball_c1 = balls[c1]
                
                for j in range(i + 1, n_sel):
                    if time.time() > deadline - 5.0:
                        break
                    c2 = selected[j]
                    ball_c2 = balls[c2]
                    
                    # Words that become uncovered if we remove c1 and c2
                    # A word w is uncovered if cover_count[w] - (w in ball_c1) - (w in ball_c2) == 0
                    # Let's compute the set of words in ball_c1 U ball_c2
                    # We need to check their coverage counts.
                    
                    # Efficient way: 
                    # Create a temporary array of counts? No, too slow.
                    # Just check the words in ball_c1 and ball_c2.
                    
                    # Get unique words in ball_c1 and ball_c2
                    # Since ball size is 17, we can just loop through them.
                    critical_words = []
                    
                    # Check words in ball_c1
                    for w in ball_c1:
                        # Count how many of {c1, c2} cover w
                        # We know c1 covers w. Does c2 cover w?
                        # We can check if w is in ball_c2. 
                        # To speed up, we can precompute a set of ball_c2? 
                        # Or just use np.isin? 
                        # Given small size, a set lookup is fast.
                        # But we are in a tight loop. 
                        # Let's use a boolean array for ball_c2 membership? 
                        # That's 6561 size. Initializing it for every pair is too slow.
                        
                        # Alternative: 
                        # We can precompute for each word w, the list of selected codewords that cover it?
                        # That's what cover_count is, but we need the specific IDs.
                        
                        # Let's stick to the simple check: 
                        # is w in ball_c2?
                        # We can use np.where(ball_c2 == w)[0]
                        # This is O(17) per check.
                        
                        if w in ball_c2:
                            # Covered by both
                            # New count = cover_count[w] - 2
                            if cover_count[w] == 2:
                                critical_words.append(w)
                        else:
                            # Covered by c1 only (among the two)
                            # New count = cover_count[w] - 1
                            if cover_count[w] == 1:
                                critical_words.append(w)
                    
                    # Check words in ball_c2 that are NOT in ball_c1
                    # We already checked w in ball_c1 above.
                    # We need to check w in ball_c2 where w NOT in ball_c1.
                    # To avoid duplicates, we can just check all w in ball_c2 and skip if w in ball_c1?
                    # Or just add all w in ball_c2 and deduplicate?
                    # Let's just check w in ball_c2.
                    for w in ball_c2:
                        if w in ball_c1:
                            continue # Already processed
                        # Covered by c2 only (among the two)
                        # New count = cover_count[w] - 1
                        if cover_count[w] == 1:
                            critical_words.append(w)
                    
                    if not critical_words:
                        # Both are redundant?
                        # Remove c1 and c2
                        cover_count[ball_c1] -= 1
                        cover_count[ball_c2] -= 1
                        # Remove from selected
                        # Need to handle indices carefully
                        # We are iterating i, j. If we remove, indices shift.
                        # It's safer to collect moves and apply them after the loop,
                        # or mark for removal.
                        # Let's just mark and break? 
                        # If we remove, the loop indices are invalid.
                        # Let's just do a simple removal if we find a 2-for-1 or 1-for-1.
                        
                        # Actually, if critical_words is empty, it means all words in B(c1) U B(c2) are covered by other codewords.
                        # So we can remove both.
                        # Let's perform the move immediately and restart the loop (or break out).
                        # To keep it simple, we'll set a flag and break.
                        
                        # Update cover counts
                        cover_count[ball_c1] -= 1
                        cover_count[ball_c2] -= 1
                        # Remove from selected
                        # We need to remove c1 and c2 from selected.
                        # Since we are in a nested loop, let's just break and let the outer while loop restart.
                        # But we need to update selected.
                        
                        # Let's create a new selected list
                        new_selected = [c for k, c in enumerate(selected) if k != i and k != j]
                        selected = new_selected
                        improved = True
                        break # Break inner loop
                    else:
                        # Check if any single codeword c_k covers all critical_words
                        # If yes, replace c1, c2 with c_k
                        
                        # To find c_k, we can check all 6561 codewords? 
                        # That's 6561 * 17 checks = ~100k ops per pair.
                        # With ~100k pairs, that's 10^10 ops. Too slow.
                        
                        # Optimization: Only check c_k that cover the first critical word.
                        # Let w0 = critical_words[0]
                        # Find all c_k such that w0 in balls[c_k]
                        # There are 17 such c_k (the ball centered at w0 and its neighbors? No, the balls that contain w0).
                        # The balls containing w0 are:
                        # 1. The ball centered at w0.
                        # 2. Balls centered at neighbors of w0 (distance 1).
                        # So there are 1 + 2*8 = 17 such balls.
                        
                        w0 = critical_words[0]
                        # Find centers c_k such that w0 in balls[c_k]
                        # This is equivalent to: c_k is w0 or c_k is at distance 1 from w0.
                        # We can generate these 17 candidates.
                        
                        # Generate candidates
                        candidates_k = [w0]
                        # Add neighbors of w0
                        # We can reuse the ball generation logic or just use the precomputed balls?
                        # balls[w0] gives the words in the ball centered at w0.
                        # We need the centers of balls that contain w0.
                        # If w0 in balls[c_k], then c_k is in "balls containing w0".
                        # This is the same set as: {w0} U {neighbors of w0}.
                        
                        # Let's compute neighbors of w0
                        # We can just use the fact that if w0 in balls[c_k], then dist(w0, c_k) <= 1.
                        # So c_k is w0 or dist(w0, c_k) == 1.
                        
                        # We can iterate through all 6561 words and check distance? 
                        # Or just generate the 17 candidates directly.
                        
                        # Generate 17 candidates for w0
                        # Center w0
                        # Neighbors: change one digit
                        for pos in range(n):
                            weight = weights[pos]
                            digit = (w0 // weight) % 3
                            for new_digit in range(q):
                                if new_digit == digit:
                                    continue
                                new_w = w0 + (new_digit - digit) * weight
                                candidates_k.append(new_w)
                        
                        # Now check each candidate c_k
                        found = False
                        for c_k in candidates_k:
                            if c_k == c1 or c_k == c2:
                                continue
                            
                            # Check if c_k covers all critical_words
                            # c_k covers w if w in balls[c_k]
                            # We can check if all w in critical_words are in balls[c_k]
                            # balls[c_k] has 17 words.
                            # We can create a set of balls[c_k] and check membership?
                            # Or just loop through critical_words and check if they are in balls[c_k].
                            # critical_words size is small (<= 34).
                            # balls[c_k] size is 17.
                            
                            # Let's convert balls[c_k] to a set for O(1) lookup?
                            # Creating a set for each candidate might be slow.
                            # Let's just use np.isin or manual loop.
                            
                            # Manual loop is probably fast enough for small sizes
                            covers_all = True
                            for w in critical_words:
                                if not (w in balls[c_k]): # 'in' on numpy array is O(17)
                                    covers_all = False
                                    break
                            
                            if covers_all:
                                # Perform swap: remove c1, c2, add c_k
                                cover_count[ball_c1] -= 1
                                cover_count[ball_c2] -= 1
                                cover_count[balls[c_k]] += 1
                                
                                # Update selected
                                new_selected = [c for k, c in enumerate(selected) if k != i and k != j]
                                new_selected.append(c_k)
                                selected = new_selected
                                improved = True
                                found = True
                                break
                            
                        if found:
                            break # Break inner loop
                
                if improved:
                    break # Break outer loop for i
            
            if improved:
                continue # Restart rounds loop
            
            # --- 1-for-1 Swaps (if 2-for-1 didn't find anything) ---
            # Similar to before, but now we can handle multiple critical words
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                
                c = selected[i]
                ball_c = balls[c]
                
                # Find critical words: covered only by c
                critical = ball_c[cover_count[ball_c] == 1]
                
                if len(critical) == 0:
                    # c is redundant
                    cover_count[ball_c] -= 1
                    selected.pop(i)
                    improved = True
                    break
                elif len(critical) <= 3: # Limit to small sets for efficiency
                    # Try to find a replacement ball that covers all critical words
                    # Use the same candidate generation as above
                    w0 = critical[0]
                    candidates_k = [w0]
                    for pos in range(n):
                        weight = weights[pos]
                        digit = (w0 // weight) % 3
                        for new_digit in range(q):
                            if new_digit == digit:
                                continue
                            new_w = w0 + (new_digit - digit) * weight
                            candidates_k.append(new_w)
                    
                    found = False
                    for c_k in candidates_k:
                        if c_k == c:
                            continue
                        
                        covers_all = True
                        for w in critical:
                            if not (w in balls[c_k]):
                                covers_all = False
                                break
                        
                        if covers_all:
                            # Perform swap
                            cover_count[ball_c] -= 1
                            cover_count[balls[c_k]] += 1
                            selected[i] = c_k
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break
            
        return selected, cover_count
    
    # Multiple restarts with different seeds
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        rng_seed = 42 + num_restarts * 1000
        rng = np.random.RandomState(rng_seed)
        
        selected = greedy_construct(seed_offset=rng_seed, rng=rng)
        if selected is None:
            continue
        
        # Compute cover counts for pruning
        cover_count = compute_cover_count(selected)
        
        # Prune redundant codewords
        selected, cover_count = prune_set(selected, cover_count)
        
        # Local search: try swaps to reduce size
        selected, cover_count = local_search_swap(selected, cover_count, max_rounds=30)
        
        # Additional pruning after swaps
        selected, cover_count = prune_set(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
    
    # Final polish: try to find a smaller solution from the best found
    if best_set is not None and time.time() < deadline - 3.0:
        cover_count = compute_cover_count(best_set)
        
        # Aggressive pruning
        for _ in range(5):
            changed = False
            new_best = []
            for c in best_set:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_best.append(c)
            best_set = new_best
            if not changed:
                break
        
        # Try swaps on the best solution
        best_set, cover_count = local_search_swap(best_set, cover_count, max_rounds=50)
        
        # Final pruning
        for _ in range(3):
            changed = False
            new_best = []
            for c in best_set:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_best.append(c)
            best_set = new_best
            if not changed:
                break
        
        best_size = len(best_set)
    
    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback: return all words
        result = []
        for w in range(total):
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            result.append(digits)
        return result
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
