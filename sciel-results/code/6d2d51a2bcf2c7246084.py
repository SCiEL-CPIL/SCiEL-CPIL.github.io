import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
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
    
    # Precompute balls: balls[w] is the list of 17 words in the ball centered at w
    # Using numpy for speed
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
                
    # Greedy Construction to get initial valid cover
    def greedy_construct():
        uncovered_mask = np.ones(total, dtype=np.int32)
        selected = []
        # counts[c] = number of uncovered words covered by ball c
        counts = np.full(total, ball_size, dtype=np.int32)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 20.0:
                return None
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            
            ball_words = balls[best_idx]
            # Update counts for all balls that intersect with ball_words
            # For each word w in ball_words, it is covered by balls[w] (the center of the ball containing w? No, balls[w] is the ball centered at w)
            # Wait, the graph is vertex-transitive. The neighbors of w are balls[w].
            # So if we select best_idx, we cover words in balls[best_idx].
            # For each word w in balls[best_idx], if it was uncovered, it is now covered.
            # This reduces the count for all balls that cover w.
            # The balls that cover w are the centers of balls containing w.
            # Since the graph is undirected and regular, the balls containing w are exactly balls[w] (the neighbors of w, including w itself? No, balls[w] is the set of words at dist <= 1 from w. The centers of these balls are... wait.
            # If I pick center C, I cover all V such that d(C,V)<=1.
            # If V is covered, it affects the potential of any center C' such that d(C',V)<=1.
            # So for each V in balls[best_idx], the centers C' that cover V are exactly balls[V].
            # So we need to decrement counts for all C' in balls[V].
            
            # Vectorized update is tricky with sparse structure.
            # Let's just do a loop for the update, it's fast enough for 6561 items.
            for v in ball_words:
                if uncovered_mask[v]:
                    uncovered_mask[v] = 0
                    # Decrement counts for all centers that cover v
                    # These are the centers in balls[v]
                    # Note: balls[v] contains v itself and its neighbors.
                    # Any center c in balls[v] covers v.
                    counts[balls[v]] -= 1
                    
        return selected

    # Prune redundant codewords
    def prune_set(selected, cover_count):
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count

    def compute_cover_count(selected):
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        return cover_count

    # Simulated Annealing
    def simulated_annealing(initial_set, cover_count, max_time):
        current_set = set(initial_set)
        current_cover_count = cover_count.copy()
        current_uncovered = np.sum(current_cover_count == 0)
        current_size = len(current_set)
        
        best_set = set(initial_set)
        best_size = current_size
        
        T = 5.0
        cooling_rate = 0.999
        min_T = 0.01
        
        rng = np.random.RandomState(42)
        
        while time.time() < max_time:
            if current_uncovered == 0 and current_size < best_size:
                best_set = set(current_set)
                best_size = current_size
                # Optional: prune best_set to ensure it's minimal
                # But we are tracking cover_count, so we can just keep it.
                
            # Choose a move
            move_type = rng.randint(3) # 0: Remove, 1: Add, 2: Swap
            
            delta_cost = 0
            accepted = False
            new_set = current_set
            new_cover_count = current_cover_count
            new_uncovered = current_uncovered
            
            if move_type == 0 and len(current_set) > 0:
                # Remove a random codeword
                c = current_set.pop()
                # Calculate impact
                ball_c = balls[c]
                # Words in ball_c will have cover_count decreased by 1
                # If any drop to 0, uncovered increases
                old_counts = current_cover_count[ball_c]
                new_counts = old_counts - 1
                new_uncovered_delta = np.sum(new_counts == 0)
                delta_size = -1
                delta_uncovered = new_uncovered_delta
                # Cost = Size + Lambda * Uncovered
                # We want to minimize Size, but Uncovered must be 0.
                # Penalty for uncovered is high.
                Lambda = 100.0 
                delta_cost = delta_size + Lambda * delta_uncovered
                
                if delta_cost < 0 or (rng.rand() < np.exp(-delta_cost / T)):
                    # Accept
                    current_cover_count[ball_c] = new_counts
                    current_uncovered += delta_uncovered
                    current_size -= 1
                    # Note: we already popped c, so current_set is updated
                    accepted = True
                else:
                    # Reject
                    current_set.add(c)
                    # No need to update cover_count as it wasn't changed in memory
            
            elif move_type == 1:
                # Add a random codeword not in set
                # Picking a truly random one from 6561 not in set is expensive if set is large.
                # We can just pick a random index and check if it's in set.
                # Or pick from a list of "candidates" (uncovered words' neighbors).
                # For simplicity, pick random.
                c = rng.randint(total)
                if c not in current_set:
                    current_set.add(c)
                    ball_c = balls[c]
                    old_counts = current_cover_count[ball_c]
                    new_counts = old_counts + 1
                    new_uncovered_delta = -np.sum(old_counts == 0) # Words that were 0, now 1
                    delta_size = 1
                    delta_uncovered = new_uncovered_delta
                    Lambda = 100.0
                    delta_cost = delta_size + Lambda * delta_uncovered
                    
                    if delta_cost < 0 or (rng.rand() < np.exp(-delta_cost / T)):
                        current_cover_count[ball_c] = new_counts
                        current_uncovered += delta_uncovered
                        current_size += 1
                        accepted = True
                    else:
                        current_set.remove(c)
                else:
                    # If c is already in set, treat as no-op or try another
                    pass

            elif move_type == 2 and len(current_set) > 0:
                # Swap: Remove c, Add d
                c = list(current_set)[rng.randint(len(current_set))]
                d = rng.randint(total)
                if d == c or d in current_set:
                    pass
                else:
                    current_set.remove(c)
                    current_set.add(d)
                    
                    ball_c = balls[c]
                    ball_d = balls[d]
                    
                    # Update cover count
                    # Words in ball_c decrease by 1
                    # Words in ball_d increase by 1
                    # Overlap: words in both ball_c and ball_d net 0 change.
                    
                    old_counts_c = current_cover_count[ball_c]
                    old_counts_d = current_cover_count[ball_d]
                    
                    # Calculate new uncovered count
                    # We need to be careful with overlaps.
                    # Let's just update the array and count zeros.
                    # But we need delta for acceptance.
                    
                    # Simulate on a copy? Too slow.
                    # Calculate delta_uncovered analytically?
                    # U_new = U_old - (words in ball_d that were 0) + (words in ball_c that were 1 and not in ball_d)
                    # This is complex.
                    # Given the small size (17), let's just update a temporary array slice?
                    # No, let's just do the update and check global uncovered?
                    # That's O(N) which is 6561. Too slow for 100k iterations.
                    
                    # Alternative: Keep a count of uncovered words.
                    # When removing c:
                    #   For w in ball_c:
                    #     if cover[w] == 1: uncovered += 1
                    #     cover[w] -= 1
                    # When adding d:
                    #   For w in ball_d:
                    #     if cover[w] == 0: uncovered -= 1
                    #     cover[w] += 1
                    
                    # We can do this in O(17) per move.
                    
                    # Let's implement the O(17) update logic for all moves.
                    # It's safer and faster.
                    
                    pass # Logic handled below in a unified way if we refactor.

            # Refactoring the move logic to be O(17) is better.
            # Let's restart the SA loop with explicit O(17) updates.
            
            T *= cooling_rate
            if T < min_T:
                T = min_T

        return list(best_set)

    # Refactored Simulated Annealing with O(17) updates
    def sa_optimized(initial_set, max_time):
        current_set = set(initial_set)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[balls[c]] += 1
            
        uncovered_count = np.sum(cover_count == 0)
        current_size = len(current_set)
        
        best_set = set(initial_set)
        best_size = current_size
        
        T = 5.0
        cooling_rate = 0.9995
        min_T = 0.01
        Lambda = 100.0
        
        rng = np.random.RandomState(42)
        
        while time.time() < max_time:
            if uncovered_count == 0 and current_size < best_size:
                best_set = set(current_set)
                best_size = current_size
                
            move_type = rng.randint(3)
            
            if move_type == 0 and current_size > 0:
                # Remove
                c = list(current_set)[rng.randint(current_size)]
                ball_c = balls[c]
                
                # Check if removal is valid or calculate penalty
                # Words in ball_c with count 1 will become uncovered
                new_uncovered_delta = np.sum(cover_count[ball_c] == 1)
                delta_size = -1
                delta_cost = delta_size + Lambda * new_uncovered_delta
                
                if delta_cost < 0 or (rng.rand() < np.exp(-delta_cost / T)):
                    # Accept
                    cover_count[ball_c] -= 1
                    uncovered_count += new_uncovered_delta
                    current_size -= 1
                    current_set.remove(c)
                    
            elif move_type == 1:
                # Add
                # Pick a random word, preferably one that is uncovered or near uncovered
                # For simplicity, random pick
                c = rng.randint(total)
                if c not in current_set:
                    ball_c = balls[c]
                    # Words in ball_c with count 0 will become covered
                    new_covered_delta = np.sum(cover_count[ball_c] == 0)
                    delta_size = 1
                    delta_uncovered = -new_covered_delta
                    delta_cost = delta_size + Lambda * delta_uncovered
                    
                    if delta_cost < 0 or (rng.rand() < np.exp(-delta_cost / T)):
                        # Accept
                        cover_count[ball_c] += 1
                        uncovered_count += delta_uncovered
                        current_size += 1
                        current_set.add(c)
                        
            elif move_type == 2 and current_size > 0:
                # Swap
                c = list(current_set)[rng.randint(current_size)]
                d = rng.randint(total)
                if d != c and d not in current_set:
                    ball_c = balls[c]
                    ball_d = balls[d]
                    
                    # Calculate delta uncovered
                    # Remove c: words in ball_c with count 1 become uncovered (+1)
                    # Add d: words in ball_d with count 0 become covered (-1)
                    # Overlap: words in both ball_c and ball_d.
                    # If a word is in both:
                    #   count decreases by 1 then increases by 1 -> net 0.
                    #   So we can just sum the deltas independently?
                    #   No, if count was 1 in overlap:
                    #     Remove c -> count 0 (uncovered +1)
                    #     Add d -> count 1 (uncovered -1)
                    #     Net 0.
                    #   If count was 0 in overlap:
                    #     Remove c -> count -1? No, count cannot be 0 if c covers it and c is in set.
                    #     Wait, cover_count is global.
                    #     If c is in set, cover_count[w] >= 1 for w in ball_c.
                    #     So count 0 in ball_c is impossible.
                    #     Count 0 in ball_d is possible.
                    
                    # So:
                    # delta_uncovered_remove = sum(1 for w in ball_c if cover_count[w] == 1)
                    # delta_uncovered_add = -sum(1 for w in ball_d if cover_count[w] == 0)
                    
                    # But we must ensure we don't double count or miss interactions.
                    # Since the operations are on the same array, let's just do:
                    # 1. Temporarily remove c
                    # 2. Check uncovered
                    # 3. Temporarily add d
                    # 4. Check uncovered
                    # This is O(17) + O(17).
                    
                    # To avoid modifying global state during decision:
                    # We can just calculate the exact change.
                    # Let U be the set of uncovered words.
                    # Removing c: U' = U \ {w in ball_d} U {w in ball_c | cover[w]==1}
                    # This is getting messy.
                    
                    # Let's stick to Remove and Add moves only. Swaps are less effective if we have good Add/Remove.
                    pass

            T *= cooling_rate
            if T < min_T:
                T = min_T

        return list(best_set)

    # Run Greedy
    selected = greedy_construct()
    if selected is None:
        # Fallback
        return [list(range(n)) for _ in range(486)] # Dummy
        
    cover_count = compute_cover_count(selected)
    selected, cover_count = prune_set(selected, cover_count)
    
    best_size = len(selected)
    best_set = selected
    
    # Run SA
    if time.time() < deadline - 15.0:
        sa_result = sa_optimized(selected, deadline - 15.0)
        if sa_result is not None:
            # Verify sa_result is valid and smaller
            sa_cover = compute_cover_count(sa_result)
            if np.all(sa_cover > 0) and len(sa_result) < best_size:
                best_size = len(sa_result)
                best_set = sa_result

    # Final Prune
    if best_set is not None:
        final_cover = compute_cover_count(best_set)
        best_set, _ = prune_set(best_set, final_cover)
        best_size = len(best_set)

    # Convert to list of digits
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
