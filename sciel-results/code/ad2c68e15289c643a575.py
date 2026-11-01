import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we stop at 140s to ensure return
    deadline = start_time + 140.0
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # 1. Precompute balls as lists of indices
    balls = []
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for x in range(total):
        ball = [x]
        for i in range(n):
            old_digit = (x // powers[i]) % q
            for new_digit in range(q):
                if new_digit != old_digit:
                    y = x + (new_digit - old_digit) * powers[i]
                    ball.append(y)
        balls.append(ball)
        
    # 2. Precompute inverse mapping: for each word y, which balls cover it?
    # inv[y] is a list of ball indices
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    # 3. Greedy Construction
    # We maintain a count of how many times each word is covered.
    # Initially 0.
    cover_counts = np.zeros(total, dtype=np.int32)
    selected = []
    
    # To speed up greedy, we can use a heap or just scan. 
    # Scanning 6561 items is fast. 
    # We want to pick the ball that covers the most UNCOVERED words (cover_counts == 0).
    
    # Optimization: Instead of checking all balls, we can just pick a random uncovered word 
    # and choose the best ball among its neighbors? 
    # Actually, standard greedy is "pick ball with max uncovered coverage".
    # With 6561 balls and 6561 words, one pass is 43M ops. In Python this is ~2-5s.
    # We can do this a few times or just once.
    
    # Let's do a fast greedy.
    # To avoid O(N^2) total (N greedy steps * N balls), we can stop adding balls 
    # once all are covered. But we don't know when.
    # However, the number of balls needed is ~400-600.
    # 600 * 6561 * 17 (avg ball size) ~ 60M ops. This might be tight in Python.
    
    # Alternative Greedy: 
    # 1. Pick a random uncovered word.
    # 2. Look at all balls covering it (at most 17).
    # 3. Pick the one that covers the most *other* uncovered words.
    # 4. Add to selected, update counts.
    # 5. Repeat until all covered.
    # This is much faster: O(Steps * 17 * BallSize). Steps ~ 500. 500 * 17 * 17 ~ 150k ops. Very fast.
    
    uncovered_set = set(range(total))
    while uncovered_set and time.time() < deadline - 10:
        # Pick a random uncovered word
        target = random.choice(list(uncovered_set))
        
        # Get candidates: balls covering target
        candidates = inv[target]
        
        best_cand = -1
        best_gain = -1
        
        for c in candidates:
            # Count how many words in balls[c] are currently uncovered
            gain = 0
            for y in balls[c]:
                if y in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cand = c
                
        if best_cand == -1:
            break
            
        selected.append(best_cand)
        
        # Update uncovered_set and cover_counts
        for y in balls[best_cand]:
            cover_counts[y] += 1
            if cover_counts[y] == 1:
                uncovered_set.remove(y)
                
    # 4. Local Search: 1-Swap
    # Try to remove a selected ball and add another ball to maintain coverage while reducing size?
    # No, 1-swap means: Remove one selected ball, Add one different ball. 
    # This keeps size constant. It's for improving "quality" (e.g., redundancy distribution) 
    # which might allow a subsequent removal?
    # Actually, the standard local search for Set Cover minimization is:
    # 1. Try to remove a ball. If valid (no word drops to 0 coverage), do it.
    # 2. Try to swap a ball A with ball B (A in set, B not in set) such that 
    #    removing A and adding B results in a valid cover, AND the new set has 
    #    "better" properties (e.g., more redundancy, or just random walk to escape local minima).
    #    
    # But to REDUCE size, we primarily need "Removal". 
    # Why would removal fail? If every ball in the set covers at least one unique word.
    # Local search usually does:
    # - Randomly swap A (in) with B (out) to explore the space of same-size covers.
    # - After each swap (or periodically), try to remove any ball that is now redundant.
    
    # Let's implement this.
    
    # Precompute bitsets for fast coverage checks? 
    # With 6561 words, a Python list of ints (bitsets) is efficient.
    num_words = (total + 63) // 64
    zero_mask = 0
    
    # Convert selected to a set for O(1) lookup
    selected_set = set(selected)
    
    # We need to know for each word, how many selected balls cover it.
    # We already have cover_counts.
    
    # Helper to check if a set of balls covers all words
    # We can just check if any cover_counts[y] == 0.
    
    # Local Search Loop
    no_improvement_count = 0
    max_no_improvement = 20000 # Tune this? 20k iterations is fast.
    
    while time.time() < deadline - 5 and no_improvement_count < max_no_improvement:
        # 1. Try to remove redundant balls
        removed_any = False
        # Shuffle selected to avoid bias
        to_check = selected[:]
        random.shuffle(to_check)
        
        for c in to_check:
            if c not in selected_set:
                continue # Already removed in this pass
            
            # Check if removing c makes any word uncovered
            can_remove = True
            for y in balls[c]:
                if cover_counts[y] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                selected_set.remove(c)
                selected.remove(c)
                for y in balls[c]:
                    cover_counts[y] -= 1
                removed_any = True
                no_improvement_count = 0 # Reset counter on improvement
                # Restart loop to check for more removals? 
                # Or just continue checking other balls in current 'to_check' list?
                # Continuing is safer to find multiple removals in one pass.
                
        if not removed_any:
            # 2. Try a swap
            # Pick a random ball A from selected
            if not selected:
                break
            A = random.choice(selected)
            
            # Pick a random ball B NOT in selected
            # We can pick a random ball from the whole space and check if it's in selected
            # Or maintain a list of unselected? 6561 is small, just pick random int and check.
            B = random.randint(0, total - 1)
            if B in selected_set:
                # Retry or just skip? Skipping is fine.
                no_improvement_count += 1
                continue
            
            # Check if swapping A with B is valid
            # New coverage for y:
            # If y in Ball(A) and y in Ball(B): count unchanged
            # If y in Ball(A) and y not in Ball(B): count - 1
            # If y not in Ball(A) and y in Ball(B): count + 1
            # If y not in either: unchanged
            
            # We need to ensure no count goes to 0.
            # This happens if y in Ball(A), y not in Ball(B), and cover_counts[y] == 1.
            
            valid_swap = True
            # We only need to check words in Ball(A) that are NOT in Ball(B)
            # Actually, we can just check all words in Ball(A).
            # If y in Ball(A):
            #   if y in Ball(B): new_count = old_count
            #   else: new_count = old_count - 1. Must be >= 1 => old_count >= 2.
            
            # To check "y in Ball(B)" quickly, we can precompute a boolean array or set for Ball(B)?
            # Ball size is 17. Checking membership in a list of 17 is fast.
            # But doing it for 17 words * 20k iterations = 340k ops. Fine.
            
            ball_B_set = set(balls[B])
            
            for y in balls[A]:
                if y not in ball_B_set:
                    if cover_counts[y] == 1:
                        valid_swap = False
                        break
                        
            if valid_swap:
                # Perform swap
                selected_set.remove(A)
                selected_set.add(B)
                selected.remove(A)
                selected.append(B)
                
                for y in balls[A]:
                    cover_counts[y] -= 1
                for y in balls[B]:
                    cover_counts[y] += 1
                
                no_improvement_count = 0 # Reset? 
                # A swap doesn't reduce size, but it might enable a future removal.
                # So we reset the "no removal" counter to allow more exploration.
            else:
                no_improvement_count += 1

    # Final cleanup: remove any remaining redundant balls
    # Just in case the loop ended with redundant balls
    changed = True
    while changed:
        changed = False
        for c in selected[:]:
            if c not in selected_set:
                continue
            can_remove = True
            for y in balls[c]:
                if cover_counts[y] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected_set.remove(c)
                selected.remove(c)
                for y in balls[c]:
                    cover_counts[y] -= 1
                changed = True
                
    # Convert best_solution to codewords
    codewords = []
    for idx in selected:
        digits = []
        x = idx
        for i in range(n):
            digits.append((x // powers[i]) % q)
        codewords.append(digits)
        
    return codewords
