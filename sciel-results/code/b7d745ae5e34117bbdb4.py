import itertools
import time

def construct(q, n, radius):
    # Estimate problem size
    total_words = q ** n
    
    # If the problem is too large for exhaustive greedy, fall back to the seed strategy
    # The seed strategy is 9 * 3^(n-4) for q=3.
    # For general q, a simple product code or random might be better, 
    # but the task is specifically the ternary 7-match pool.
    if total_words > 20000:
        # Fallback: Direct product of a small perfect code if possible, or just random
        # For q=3, n>=4, use tetracode product
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: Random sampling until covered (slow but valid)
            # Or just return a large set
            # For safety, return a grid-like structure
            # This case is unlikely for this specific task
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Greedy Set Cover Approach
    start_time = time.time()
    deadline = start_time + 105  # 15s margin
    
    # 1. Generate all words and map to indices
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_tuples = [tuple(w) for w in all_words]
    
    # 2. Precompute balls of radius 'radius' (assumed 1)
    # For efficiency, assume radius=1. If radius > 1, this needs adjustment.
    # The task specifies radius=1.
    balls = []
    for w in word_tuples:
        ball = set()
        ball.add(w)
        if radius == 1:
            for i in range(n):
                orig = w[i]
                for val in range(q):
                    if val != orig:
                        new_w = list(w)
                        new_w[i] = val
                        ball.add(tuple(new_w))
        else:
            # Generic radius handling (slower)
            # Generate all words and check distance
            pass # Fallback to distance check if radius != 1
            
            # For now, assume radius 1 is the main case
            # If radius != 1, we compute distance
            for other in word_tuples:
                dist = sum(1 for a, b in zip(w, other) if a != b)
                if dist <= radius:
                    ball.add(other)
            
        balls.append(ball)
        
    # 3. Greedy Selection
    uncovered = set(word_tuples)
    solution_indices = []
    
    # To speed up, we can maintain a list of active candidate indices
    active_candidates = list(range(total_words))
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = -1
        best_ball = None
        
        # Find candidate that covers most uncovered words
        for idx in active_candidates:
            # Optimization: if the ball of this candidate is a subset of already covered, skip?
            # We don't track covered explicitly, but uncovered is the complement.
            # Intersection size is what matters.
            count = len(uncovered & balls[idx])
            if count > best_count:
                best_count = count
                best_idx = idx
                best_ball = balls[idx]
                
        if best_count == 0:
            break # Should not happen
            
        solution_indices.append(best_idx)
        uncovered -= best_ball
        
        # Optional: Remove candidates that are now "worse" or redundant?
        # Not strictly necessary for 2187 items.
        
    # 4. Redundancy Removal
    # Convert indices to word tuples for processing
    solution_words = [word_tuples[i] for i in solution_indices]
    
    # Maintain coverage counts
    # cover_count[w] = number of codewords in solution that cover w
    cover_count = {w: 0 for w in word_tuples}
    
    for w in solution_words:
        for u in balls[word_tuples.index(w)]: # This index lookup is slow. 
            # Better to store balls by index and map solution_indices to balls
            pass
            
    # Let's redo redundancy removal using indices
    # cover_count is an array of size total_words
    cover_count = [0] * total_words
    
    # Map word tuple to index for fast access? We have word_tuples.
    # But balls are sets of tuples.
    # Let's convert balls to sets of indices?
    # That would be faster for counting.
    
    # Re-initialize for speed
    # 1. Create index for each word
    word_to_idx = {w: i for i, w in enumerate(word_tuples)}
    
    # 2. Convert balls to sets of indices
    balls_idx = []
    for b in balls:
        balls_idx.append(set(word_to_idx[w] for w in b))
        
    # 3. Recompute cover counts based on solution_indices
    for idx in solution_indices:
        for u_idx in balls_idx[idx]:
            cover_count[u_idx] += 1
            
    # 4. Iteratively remove redundant codewords
    # A codeword idx is redundant if for all u in balls_idx[idx], cover_count[u] > 1
    # (because if we remove it, count becomes count-1, which must be > 0)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate over a copy of solution_indices to allow removal
        for i in range(len(solution_indices)):
            if time.time() > deadline: break
            
            idx = solution_indices[i]
            is_redundant = True
            for u_idx in balls_idx[idx]:
                if cover_count[u_idx] <= 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # Remove idx
                for u_idx in balls_idx[idx]:
                    cover_count[u_idx] -= 1
                solution_indices.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    # Final check: ensure coverage
    # If the time limit was hit during greedy, we might have uncovered words.
    # We must add more codewords to cover them.
    if uncovered:
        # Add greedy picks for remaining
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            best_ball = None
            for idx in range(total_words):
                count = len(uncovered & balls[idx])
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    best_ball = balls[idx]
            if best_count == 0: break
            solution_indices.append(best_idx)
            uncovered -= best_ball
            # Update cover counts? We just need to return the list.
            # The redundancy removal is skipped for these added ones to save time,
            # or we can run it again if time permits.
            
    # Convert back to list of lists
    result = [all_words[i] for i in solution_indices]
    
    return result
