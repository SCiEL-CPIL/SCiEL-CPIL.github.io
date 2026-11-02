"""Combined program: Greedy Set Cover with Local Search Pruning."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit

    # Fallback for non-target parameters (though task specifies q=3, n=6, R=1)
    if q != 3 or n != 6 or radius != 1:
        # Direct sum of tetracode (9 words, len 4) with all words for remaining coords
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: Random or simple construction
            # Just return a large random set to ensure validity, though score will be poor
            # For this specific task, we assume q=3, n=6.
            return list(itertools.product(range(q), repeat=n))[:100]

    total = q ** n
    # Map word (tuple) to index (int)
    def to_idx(word):
        b = 0
        for x in word:
            b = b * q + x
        return b

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute balls for all words
    # ball[i] is a list of indices within Hamming distance 1 of idx i
    balls = []
    for i in range(total):
        w = idx_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for v in range(q):
                if v != orig:
                    new_w = w[:]
                    new_w[pos] = v
                    ball.append(to_idx(new_w))
        balls.append(ball)

    # Greedy Set Cover
    uncovered = set(range(total))
    code = []
    
    # To speed up gain calculation, we can just iterate.
    # N=729, ball size=13.
    # Iterations ~ 60-80.
    # Ops ~ 80 * 729 * 13 = 750k. Very fast.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_idx = -1
        
        # Find word with max new coverage
        for i in range(total):
            # Optimization: skip if i is already in code? 
            # No, we can pick any word. But picking an uncovered word is usually good.
            # Actually, any word can be a codeword.
            # However, if a word is already covered, its "new" coverage is smaller.
            # We just compute intersection with uncovered.
            
            # Small optimization: if i is in code, skip? No, we don't track code membership for gain.
            # But we don't need to add the same word twice.
            if i in code: 
                continue
                
            gain = 0
            for b in balls[i]:
                if b in uncovered:
                    gain += 1
                    if gain <= best_gain:
                        break # Early exit if we can't beat best_gain
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            break # Should not happen
            
        code.append(best_idx)
        # Remove covered
        for b in balls[best_idx]:
            if b in uncovered:
                uncovered.remove(b)
                
    # Local Search: Pruning
    # Try to remove words from the code if the coverage is still maintained.
    # We repeat this in random orders to avoid local minima.
    
    def is_valid(code_indices):
        # Check if union of balls covers all
        covered = set()
        for c in code_indices:
            covered.update(balls[c])
        return len(covered) == total

    current_code = list(code)
    
    # We can try multiple random permutations for pruning
    # But first, try a deterministic pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to find different removals
        random.shuffle(current_code)
        
        for i in range(len(current_code)):
            if time.time() > deadline:
                break
            
            # Try removing current_code[i]
            c_to_remove = current_code[i]
            temp_code = current_code[:i] + current_code[i+1:]
            
            if is_valid(temp_code):
                current_code = temp_code
                improved = True
                # Since list changed, restart loop or adjust index
                # Restarting is safer and simple
                break
        
    # Additional refinement: Random restarts of the whole greedy? 
    # Or just return the best found.
    # The pruning step is the key improvement over raw greedy.
    
    # Convert back to words
    result = [idx_to_word(idx) for idx in current_code]
    
    # Verify validity one last time (safety)
    if not is_valid(current_code):
        # Fallback to the greedy result before pruning if pruning broke it (shouldn't happen)
        result = [idx_to_word(idx) for idx in code]
        
    return result
# EVOLVE-BLOCK-END
