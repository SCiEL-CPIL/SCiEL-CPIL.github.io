# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm with random restarts and post-processing reduction.
    Uses bitmasks (Python integers) for efficient set operations.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    # For very large spaces, bitmask approach might be memory heavy or slow,
    # but for q=3, n=6, total_words=729, it is extremely efficient.
    if total_words > 20000:
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_bitmask(q, n, radius)

def _construct_greedy_bitmask(q, n, radius):
    total_words = q ** n
    all_mask = (1 << total_words) - 1
    
    # Precompute ball masks
    # ball_masks[i] is a bitmask of all words within Hamming distance 'radius' of i
    ball_masks = [0] * total_words
    
    # Helper to get index from word
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    # Helper to get word from index
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors for radius 1 (most common case for this task)
    # We can precompute the list of neighbors for each index to speed up mask generation
    # Or just generate the mask directly.
    
    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            word = idx_to_word(i)
            for pos in range(n):
                orig_val = word[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    neighbor = word.copy()
                    neighbor[pos] = val
                    j = word_to_idx(neighbor)
                    mask |= (1 << j)
            ball_masks[i] = mask
    else:
        # Generic radius support (slower)
        for i in range(total_words):
            mask = 1 << i
            word = idx_to_word(i)
            # Generate all words within distance 'radius'
            # This is expensive for large radius/n, but acceptable for small spaces
            # Simple BFS or iterative expansion
            # For radius 2, we can do 2 steps
            # Here we use a simple brute force for neighbors within radius
            # Since total_words is small, we can just check all j? No, that's O(N^2).
            # Better to expand.
            current_level = [i]
            visited = {i}
            for _ in range(radius):
                next_level = []
                for u in current_level:
                    w_u = idx_to_word(u)
                    for pos in range(n):
                        orig_val = w_u[pos]
                        for val in range(q):
                            if val == orig_val:
                                continue
                            neighbor = w_u.copy()
                            neighbor[pos] = val
                            v = word_to_idx(neighbor)
                            if v not in visited:
                                visited.add(v)
                                next_level.append(v)
                                mask |= (1 << v)
                current_level = next_level
            ball_masks[i] = mask

    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 115.0 # Leave 5s for safety
    
    rng = random.Random(42)
    
    restart_count = 0
    
    while True:
        # Randomized Greedy Set Cover
        # We maintain a list of uncovered bits.
        # To introduce randomness, we can shuffle the candidate order or use random tie-breaking.
        # A common technique: at each step, pick the candidate with max coverage.
        # If ties, pick randomly among the max.
        # Alternatively, randomize the initial order and use a priority queue? 
        # Simpler: iterate through all candidates, find max. If multiple have same max, pick one randomly.
        
        uncovered = all_mask
        codeword_indices = []
        
        while uncovered != 0:
            max_score = -1
            best_candidates = []
            
            for idx in range(total_words):
                # Calculate coverage of this idx on uncovered set
                score = (ball_masks[idx] & uncovered).bit_count()
                if score > max_score:
                    max_score = score
                    best_candidates = [idx]
                elif score == max_score and score > 0:
                    best_candidates.append(idx)
            
            if max_score == 0:
                break # Should not happen if ball_masks cover everything
                
            # Pick randomly from best candidates to ensure diversity in restarts
            chosen_idx = rng.choice(best_candidates)
            codeword_indices.append(chosen_idx)
            uncovered &= ~ball_masks[chosen_idx]
            
        # Post-processing: Reduction
        # Try to remove redundant codewords.
        # A codeword c is redundant if for every w in ball(c), there exists another c' in C\{c} such that w in ball(c').
        # Equivalent: union(ball(c') for c' in C\{c}) covers ball(c).
        # Or: (union of all balls) & ball(c) == ball(c).
        # Actually, if we maintain the union mask, removing c is valid if (union_mask_without_c) & ball(c) == ball(c).
        # union_mask_without_c = total_union ^ ball(c) is not quite right if balls overlap.
        # We need the union of the *other* balls.
        
        if len(codeword_indices) > 0:
            # Calculate coverage counts or just use a union mask approach?
            # Union mask approach:
            # Let U = OR of all ball_masks in codeword_indices.
            # If we remove c, new U' = OR of ball_masks in codeword_indices \ {c}.
            # c is redundant if ball(c) is subset of U'.
            # i.e., (ball(c) & U') == ball(c).
            
            # To make this fast, we can try removing each codeword one by one.
            # If we remove one, we must recompute the union of the rest? That's O(N^2).
            # Given N~73, N^2 is ~5000 ops, which is fine.
            
            active_indices = list(codeword_indices)
            
            # Sort by "centrality" or just iterate?
            # Iterating and removing if possible is a standard greedy reduction.
            # Order might matter. Let's try removing in reverse order of addition (last added often cover less unique area).
            
            changed = True
            while changed:
                changed = False
                for i in range(len(active_indices) - 1, -1, -1):
                    c_idx = active_indices[i]
                    # Calculate union of all other balls
                    union_others = 0
                    for j in range(len(active_indices)):
                        if j != i:
                            union_others |= ball_masks[active_indices[j]]
                    
                    # Check if ball(c_idx) is covered by union_others
                    if (ball_masks[c_idx] & union_others) == ball_masks[c_idx]:
                        # Redundant, remove
                        active_indices.pop(i)
                        changed = True
                        break # Restart loop as indices changed
                
            # Convert to words
            codewords = [idx_to_word(idx) for idx in active_indices]
            
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords
                
                # Early termination if we hit known optimal (73 for this specific instance)
                if best_len == 73:
                    break
                
        restart_count += 1
        
        # Check time
        if time.time() - start_time > max_time:
            break
            
        # Safety cap
        if restart_count > 20000:
            break
            
    return best_codewords

def _construct_greedy_simple(q, n, radius):
    """Fallback simple greedy for larger spaces."""
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        ball_masks[idx] = mask
        
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
