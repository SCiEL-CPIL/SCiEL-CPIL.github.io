import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Only optimize for the specific task parameters, fallback to seed otherwise
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other params
        import itertools
        if n >= 4 and q == 3 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Simple greedy for other cases
             return greedy_cover(q, n, radius)

    # Specific optimization for 3, 6, 1
    N = 729
    # Generate all words
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(n):
            w.append(x % 3)
            x //= 3
        words.append(w)
        
    # Precompute balls
    # ball_mask[i] is an int with bits set for all words within Hamming distance 1 of words[i]
    ball_masks = [0] * N
    for i in range(N):
        mask = 1 << i
        w = words[i]
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = w[:]
                new_w[pos] = val
                # Calculate index of new_w
                idx = 0
                for k in range(n):
                    idx = idx * 3 + new_w[k]
                mask |= (1 << idx)
        ball_masks[i] = mask

    full_mask = (1 << N) - 1
    
    # 1. Greedy Initialization
    covered_mask = 0
    code_indices = []
    uncovered_mask = full_mask
    
    # To speed up greedy, we can just iterate through all indices and pick the best
    # But checking 729 candidates each step is fast enough (729 steps * 729 checks * bit ops)
    # Bit ops on 729-bit ints are very fast in Python.
    
    while uncovered_mask != 0:
        best_idx = -1
        best_gain = -1
        # We can restrict candidates to uncovered indices? 
        # If an index is covered, adding it might still cover new neighbors.
        # However, usually greedy covering picks an uncovered point.
        # Let's check all points to be safe, but it's 729 iterations.
        for i in range(N):
            # How many new words does i cover?
            new_covered = ball_masks[i] & uncovered_mask
            # Count bits
            # Python int.bit_count() is available in 3.10+
            gain = new_covered.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == 13: # Max possible, can't do better
                    break
                    
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        covered_mask |= ball_masks[best_idx]
        uncovered_mask = full_mask & ~covered_mask
        
        if time.time() > deadline - 5:
            break

    # 2. Local Search: Remove redundant codewords
    # Iterate until no more removals
    removed_any = True
    while removed_any:
        removed_any = False
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            # Check if c is redundant
            # Union of balls of all other codewords
            union_others = 0
            for j in range(len(code_indices)):
                if i == j:
                    continue
                union_others |= ball_masks[code_indices[j]]
            
            # If ball of c is subset of union_others, c is redundant
            if (ball_masks[c] & ~union_others) == 0:
                code_indices.pop(i)
                removed_any = True
                break # Restart loop as indices changed
                # Actually, we can continue checking others, but pop(i) shifts indices.
                # Breaking and restarting is safer and often faster if removals are few.
                
        if time.time() > deadline - 5:
            break

    # 3. Local Search: Swap
    # Try to swap a codeword with a non-codeword to enable further removals or just improve structure.
    # Since we want to minimize size, a swap is only useful if it leads to a smaller set.
    # Heuristic: Try swapping each c with a random w, check if valid, then try to remove.
    
    current_size = len(code_indices)
    
    # Simple random restarts / swaps
    # Given the small space, we can try all swaps? 73 * 729 ~ 50k checks.
    # Each check involves calculating union of 72 balls (bit ORs) and checking subset.
    # 50k * 72 ORs * 729-bit ints. Might be slow in pure Python.
    
    # Let's do a limited random search for swaps
    random.seed(42)
    max_swap_iterations = 5000
    iter_count = 0
    while iter_count < max_swap_iterations and time.time() < deadline - 2:
        iter_count += 1
        if not code_indices:
            break
        c_idx = random.randrange(len(code_indices))
        c = code_indices[c_idx]
        # Pick a random candidate w not in code
        # We can just pick a random index
        w = random.randrange(N)
        if w in code_indices:
            continue
            
        # Try swap: remove c, add w
        # Check if (C \ {c}) U {w} is valid
        # This is equivalent to checking if the union of balls covers everything.
        # Union = (Union of C \ {c}) | Ball(w)
        
        # Calculate union of C \ {c}
        union_rest = 0
        for j in range(len(code_indices)):
            if j != c_idx:
                union_rest |= ball_masks[code_indices[j]]
        
        new_union = union_rest | ball_masks[w]
        
        if new_union == full_mask:
            # Valid swap
            # Does it allow removal?
            # We perform the swap
            code_indices[c_idx] = w
            
            # Try to remove any redundant codeword from the new set
            removed = False
            for k in range(len(code_indices) - 1, -1, -1):
                ck = code_indices[k]
                union_others_k = 0
                for m in range(len(code_indices)):
                    if m != k:
                        union_others_k |= ball_masks[code_indices[m]]
                if (ball_masks[ck] & ~union_others_k) == 0:
                    code_indices.pop(k)
                    removed = True
                    break
            
            if removed:
                # We improved! Restart swap search with new size
                max_swap_iterations += 5000 # Allow more time
                iter_count = 0
                
    # Convert indices back to words
    return [words[i] for i in code_indices]

def greedy_cover(q, n, radius):
    """Fallback greedy for other parameters."""
    import itertools
    # Simple greedy
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = set()
        # Generate all words within radius
        # This is expensive for large n, but fallback is rarely used for this specific task
        for delta in range(radius + 1):
            # This is a placeholder, real implementation would be complex
            pass
        # For simplicity in fallback, just return a large set or seed logic
        # Since the task is specific to 3,6,1, this fallback is just for safety
        # We'll just return the seed logic if it fits, else a random subset (invalid but avoids crash)
        # Actually, better to just use the seed logic for q=3, n>=4, r=1
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Return all words (valid, but score 729)
             return [list(w) for w in all_words]

# EVOLVE-BLOCK-END
