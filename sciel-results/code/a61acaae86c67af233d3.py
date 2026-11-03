import time
import sys
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Recursive construction
    word = [1]
    
    # Pre-compute factorials? Not strictly needed if we just build.
    
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        L = len(word)
        # word length is L. windows of size m-1
        for i in range(L - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # If this is the last step (m == n) and n > 4, try to optimize the order
        # of permutations to maximize overlap.
        if m == n and n > 4:
            # We have a list of perms (tuples of length n-1)
            # We want to order them to maximize sum of overlaps.
            # Overlap(u, v) = longest k such that u[-k:] == v[:k]
            
            # Heuristic: Nearest Neighbor
            # Start with the first perm in the list (or any)
            # Then repeatedly pick the unused perm with max overlap with current end.
            
            remaining = set(perms)
            current = perms[0]
            remaining.remove(current)
            ordered_perms = [current]
            
            # Optimization: To speed up, we can precompute overlaps?
            # 720^2 = 518,400 comparisons. Each comparison is up to 6 chars.
            # This is fast enough.
            
            # Greedy: Pick neighbor with max overlap.
            # To break ties, maybe pick the one with max overlap with the *next* neighbor?
            # Or just simple nearest neighbor.
            
            # To improve on simple NN, we can do a 2-opt local search on the order.
            # But NN is a good start.
            
            # Let's implement a simple greedy with a look-ahead or just NN.
            # Given time constraints, NN is safe.
            
            # Actually, the "order of appearance" in the standard construction is often quite good.
            # Let's try to see if we can improve it.
            
            # Fast overlap calculation
            def calc_overlap(u, v):
                max_k = min(len(u), len(v))
                for k in range(max_k, 0, -1):
                    if u[-k:] == v[:k]:
                        return k
                return 0

            # To make it faster, we can group perms by their suffixes.
            # But for 720 items, direct calculation is O(720^2 * 6) ~ 3M ops. Very fast.
            
            # Let's try to find a better starting point or use a simple local search.
            # We'll use the standard order as a baseline and try to improve it with 2-opt swaps.
            
            base_order = perms
            base_score = 0
            for i in range(len(base_order) - 1):
                base_score += calc_overlap(base_order[i], base_order[i+1])
                
            # Try to improve with local search
            current_order = list(base_order)
            current_score = base_score
            improved = True
            max_iterations = 1000 # Limit iterations to avoid timeout
            
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Randomly pick a segment and try to reverse it or move it?
                # Simple 2-opt: pick i, j and reverse segment [i:j]
                # Or swap i, j?
                # Reversing a segment in a path changes the connections at i-1/i and j/j+1.
                
                # To be efficient, we can just try a few random reversals.
                for _ in range(100):
                    if len(current_order) < 2: break
                    i = random.randint(0, len(current_order) - 2)
                    j = random.randint(i + 1, len(current_order) - 1)
                    
                    # Calculate score change for reversing current_order[i:j+1]
                    # Old links: (i-1, i) and (j, j+1)
                    # New links: (i-1, j) and (i, j+1)
                    
                    old_score = 0
                    new_score = 0
                    
                    if i > 0:
                        old_score += calc_overlap(current_order[i-1], current_order[i])
                        new_score += calc_overlap(current_order[i-1], current_order[j])
                    else:
                        # No left neighbor for i
                        pass
                        
                    if j < len(current_order) - 1:
                        old_score += calc_overlap(current_order[j], current_order[j+1])
                        new_score += calc_overlap(current_order[i], current_order[j+1])
                    else:
                        pass
                        
                    # Internal links in [i:j] are reversed, but sum of overlaps might change?
                    # Overlap is not symmetric. u->v != v->u.
                    # So we must recalculate the whole segment or ensure symmetry?
                    # Overlap(u, v) is not symmetric.
                    # So reversing the segment changes all internal overlaps.
                    # This makes 2-opt expensive if we don't precompute.
                    
                    # Given the small size, we can just recalculate the whole score?
                    # No, 720 items. 1000 iterations * 720 ops = 720k ops. Fast.
                    
                    # Let's just recalculate the total score for the modified order.
                    # It's O(N).
                    new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                    
                    score = 0
                    for k in range(len(new_order) - 1):
                        score += calc_overlap(new_order[k], new_order[k+1])
                        
                    if score > current_score:
                        current_score = score
                        current_order = new_order
                        improved = True
                        break # Restart loop? Or continue?
                        
            if current_score > base_score:
                perms = current_order
            else:
                perms = base_order
        else:
            # For smaller m, just use the standard order (which is already optimized by construction)
            pass

        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
    
    return "".join(map(str, word))
