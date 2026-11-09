import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 105  # Safety margin for 120s limit
    
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for other parameters
        total_words = q ** n
        centers = []
        covered = set()
        all_words = list(itertools.product(range(q), repeat=n))
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while covered != set(all_words) and time.time() < deadline:
            best_center = None
            max_new = 0
            for c in candidates:
                neigh = [c]
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_c = list(c)
                        new_c[i] = val
                        neigh.append(tuple(new_c))
                new_covered = [w for w in neigh if w not in covered]
                if len(new_covered) > max_new:
                    max_new = len(new_covered)
                    best_center = c
                    best_neigh = neigh
            if best_center is None or max_new == 0:
                break
            centers.append(list(best_center))
            for w in best_neigh:
                covered.add(w)
            candidates.remove(best_center)
        return centers
    
    # --- Optimized Path for q=3, n=8, radius=1 ---
    total_words = 3 ** 8  # 6561
    
    # 1. Pre-compute mapping and neighborhoods
    word_to_idx = {}
    idx_to_word = [None] * total_words
    
    count = 0
    for word in itertools.product(range(3), repeat=8):
        idx = 0
        for digit in word:
            idx = idx * 3 + digit
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        count += 1
    
    # neighbors[i] is a list of indices covered by center i (including i itself)
    neighbors = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        center = idx_to_word[i]
        neighbors[i].append(i)
        for pos in range(8):
            old_val = center[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                n_idx = word_to_idx[tuple(neighbor)]
                neighbors[i].append(n_idx)
    
    # 2. Greedy Set Cover
    uncovered = set(range(total_words))
    code_indices = []
    
    # To speed up, we can maintain a list of "active" candidates? 
    # Or just iterate all 6561 each time. 
    # 6561 * 17 * ~300 steps = ~33M ops. Might be slow in Python.
    # Let's optimize the inner loop.
    
    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        max_covered = 0
        
        for i in range(total_words):
            # Count how many of i's neighbors are in uncovered
            cnt = 0
            for j in neighbors[i]:
                if j in uncovered:
                    cnt += 1
            if cnt > max_covered:
                max_covered = cnt
                best_idx = i
                # Max possible is 17
                if cnt == 17:
                    break
        
        if best_idx == -1 or max_covered == 0:
            break
        
        code_indices.append(best_idx)
        # Remove covered words from uncovered
        for j in neighbors[best_idx]:
            if j in uncovered:
                uncovered.remove(j)
        
        # Optimization: Remove centers that are fully covered? 
        # No, a center might still cover uncovered words even if it's covered.
        # But we don't add duplicates.
        
    # If timeout occurred in greedy, the code might be invalid.
    # Let's check.
    covered_set = set()
    for i in code_indices:
        covered_set.update(neighbors[i])
    
    if len(covered_set) < total_words:
        # Fallback to valid construction (Direct Sum / Tetracode)
        # K_3(4,1) = 9 (Tetracode)
        # K_3(8,1) <= 9 * 9 = 81? No, that's not right.
        # Direct sum of two tetracodes:
        # C = { (c1, c2) | c1 in C4, c2 in C4 }
        # Size = 9 * 9 = 81?
        # Wait, the direct sum construction usually yields q^2 * K_q(n/2, 1) if n is even?
        # Actually, the standard "product" construction for covering codes:
        # If C1 is a covering code for (n1, r1) and C2 for (n2, r2),
        # then C1 x C2 covers (n1+n2, r1+r2).
        # Here r=1. If we split 8 into 4+4.
        # We need r1+r2 >= 1.
        # If r1=0, r2=1.
        # K_3(4,0) = 3^4 = 81.
        # K_3(4,1) = 9.
        # So C = All_4 x Tetracode_4.
        # Size = 81 * 9 = 729.
        # This is the 729 bound mentioned in the prompt analysis (seed program logic).
        
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        all4 = list(itertools.product(range(3), repeat=4))
        result = [list(h) + list(tail) for h in all4 for tail in tetracode]
        return result

    # 3. Local Search Minimization
    # We have a valid cover. Now try to reduce size.
    
    # Build counts: count[j] = number of centers in code_indices covering j
    counts = [0] * total_words
    
    # Helper to update counts
    def add_counts(idx):
        for j in neighbors[idx]:
            counts[j] += 1
            
    def remove_counts(idx):
        for j in neighbors[idx]:
            counts[j] -= 1

    for idx in code_indices:
        add_counts(idx)
        
    code_set = set(code_indices)
    
    # Phase A: Greedy Removal
    # Try to remove any center that doesn't reduce coverage
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for idx in list(code_set):
            # Temporarily remove
            remove_counts(idx)
            
            can_remove = True
            for j in neighbors[idx]:
                if counts[j] == 0:
                    can_remove = False
                    break
            
            if can_remove:
                code_set.remove(idx)
                removed_any = True
            else:
                # Restore
                add_counts(idx)
        
    # Phase B: Swap Search (1-for-1 replacement to enable further removals)
    # This is harder. We look for a center c in code, and a center c' not in code,
    # such that swapping c with c' allows us to remove another center c2.
    # Or simpler: just try to find a swap that keeps the code valid and maybe reduces "redundancy" to allow removals.
    # Given time constraints, let's stick to a more aggressive local search:
    # 1. Pick a center c.
    # 2. Identify "critical" neighbors of c (neighbors covered only by c).
    # 3. If there are critical neighbors, c cannot be removed.
    # 4. Try to find c' that covers all critical neighbors of c.
    # 5. If found, swap c with c'.
    # 6. Repeat removal phase.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Run removal phase first
        removed_any = True
        while removed_any:
            removed_any = False
            for idx in list(code_set):
                remove_counts(idx)
                can_remove = True
                for j in neighbors[idx]:
                    if counts[j] == 0:
                        can_remove = False
                        break
                if can_remove:
                    code_set.remove(idx)
                    removed_any = True
                else:
                    add_counts(idx)
                    
        # Now try swaps
        # To make this efficient, we should pick a "worse" center to swap?
        # Or just iterate.
        for c in list(code_set):
            if time.time() > deadline:
                break
                
            # Find neighbors of c that are ONLY covered by c
            critical = []
            for j in neighbors[c]:
                if counts[j] == 1:
                    critical.append(j)
                    
            if not critical:
                # c is removable, but removal phase should have caught it.
                # Unless counts changed?
                continue
                
            # Find c' not in code_set that covers all 'critical'
            found_swap = False
            # Limit search for c' to speed up?
            # Just iterate all non-codes.
            for c_prime in range(total_words):
                if c_prime in code_set:
                    continue
                    
                # Check if c_prime covers all critical
                valid_swap = True
                # Optimization: Pre-compute neighbor sets? 
                # neighbors[c_prime] is a list. 
                # Let's check intersection.
                # Since critical is small, loop is fast.
                for j in critical:
                    if j not in neighbors[c_prime]:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    # Perform swap
                    # Remove c
                    remove_counts(c)
                    code_set.remove(c)
                    
                    # Add c_prime
                    add_counts(c_prime)
                    code_set.add(c_prime)
                    
                    improved = True
                    found_swap = True
                    break # Restart outer loop? Or just continue?
                    # Restarting is safer to re-evaluate removals.
                    break
                    
            if found_swap:
                break
        
    # Final verification
    covered_set = set()
    for i in code_set:
        covered_set.update(neighbors[i])
        
    if len(covered_set) != total_words:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        all4 = list(itertools.product(range(3), repeat=4))
        result = [list(h) + list(tail) for h in all4 for tail in tetracode]
        return result
        
    # Convert to list of lists
    result = [list(idx_to_word[idx]) for idx in code_set]
    return result
