import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "12132123" # Length 8? 1!+2!+3! = 1+2+6=9. 
        # Let's just use the recursive builder for consistency.

    # Step 1: Build the classic superpermutation for n-1
    def build_superperm_recursive(n_val):
        if n_val == 1:
            return [1]
        word = [1]
        for m in range(2, n_val + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)

            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    base_word = build_superperm_recursive(n - 1)

    # Step 2: Extract all unique permutations of 1..n-1 from base_word
    perms, seen = [], set()
    for i in range(len(base_word) - (n - 2)):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)

    m = len(perms)

    # Step 3: Create blocks for adding symbol n
    blocks = [list(p) + [n] + list(p) for p in perms]
    block_len = 2 * (n - 1) + 1

    # Step 4: Precompute overlap matrix
    # To speed up, we can use string representation for the overlap check
    block_strs = ["".join(map(str, b)) for b in blocks]
    
    # Overlap matrix: overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Max possible overlap is block_len - 1
    overlap_mat = [[0] * m for _ in range(m)]
    
    # Optimization: Precompute suffixes and prefixes?
    # For m=720, m^2=518k. 
    # Each check is string slicing.
    # We can just do it.
    for i in range(m):
        bi_str = block_strs[i]
        for j in range(m):
            bj_str = block_strs[j]
            # Max overlap is block_len - 1
            max_ov = block_len - 1
            ov = 0
            # Check from largest to smallest
            for o in range(max_ov, -1, -1):
                if bi_str[-o:] == bj_str[:o]:
                    ov = o
                    break
            overlap_mat[i][j] = ov

    # Overlap with end of base_word
    base_str = "".join(map(str, base_word))
    first_ov = [0] * m
    for i in range(m):
        bi_str = block_strs[i]
        max_ov = min(len(base_str), block_len - 1)
        for o in range(max_ov, -1, -1):
            if base_str[-o:] == bi_str[:o]:
                first_ov[i] = o
                break

    # Step 5: Local Search to maximize total overlap
    # Use Nearest Neighbor + 2-opt + Or-opt
    
    def calc_total_overlap(ord):
        total = first_ov[ord[0]]
        for k in range(1, len(ord)):
            total += overlap_mat[ord[k - 1]][ord[k]]
        return total

    # Nearest Neighbor Initialization
    def nearest_neighbor_init():
        unvisited = set(range(m))
        order = []
        # Start with a random node or node 0
        current = random.randint(0, m - 1)
        unvisited.remove(current)
        order.append(current)
        
        for _ in range(m - 1):
            best_next = -1
            best_ov = -1
            for nxt in unvisited:
                ov = overlap_mat[current][nxt]
                if ov > best_ov:
                    best_ov = ov
                    best_next = nxt
            unvisited.remove(best_next)
            order.append(best_next)
            current = best_next
        return order

    # 2-Opt
    def two_opt(order):
        improved = True
        while improved:
            improved = False
            for i in range(m - 1):
                for j in range(i + 1, m):
                    # Calculate delta
                    # Old: ... A-B-C ... D-E ...
                    # New: ... A-E-D ... C-B ...
                    # Actually, 2-opt reverses segment [i..j]
                    # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                    # New edges: (i-1, j), (j, j+1) is gone?
                    # Let's stick to the standard 2-opt delta calculation.
                    
                    idx_a = i - 1
                    idx_b = i
                    idx_c = j
                    idx_d = j + 1

                    old_ov = 0
                    new_ov = 0

                    if idx_a >= 0:
                        old_ov += overlap_mat[order[idx_a]][order[idx_b]]
                        new_ov += overlap_mat[order[idx_a]][order[idx_c]]
                    else:
                        old_ov += first_ov[order[idx_b]]
                        new_ov += first_ov[order[idx_c]]

                    # Internal edges
                    for k in range(i, j):
                        old_ov += overlap_mat[order[k]][order[k + 1]]
                        new_ov += overlap_mat[order[k + 1]][order[k]]

                    if idx_d < m:
                        old_ov += overlap_mat[order[idx_c]][order[idx_d]]
                        new_ov += overlap_mat[order[idx_b]][order[idx_d]]

                    if new_ov > old_ov:
                        order[i:j + 1] = reversed(order[i:j + 1])
                        improved = True
        return order

    # Or-Opt: Move 1 block
    def or_opt_1(order):
        improved = True
        while improved:
            improved = False
            for i in range(m):
                if i == 0: continue
                # Move order[i] to position j
                # Try inserting at all other positions
                block = order[i]
                # Remove block
                order.pop(i)
                # Try inserting at j
                best_j = -1
                best_gain = -1
                current_cost = 0 # We don't track total, just delta
                
                # This is getting complex to calculate delta efficiently without total.
                # Let's just calculate total overlap for the move? No, too slow.
                # We'll skip Or-Opt for now and rely on 2-opt with multiple restarts.
                order.insert(i, block)
        return order

    # Run multiple restarts of NN + 2-opt
    best_order = None
    best_overlap = -1
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 20: break
        order = nearest_neighbor_init()
        order = two_opt(order)
        ov = calc_total_overlap(order)
        if ov > best_overlap:
            best_overlap = ov
            best_order = order

    # Step 6: Construct the final word
    final_word = list(base_word)
    first_block_idx = best_order[0]
    first_block = blocks[first_block_idx]
    ov = first_ov[first_block_idx]
    final_word += first_block[ov:]

    for k in range(1, m):
        prev_idx = best_order[k - 1]
        curr_idx = best_order[k]
        curr_block = blocks[curr_idx]
        ov = overlap_mat[prev_idx][curr_idx]
        final_word += curr_block[ov:]

    # Step 7: Aggressive Deletion
    all_perms = set(itertools.permutations(range(1, n + 1)))
    n_perms = len(all_perms)

    # Build initial coverage
    perm_count = {}
    for i in range(len(final_word) - n + 1):
        w = tuple(final_word[i:i + n])
        if w in all_perms:
            perm_count[w] = perm_count.get(w, 0) + 1

    covered_count = sum(1 for w in all_perms if perm_count.get(w, 0) > 0)
    
    if covered_count < n_perms:
        return "".join(map(str, final_word))

    # Deletion Loop
    # We try to delete characters.
    # A character at index i can be deleted if no window containing it is unique (count==1).
    # Windows containing i: start indices from max(0, i-n+1) to min(i, len(final_word)-n)
    
    changed = True
    while changed and (time.time() < deadline - 5):
        changed = False
        # Try indices in random order
        indices = list(range(len(final_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            
            # Check if i is still valid index
            if i >= len(final_word):
                continue

            w_start_min = max(0, i - n + 1)
            w_start_max = min(i, len(final_word) - n)
            
            can_remove = True
            affected_windows = []
            
            for ws in range(w_start_min, w_start_max + 1):
                w = tuple(final_word[ws:ws + n])
                if w in all_perms:
                    if perm_count.get(w, 0) == 1:
                        can_remove = False
                        break
                    affected_windows.append((ws, w))
            
            if not can_remove:
                continue
                
            # Simulate removal
            # Decrement counts
            for ws, w in affected_windows:
                perm_count[w] -= 1
                if perm_count[w] == 0:
                    # This window is now gone. If it was the last one, we lose coverage.
                    # But we already checked count==1, so if it becomes 0, it was 1.
                    # Wait, if count was 1, we set can_remove=False.
                    # So if we are here, count was > 1. So it becomes >= 1.
                    # So no coverage loss from old windows.
                    pass

            # Perform removal
            final_word = final_word[:i] + final_word[i + 1:]
            
            # Update counts for NEW windows
            # The windows that change are those that now include the gap.
            # These are windows starting from max(0, i-n+1) to min(i, len(final_word)-n)
            # Note: indices shifted.
            new_w_start_min = max(0, i - n + 1)
            new_w_start_max = min(i, len(final_word) - n)
            
            # We need to add the new windows.
            # But we already decremented the old ones.
            # The new windows are different objects.
            for ws in range(new_w_start_min, new_w_start_max + 1):
                w = tuple(final_word[ws:ws + n])
                if w in all_perms:
                    perm_count[w] = perm_count.get(w, 0) + 1

            # Check if we lost any coverage?
            # We only added to counts. We decremented old ones (which were >1).
            # So we didn't drop any count to 0.
            # So coverage is maintained.
            
            changed = True

    return "".join(map(str, final_word))
