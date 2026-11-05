# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if n <= 6:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)
        
        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                pj = perms[j]
                best = 0
                # Max possible overlap is 5 since pi != pj
                for k in range(5, 0, -1):
                    if pi[-k:] == pj[:k]:
                        best = k
                        break
                ov[i][j] = best

        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total

        def build_word_from_path(path):
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word

        # Initialize with greedy nearest-neighbor for better starting point
        current_path = []
        used = [False] * N
        # Start with first permutation
        current_path.append(0)
        used[0] = True
        
        for _ in range(N - 1):
            last = current_path[-1]
            best_next = -1
            best_ov = -1
            for j in range(N):
                if not used[j]:
                    o = ov[last][j]
                    if o > best_ov:
                        best_ov = o
                        best_next = j
            current_path.append(best_next)
            used[best_next] = True
        
        current_overlap = calc_total_overlap(current_path)
        best_path = current_path[:]
        best_overlap = current_overlap

        # Simulated Annealing
        # Parameters
        T0 = 10.0
        Tend = 0.01
        alpha = 0.99999  # Cooling rate
        T = T0
        
        # Precompute prefix sums for faster cost update? 
        # For SA, we usually just update the delta cost directly for local moves.
        # Delta cost for moving elements requires checking neighbors.
        
        # To speed up SA, we can use a few restarts if time permits, 
        # but a single long SA run with good moves is often better.
        
        # Move types:
        # 1. Swap adjacent (i, i+1)
        # 2. 2-Opt (reverse segment i..j)
        # 3. Relocate (move element i to position j)
        
        # We will use a mix.
        
        # Helper to calculate cost change for 2-Opt reverse [i, j]
        # Cost = sum ov[p[k]][p[k+1]]
        # Old cost parts: ov[p[i-1]][p[i]] + sum_{k=i}^{j-1} ov[p[k]][p[k+1]] + ov[p[j]][p[j+1]]
        # New cost parts: ov[p[i-1]][p[j]] + sum_{k=i}^{j-1} ov[p[k+1]][p[k]] + ov[p[i]][p[j+1]]
        # We need prefix sums of forward and backward overlaps to do this in O(1).
        
        fwd_prefix = [0] * (N + 1)
        for i in range(N - 1):
            fwd_prefix[i + 1] = fwd_prefix[i] + ov[current_path[i]][current_path[i+1]]
            
        bwd_prefix = [0] * (N + 1)
        for i in range(N - 1):
            bwd_prefix[i + 1] = bwd_prefix[i] + ov[current_path[i+1]][current_path[i]]

        def get_fwd_sum(i, j):
            # Sum of ov[p[k]][p[k+1]] for k in [i, j-1]
            if i >= j:
                return 0
            return fwd_prefix[j] - fwd_prefix[i]
            
        def get_bwd_sum(i, j):
            # Sum of ov[p[k+1]][p[k]] for k in [i, j-1]
            if i >= j:
                return 0
            return bwd_prefix[j] - bwd_prefix[i]

        def update_prefixes(path):
            nonlocal fwd_prefix, bwd_prefix
            fwd_prefix = [0] * (N + 1)
            for i in range(N - 1):
                fwd_prefix[i + 1] = fwd_prefix[i] + ov[path[i]][path[i+1]]
            bwd_prefix = [0] * (N + 1)
            for i in range(N - 1):
                bwd_prefix[i + 1] = bwd_prefix[i] + ov[path[i+1]][path[i]]

        # SA Loop with O(1) prefix updates
        def update_prefixes_local(i, j):
            """Update fwd_prefix and bwd_prefix for range [i, j] in-place (O(j-i))"""
            nonlocal fwd_prefix, bwd_prefix
            # Recompute from i to j
            for k in range(i, j):
                fwd_prefix[k + 1] = fwd_prefix[k] + ov[current_path[k]][current_path[k+1]]
                bwd_prefix[k + 1] = bwd_prefix[k] + ov[current_path[k+1]][current_path[k]]
            # Propagate forward from j+1
            for k in range(j, N - 1):
                fwd_prefix[k + 1] = fwd_prefix[k] + ov[current_path[k]][current_path[k+1]]
                bwd_prefix[k + 1] = bwd_prefix[k] + ov[current_path[k+1]][current_path[k]]

        # Better cooling: linear interpolation
        sa_steps = 0
        max_sa_steps = 200000
        
        while T > Tend and time.time() < deadline - 5 and sa_steps < max_sa_steps:
            sa_steps += 1
            # Choose move type
            r = random.random()
            
            if r < 0.25:
                # 2-Opt: reverse segment [i, j]
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                
                left_idx = i - 1
                right_idx = j + 1
                
                old_cost = 0
                if left_idx != -1:
                    old_cost += ov[current_path[left_idx]][current_path[i]]
                old_cost += get_fwd_sum(i, j)
                if right_idx != N:
                    old_cost += ov[current_path[j]][current_path[right_idx]]
                
                new_cost = 0
                if left_idx != -1:
                    new_cost += ov[current_path[left_idx]][current_path[j]]
                new_cost += get_bwd_sum(i, j)
                if right_idx != N:
                    new_cost += ov[current_path[i]][current_path[right_idx]]
                
                delta = new_cost - old_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_path[i:j+1] = reversed(current_path[i:j+1])
                    current_overlap += delta
                    # Update prefixes only for the affected range
                    update_prefixes_local(i, j)
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = current_path[:]
                        
            elif r < 0.5:
                # Swap adjacent
                i = random.randint(0, N - 2)
                j = i + 1
                
                left_idx = i - 1
                right_idx = j + 1
                
                old_cost = 0
                if left_idx != -1:
                    old_cost += ov[current_path[left_idx]][current_path[i]]
                old_cost += ov[current_path[i]][current_path[j]]
                if right_idx != N:
                    old_cost += ov[current_path[j]][current_path[right_idx]]
                
                new_cost = 0
                if left_idx != -1:
                    new_cost += ov[current_path[left_idx]][current_path[j]]
                new_cost += ov[current_path[j]][current_path[i]]
                if right_idx != N:
                    new_cost += ov[current_path[i]][current_path[right_idx]]
                
                delta = new_cost - old_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_path[i], current_path[j] = current_path[j], current_path[i]
                    current_overlap += delta
                    # Update prefixes for the small affected range
                    update_prefixes_local(i, j)
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = current_path[:]
                        
            else:
                # Relocate: remove element at i, insert at position j
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j or i == j + 1:
                    T = T0 * (Tend / T0) ** (sa_steps / max_sa_steps)
                    continue
                    
                item = current_path[i]
                
                # Calculate old cost contribution of removing item at position i
                left_idx = i - 1
                right_idx = i + 1
                
                old_remove = 0
                if left_idx != -1 and right_idx != N:
                    old_remove = ov[current_path[left_idx]][item] + ov[item][current_path[right_idx]] - ov[current_path[left_idx]][current_path[right_idx]]
                elif left_idx != -1:
                    old_remove = ov[current_path[left_idx]][item]
                elif right_idx != N:
                    old_remove = ov[item][current_path[right_idx]]
                
                # Determine insertion neighbors in the original list (before removal)
                # After removing at i, the list shifts. We insert at position j in the new list.
                # If j > i, the original index is j (since we removed before it, it shifts left by 1, 
                # but we're inserting at j in the new list, which corresponds to j in old list if j > i)
                # If j <= i, the original index is j.
                
                if j > i:
                    # In the new list, position j corresponds to original position j
                    # (because removing at i < j shifts everything from i+1 to j-1 left by 1,
                    #  so new position j = old position j)
                    ins_left_orig = j - 1  # This is >= i, so it's the element that was at j-1 in old list
                    ins_right_orig = j + 1 if j < N else -1
                    # But wait: after removing at i, the element at old position j is now at new position j-1
                    # We want to insert at new position j, which is between new[j-1] and new[j]
                    # new[j-1] = old[j-1] if j-1 >= i, else old[j-2]... 
                    # This is getting complex. Let me just do it carefully.
                    
                    # After removing index i from a list of length N, we have N-1 elements.
                    # We insert at position j (0-indexed) in this new list of N-1 elements.
                    # The neighbors in the new list are:
                    #   left neighbor: new_list[j-1] if j > 0
                    #   right neighbor: new_list[j] if j < N-1
                    
                    # Mapping from new_list indices to old indices:
                    # For new_list index k:
                    #   if k < i: old index = k
                    #   if k >= i: old index = k + 1
                    
                    if j == 0:
                        nl_left = -1
                    else:
                        if j - 1 < i:
                            nl_left = j - 1
                        else:
                            nl_left = j  # new[j-1] = old[j]
                    
                    if j == N - 1:
                        nl_right = -1
                    else:
                        if j < i:
                            nl_right = j
                        else:
                            nl_right = j + 1  # new[j] = old[j+1]
                    
                else:
                    # j <= i
                    # After removing at i, positions 0..i-1 are unchanged, i..N-2 shift left.
                    # Insert at position j in new list.
                    # new_list[j-1] and new_list[j] map to:
                    if j == 0:
                        nl_left = -1
                    else:
                        # j-1 < i always since j <= i and j-1 < j <= i
                        nl_left = j - 1
                    
                    if j == N - 1:
                        nl_right = -1
                    else:
                        # j < i or j == i
                        if j < i:
                            nl_right = j
                        else:
                            # j == i, new_list[i] = old[i+1]
                            nl_right = i + 1 if i + 1 < N else -1
                
                # Now calculate insertion gain using old-list indices
                gain_insert = 0
                if nl_left != -1 and nl_right != -1:
                    gain_insert = ov[current_path[nl_left]][item] + ov[item][current_path[nl_right]] - ov[current_path[nl_left]][current_path[nl_right]]
                elif nl_left != -1:
                    gain_insert = ov[current_path[nl_left]][item]
                elif nl_right != -1:
                    gain_insert = ov[item][current_path[nl_right]]
                
                delta = -old_remove + gain_insert
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Perform the move
                    del current_path[i]
                    if i < j:
                        j -= 1
                    current_path.insert(j, item)
                    current_overlap += delta
                    # Update prefixes - this is O(N) but relocate is less frequent
                    update_prefixes(current_path)
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = current_path[:]
            
            # Linear cooling schedule
            T = T0 * (Tend / T0) ** (sa_steps / max_sa_steps)

        # Second SA restart if time permits
        if time.time() < deadline - 20 and best_overlap < current_overlap + 50:
            # Reset with a different random starting point
            random.seed(int(time.time() * 1000) % (2**31))
            current_path = list(range(N))
            random.shuffle(current_path)
            current_overlap = calc_total_overlap(current_path)
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = current_path[:]
            
            update_prefixes(current_path)
            
            T = T0 * 0.5  # Start cooler for second pass
            sa_steps = 0
            max_sa_steps2 = 100000
            
            while T > Tend * 0.5 and time.time() < deadline - 10 and sa_steps < max_sa_steps2:
                sa_steps += 1
                r = random.random()
                
                if r < 0.3:
                    # 2-Opt
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    left_idx = i - 1
                    right_idx = j + 1
                    
                    old_cost = 0
                    if left_idx != -1:
                        old_cost += ov[current_path[left_idx]][current_path[i]]
                    old_cost += get_fwd_sum(i, j)
                    if right_idx != N:
                        old_cost += ov[current_path[j]][current_path[right_idx]]
                    
                    new_cost = 0
                    if left_idx != -1:
                        new_cost += ov[current_path[left_idx]][current_path[j]]
                    new_cost += get_bwd_sum(i, j)
                    if right_idx != N:
                        new_cost += ov[current_path[i]][current_path[right_idx]]
                    
                    delta = new_cost - old_cost
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_path[i:j+1] = reversed(current_path[i:j+1])
                        current_overlap += delta
                        update_prefixes_local(i, j)
                        if current_overlap > best_overlap:
                            best_overlap = current_overlap
                            best_path = current_path[:]
                else:
                    # Swap adjacent
                    i = random.randint(0, N - 2)
                    j = i + 1
                    left_idx = i - 1
                    right_idx = j + 1
                    
                    old_cost = 0
                    if left_idx != -1:
                        old_cost += ov[current_path[left_idx]][current_path[i]]
                    old_cost += ov[current_path[i]][current_path[j]]
                    if right_idx != N:
                        old_cost += ov[current_path[j]][current_path[right_idx]]
                    
                    new_cost = 0
                    if left_idx != -1:
                        new_cost += ov[current_path[left_idx]][current_path[j]]
                    new_cost += ov[current_path[j]][current_path[i]]
                    if right_idx != N:
                        new_cost += ov[current_path[i]][current_path[right_idx]]
                    
                    delta = new_cost - old_cost
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_path[i], current_path[j] = current_path[j], current_path[i]
                        current_overlap += delta
                        update_prefixes_local(i, j)
                        if current_overlap > best_overlap:
                            best_overlap = current_overlap
                            best_path = current_path[:]
                
                T = T0 * 0.5 * (Tend * 0.5 / (T0 * 0.5)) ** (sa_steps / max_sa_steps2)

        # Final Polish: Best Improvement Local Search on the best path found by SA
        # This often recovers a few more points of overlap.
        current_path = best_path[:]
        current_overlap = best_overlap
        update_prefixes(current_path)
        
        improved = True
        passes = 0
        max_passes = 100
        
        while improved and time.time() < deadline - 2 and passes < max_passes:
            improved = False
            passes += 1
            
            # 2-Opt Best Improvement
            best_gain = 0
            best_i = -1
            best_j = -1
            
            for i in range(N - 1):
                left_idx = i - 1
                for j in range(i + 1, N):
                    right_idx = j + 1
                    
                    old_cost = 0
                    if left_idx != -1:
                        old_cost += ov[current_path[left_idx]][current_path[i]]
                    old_cost += get_fwd_sum(i, j)
                    if right_idx != N:
                        old_cost += ov[current_path[j]][current_path[right_idx]]
                    
                    new_cost = 0
                    if left_idx != -1:
                        new_cost += ov[current_path[left_idx]][current_path[j]]
                    new_cost += get_bwd_sum(i, j)
                    if right_idx != N:
                        new_cost += ov[current_path[i]][current_path[right_idx]]
                    
                    gain = new_cost - old_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                current_path[best_i:best_j+1] = reversed(current_path[best_i:best_j+1])
                current_overlap += best_gain
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = current_path[:]
                update_prefixes(current_path)
                improved = True
                continue

            # 1-Opt Relocate Best Improvement (Fixed)
            best_gain = 0
            best_i = -1
            best_j = -1
            for i in range(N):
                item = current_path[i]
                left_val = current_path[i-1] if i > 0 else -1
                right_val = current_path[i+1] if i < N - 1 else -1
                
                gain_remove = 0
                if left_val != -1 and right_val != -1:
                    gain_remove = ov[left_val][right_val] - ov[left_val][item] - ov[item][right_val]
                elif left_val != -1:
                    gain_remove = -ov[left_val][item]
                elif right_val != -1:
                    gain_remove = -ov[item][right_val]
                
                for j in range(N):
                    if j == i or j == i + 1:
                        continue
                    
                    # Determine neighbors at insertion position j in the list AFTER removing i
                    if j < i:
                        # Insert before position i, so indices 0..j-1 unchanged, j..i-1 shift right
                        ins_left = current_path[j-1] if j > 0 else -1
                        ins_right = current_path[j]
                    else:
                        # j > i (j == i+1 already skipped)
                        # After removing i, position j in new list:
                        # if j == i: skip (handled)
                        # if j > i: new_list[j-1] = old_list[j-1] if j-1 < i else old_list[j]
                        #          new_list[j] = old_list[j] if j < i else old_list[j+1]
                        # Since j > i:
                        #   new_list[j-1]: j-1 >= i, so = old_list[j]
                        #   new_list[j]: j >= i, so = old_list[j+1] if j+1 < N else -1
                        if j - 1 < i:
                            ins_left = current_path[j-1]
                        else:
                            ins_left = current_path[j] if j < N else -1
                        
                        if j < N - 1:
                            if j < i:
                                ins_right = current_path[j]
                            else:
                                ins_right = current_path[j+1]
                        else:
                            ins_right = -1
                    
                    gain_insert = 0
                    if ins_left != -1 and ins_right != -1:
                        gain_insert = ov[ins_left][item] + ov[item][ins_right] - ov[ins_left][ins_right]
                    elif ins_left != -1:
                        gain_insert = ov[ins_left][item]
                    elif ins_right != -1:
                        gain_insert = ov[item][ins_right]
                    
                    total_gain = gain_remove + gain_insert
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                i = best_i
                j = best_j
                item = current_path[i]
                del current_path[i]
                if i < j:
                    j -= 1
                current_path.insert(j, item)
                current_overlap += best_gain
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = current_path[:]
                update_prefixes(current_path)
                improved = True

        # Build final word
        final_word = build_word_from_path(best_path)
        return "".join(map(str, final_word))
    
    # Fallback for n > 7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
