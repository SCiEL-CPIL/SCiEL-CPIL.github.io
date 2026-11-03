import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        pass

    # Generate standard S_{n-1}
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # Extract perms of size n-1
    perms, seen = [], set()
    w_size = n - 1
    for i in range(len(word) - w_size + 1):
        window = tuple(word[i:i + w_size])
        if len(set(window)) == w_size:
            if window not in seen:
                seen.add(window)
                perms.append(window)
                
    N = len(perms)
    
    if n >= 5:
        # Precompute overlap matrix
        # W[i][j] = overlap(perms[i], perms[j])
        W = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j: continue
                pj = perms[j]
                max_len = min(len(pi), len(pj))
                ov = 0
                for k in range(max_len, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                W[i][j] = ov

        # Initial order
        order = list(range(N))
        
        # Calculate initial total overlap
        total_ov = 0
        for i in range(1, N):
            total_ov += W[order[i-1]][order[i]]
            
        # Cost is proportional to -total_ov. Minimize length = Maximize total_ov.
        # Length = N * (2n-1) - total_ov.
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1-Opt: Swap adjacent
            for i in range(N - 1):
                if time.time() > deadline - 10: break
                
                idx_i = order[i]
                idx_ip1 = order[i+1]
                
                if i == 0:
                    left = -1
                else:
                    left = order[i-1]
                    
                if i+1 == N-1:
                    right = -1
                else:
                    right = order[i+2]
                    
                old_ov = 0
                if left != -1:
                    old_ov += W[left][idx_i]
                old_ov += W[idx_i][idx_ip1]
                if right != -1:
                    old_ov += W[idx_ip1][right]
                
                new_ov = 0
                if left != -1:
                    new_ov += W[left][idx_ip1]
                new_ov += W[idx_ip1][idx_i]
                if right != -1:
                    new_ov += W[idx_i][right]
                
                if new_ov > old_ov:
                    order[i], order[i+1] = order[i+1], order[i]
                    total_ov += (new_ov - old_ov)
                    improved = True
                    break # Restart scan
            
            # Or-Opt: Move single element from position i to position j
            if not improved:
                for i in range(N):
                    if time.time() > deadline - 10: break
                    elem = order[i]
                    
                    # Remove elem from position i
                    # Neighbors before removal
                    if i == 0:
                        left_i = -1
                    else:
                        left_i = order[i-1]
                    if i == N-1:
                        right_i = -1
                    else:
                        right_i = order[i+1]
                    
                    # Old cost contribution from edges involving position i
                    old_contrib = 0
                    if left_i != -1:
                        old_contrib += W[left_i][elem]
                    if right_i != -1:
                        old_contrib += W[elem][right_i]
                    if left_i != -1 and right_i != -1:
                        old_contrib += W[left_i][right_i]  # This edge disappears
                    # Actually, let me think more carefully.
                    # Before: ... left_i, elem, right_i ...
                    # Edges: (left_i, elem) + (elem, right_i)
                    # After removal: ... left_i, right_i ...
                    # Edge: (left_i, right_i)
                    # So delta from removal = W[left_i][right_i] - W[left_i][elem] - W[elem][right_i]
                    # (with appropriate handling of boundaries)
                    
                    # Try inserting at each position j (in the list after removal)
                    # After removal, list has N-1 elements.
                    # We insert elem at position j (0-indexed in the new list).
                    # If j == 0: elem is first
                    # If j == N-1: elem is last
                    # Otherwise: between new_list[j-1] and new_list[j]
                    
                    # To avoid modifying order, we compute deltas.
                    # Let's try all insertion positions.
                    
                    # We need to be careful: when we remove elem from position i,
                    # the positions after i shift left by 1.
                    
                    # Let's just try each possible target position in the original list,
                    # skipping i and i+1 (since moving to i+1 is equivalent to no move or adjacent swap).
                    
                    for j in range(N):
                        if j == i or j == i + 1:
                            continue
                        
                        # Determine insertion point in the list after removing elem
                        # If j > i: after removal, position j-1 in new list corresponds to original position j
                        # If j < i: after removal, position j in new list corresponds to original position j
                        
                        # Insertion neighbors in the final list:
                        if j == 0:
                            ins_left = -1
                        elif j > i:
                            # In the original list, after removing position i,
                            # the element now at position j-1 was originally at j
                            ins_left = order[j-1]
                        else:
                            # j < i, so position j is unchanged
                            ins_left = order[j-1] if j > 0 else -1
                        
                        if j == N-1:
                            ins_right = -1
                        elif j > i:
                            # After removing i, the element now at position j was originally at j+1
                            ins_right = order[j+1] if j+1 < N else -1
                        else:
                            # j < i, position j is unchanged
                            ins_right = order[j]
                        
                        # New cost contribution from edges at insertion point
                        new_contrib = 0
                        if ins_left != -1:
                            new_contrib += W[ins_left][elem]
                        if ins_right != -1:
                            new_contrib += W[elem][ins_right]
                        if ins_left != -1 and ins_right != -1:
                            new_contrib += W[ins_left][ins_right]  # This edge is replaced
                        # Wait, I need to think about this differently.
                        
                        # Let me recompute properly.
                        # The change affects:
                        # 1. The two edges at the removal point (i)
                        # 2. The two edges at the insertion point (j)
                        
                        # Removal at i:
                        # Before: left_i -> elem -> right_i  (edges: W[left_i][elem] + W[elem][right_i])
                        # After: left_i -> right_i  (edge: W[left_i][right_i])
                        # Delta_remove = W[left_i][right_i] - W[left_i][elem] - W[elem][right_i]
                        # (with -1 handling: if left_i == -1, no edge; etc.)
                        
                        delta_remove = 0
                        if left_i != -1 and right_i != -1:
                            delta_remove += W[left_i][right_i] - W[left_i][elem] - W[elem][right_i]
                        elif left_i != -1:
                            delta_remove += -W[left_i][elem]
                        elif right_i != -1:
                            delta_remove += -W[elem][right_i]
                        
                        # Insertion at j (in the modified list):
                        # Before insertion: ins_left -> ins_right (edge: W[ins_left][ins_right])
                        # After insertion: ins_left -> elem -> ins_right (edges: W[ins_left][elem] + W[elem][ins_right])
                        # Delta_insert = W[ins_left][elem] + W[elem][ins_right] - W[ins_left][ins_right]
                        
                        delta_insert = 0
                        if ins_left != -1 and ins_right != -1:
                            delta_insert += W[ins_left][elem] + W[elem][ins_right] - W[ins_left][ins_right]
                        elif ins_left != -1:
                            delta_insert += W[ins_left][elem]
                        elif ins_right != -1:
                            delta_insert += W[elem][ins_right]
                        
                        total_delta = delta_remove + delta_insert
                        
                        if total_delta > 0:
                            # Perform the move
                            # Remove elem from position i
                            del order[i]
                            # Adjust j for the shift
                            if j > i:
                                j -= 1
                            # Insert elem at position j
                            order.insert(j, elem)
                            total_ov += total_delta
                            improved = True
                            break  # Restart scan
                    
                    if improved:
                        break

            # 2-Opt: Reverse a segment
            if not improved:
                for i in range(N - 1):
                    if time.time() > deadline - 10: break
                    for j in range(i + 1, N):
                        # Reverse segment order[i..j]
                        # Edges affected:
                        # Before: (left, order[i]) + (order[i], order[i+1]) + ... + (order[j-1], order[j]) + (order[j], right)
                        # After: (left, order[j]) + (order[j], order[j-1]) + ... + (order[i+1], order[i]) + (order[i], right)
                        
                        if i == 0:
                            left = -1
                        else:
                            left = order[i-1]
                        if j == N-1:
                            right = -1
                        else:
                            right = order[j+1]
                        
                        # Old edges in the segment and at boundaries
                        old_edges = 0
                        if left != -1:
                            old_edges += W[left][order[i]]
                        for k in range(i, j):
                            old_edges += W[order[k]][order[k+1]]
                        if right != -1:
                            old_edges += W[order[j]][right]
                        
                        # New edges
                        new_edges = 0
                        if left != -1:
                            new_edges += W[left][order[j]]
                        for k in range(i, j):
                            new_edges += W[order[j-k+i]][order[j-k+i-1]]
                        if right != -1:
                            new_edges += W[order[i]][right]
                        
                        delta = new_edges - old_edges
                        if delta > 0:
                            order[i:j+1] = reversed(order[i:j+1])
                            total_ov += delta
                            improved = True
                            break  # Restart scan
                    
                    if improved:
                        break

    # Construct final word
    final_word = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0:
                overlap = 0
                break
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
        
    return "".join(map(str, final_word))
