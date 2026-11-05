import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate standard superpermutation for n-1
    prev_word = construct(n-1)
    prev_list = [int(c) for c in prev_word]
    
    # Step 2: Extract the sequence of (n-1)-permutations from prev_word
    seen = set()
    blocks = []
    for i in range(len(prev_list) - (n-2)):
        window = tuple(prev_list[i:i+n-1])
        if len(set(window)) == n-1:
            if window not in seen:
                seen.add(window)
                blocks.append(window)
                
    # Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    num_blocks = len(blocks)
    overlap_matrix = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                overlap_matrix[i][j] = n - 1
                continue
            max_k = n - 1
            curr = blocks[i]
            next_b = blocks[j]
            for k in range(max_k, 0, -1):
                if curr[-k:] == next_b[:k]:
                    overlap_matrix[i][j] = k
                    break
    
    # Function to calculate length given an order of block indices
    def calc_length(order):
        if not order:
            return 0
        # First block: length is (n-1) + 1 + (n-1) = 2n-1
        total_len = 2 * (n - 1) + 1
        for i in range(len(order) - 1):
            curr_idx = order[i]
            next_idx = order[i+1]
            ov = overlap_matrix[curr_idx][next_idx]
            # Added length: 1 (for n) + (n-1) + (n-1) - ov
            total_len += 1 + 2 * (n - 1) - ov
        return total_len
    
    # Initialize with standard order (0 to num_blocks-1)
    best_order = list(range(num_blocks))
    best_len = calc_length(best_order)
    
    start_time = time.time()
    time_limit = 115 # Leave 5s for overhead
    
    # Variable Neighborhood Search
    # Neighborhoods:
    # 1. Swap adjacent
    # 2. 2-opt (reverse segment)
    # 3. Or-opt (move segment of length 1, 2, 3)
    
    def try_move_2opt(order, i, j):
        # Reverse segment i..j
        if i >= j:
            return None, None
        new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
        new_len = calc_length(new_order)
        return new_order, new_len
    
    def try_move_oropt(order, i, l, j):
        # Move segment of length l from i to j (j is insertion point)
        if l == 0 or i >= len(order) - l:
            return None, None
        segment = order[i:i+l]
        rest = order[:i] + order[i+l:]
        # Insert at j in rest
        # j is index in original order, need to adjust if j > i
        if j > i:
            insert_idx = j - l
        else:
            insert_idx = j
        if insert_idx < 0 or insert_idx > len(rest):
            return None, None
        new_order = rest[:insert_idx] + segment + rest[insert_idx:]
        if new_order == order:
            return None, None
        new_len = calc_length(new_order)
        return new_order, new_len

    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Neighborhood 1: Swap adjacent
        for i in range(num_blocks - 1):
            if time.time() - start_time > time_limit:
                break
            new_order, new_len = try_move_2opt(best_order, i, i+1)
            if new_len is not None and new_len < best_len:
                best_order = new_order
                best_len = new_len
                improved = True
        
        # Neighborhood 2: 2-opt (reverse segment)
        if time.time() - start_time > time_limit:
            break
        for i in range(num_blocks - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i+1, num_blocks):
                new_order, new_len = try_move_2opt(best_order, i, j)
                if new_len is not None and new_len < best_len:
                    best_order = new_order
                    best_len = new_len
                    improved = True
                    # Reset j to restart from i+1 for this i? Or just continue.
                    # Usually in 2-opt, you accept and continue.
                    break
        
        # Neighborhood 3: Or-opt (move segments of length 1, 2, 3)
        if time.time() - start_time > time_limit:
            break
        for l in [1, 2, 3]:
            if time.time() - start_time > time_limit:
                break
            for i in range(num_blocks - l + 1):
                if time.time() - start_time > time_limit:
                    break
                for j in range(num_blocks):
                    if j == i:
                        continue
                    new_order, new_len = try_move_oropt(best_order, i, l, j)
                    if new_len is not None and new_len < best_len:
                        best_order = new_order
                        best_len = new_len
                        improved = True
                        # Restart j loop for this i, l
                        break
                if improved:
                    # Reset i to 0? Or continue.
                    # Let's just break out of j and continue i
                    pass

    # Reconstruct the word from best_order
    # The word structure is:
    # b1 n b1 b2 n b2 ... bk n bk
    # But with overlaps.
    # Actually, the standard recursive construction builds it as:
    # Start with empty list.
    # For each block P in order:
    #   Add P, then n, then P?
    # No, the standard construction is:
    # S_n = P_1 n P_1 P_2 n P_2 ... P_k n P_k
    # But with overlaps between P_i (end of previous segment) and P_{i+1} (start of next segment).
    # Wait, the segment for P_i is "P_i n P_i".
    # The end of segment i is P_i.
    # The start of segment i+1 is P_{i+1}.
    # Overlap is between P_i and P_{i+1}.
    
    word = []
    for idx in best_order:
        p = blocks[idx]
        segment = list(p) + [n] + list(p)
        # Find overlap with current word
        max_ov = min(len(word), len(segment))
        ov = 0
        for k in range(max_ov, 0, -1):
            if word[-k:] == segment[:k]:
                ov = k
                break
        word.extend(segment[ov:])
        
    return "".join(map(str, word))
