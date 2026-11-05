import time
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations of 1..n-1
    base_symbols = list(range(1, n))
    all_perms = list(permutations(base_symbols))
    num_blocks = len(all_perms)  # (n-1)!
    
    # Precompute overlap matrix: overlap[i][j] = longest k such that all_perms[i][-k:] == all_perms[j][:k]
    # This is the overlap when transitioning from block i to block j
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                continue
            b1 = all_perms[i]
            b2 = all_perms[j]
            max_k = min(len(b1), len(b2))
            for k in range(max_k, 0, -1):
                if b1[-k:] == b2[:k]:
                    overlap[i][j] = k
                    break
    
    # The length of the superpermutation given an order of blocks:
    # First block: len(P) + 1 + len(P) = 2*(n-1) + 1
    # Each subsequent block j after block i: 2*(n-1) + 1 - overlap[i][j]
    # Total: 2*(n-1) + 1 + sum_{transitions} (2*(n-1) + 1 - overlap[i][j])
    #       = 2*(n-1) + 1 + (num_blocks - 1) * (2*(n-1) + 1) - sum(overlaps)
    # To minimize length, maximize sum of overlaps.
    
    def calc_length(order):
        total_overlap = 0
        for i in range(len(order) - 1):
            total_overlap += overlap[order[i]][order[i + 1]]
        base = 2 * (n - 1) + 1
        return base + (len(order) - 1) * (2 * (n - 1) + 1) - total_overlap
    
    # Initialize with standard order (lexicographic)
    best_order = list(range(num_blocks))
    best_len = calc_length(best_order)
    
    start_time = time.time()
    time_limit = 115  # Leave 5s safety margin for 120s limit
    
    # Local search with multiple neighborhoods
    # Use incremental overlap sum tracking for efficiency
    
    def vns_search(order, current_len):
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Neighborhood 1: Or-opt (move segment of length 1, 2, 3)
            for seg_len in [1, 2, 3]:
                if time.time() - start_time > time_limit:
                    return order, current_len
                for i in range(num_blocks - seg_len):
                    if time.time() - start_time > time_limit:
                        return order, current_len
                    # Calculate current overlap contribution of the segment
                    # The segment occupies positions i to i+seg_len-1
                    # Overlaps affected: (i-1, i), (i, i+1), ..., (i+seg_len-2, i+seg_len-1), (i+seg_len-1, i+seg_len)
                    # When we remove and reinsert, these change
                    
                    segment = order[i:i + seg_len]
                    
                    for j in range(num_blocks - seg_len + 1):
                        if time.time() - start_time > time_limit:
                            return order, current_len
                        # Skip if j is too close to i (would result in same or similar order)
                        if j == i or (j < i + seg_len and j >= i - seg_len + 1):
                            continue
                        
                        # Calculate new length efficiently
                        # Remove segment from position i, insert at position j
                        # Overlaps that change:
                        # Old: (i-1, i), (i, i+1), ..., (i+seg_len-1, i+seg_len)
                        # New: depends on new positions
                        
                        new_order = order[:]
                        del new_order[i:i + seg_len]
                        adj_j = j if j <= i else j - seg_len
                        new_order[adj_j:adj_j] = segment
                        
                        new_len = calc_length(new_order)
                        if new_len < current_len:
                            current_len = new_len
                            order = new_order
                            improved = True
                            
            if time.time() - start_time > time_limit:
                return order, current_len
                
            # Neighborhood 2: 2-opt (reverse segment)
            for i in range(num_blocks - 1):
                if time.time() - start_time > time_limit:
                    return order, current_len
                for j in range(i + 1, num_blocks):
                    if time.time() - start_time > time_limit:
                        return order, current_len
                    new_order = order[:]
                    new_order[i:j + 1] = new_order[i:j + 1][::-1]
                    new_len = calc_length(new_order)
                    if new_len < current_len:
                        current_len = new_len
                        order = new_order
                        improved = True
                        
        return order, current_len
    
    best_order, best_len = vns_search(best_order, best_len)
    
    # Additional random restarts if time permits
    if time.time() - start_time < time_limit - 10:
        import random
        for restart in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            random_order = list(range(num_blocks))
            random.shuffle(random_order)
            random_len = calc_length(random_order)
            if random_len < best_len:
                best_order, best_len = vns_search(random_order, random_len)
    
    # Reconstruct the word from best_order
    # Structure: P_1, n, P_1, P_2, n, P_2, ...
    # But with overlaps between consecutive P's
    word = []
    first_block = list(all_perms[best_order[0]])
    word.extend(first_block)
    word.append(n)
    word.extend(first_block)
    
    for i in range(1, len(best_order)):
        prev_block = list(all_perms[best_order[i - 1]])
        curr_block = list(all_perms[best_order[i]])
        
        # Find overlap between suffix of prev_block and prefix of curr_block
        max_k = min(len(prev_block), len(curr_block))
        overlap_len = 0
        for k in range(max_k, 0, -1):
            if prev_block[-k:] == curr_block[:k]:
                overlap_len = k
                break
        
        # Add curr_block, skipping the overlapping part
        word.extend(curr_block[overlap_len:])
        # Add n
        word.append(n)
        # Add curr_block again
        word.extend(curr_block)
    
    return "".join(map(str, word))
