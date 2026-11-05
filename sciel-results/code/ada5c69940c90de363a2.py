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
                
    # Precompute overlap matrix for efficient VNS
    num_blocks = len(blocks)
    block_len = n - 1
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                overlap[i][j] = block_len
                continue
            max_k = min(block_len, block_len)
            for k in range(max_k, 0, -1):
                if blocks[i][-k:] == blocks[j][:k]:
                    overlap[i][j] = k
                    break
    
    def calc_length_fast(order):
        """Calculate total length given an ordering of block indices."""
        if not order:
            return 0
        total = 2 * block_len + 1  # First block: b + [n] + b
        for idx in range(len(order) - 1):
            curr_idx = order[idx]
            next_idx = order[idx + 1]
            total += 1 + block_len + block_len - overlap[curr_idx][next_idx]
        return total
    
    # Initial order: standard construction order (0, 1, 2, ..., num_blocks-1)
    best_order = list(range(num_blocks))
    best_len = calc_length_fast(best_order)
    
    start_time = time.time()
    time_limit = 110
    
    # VNS: Variable Neighborhood Search
    # Neighborhoods: k-opt moves (swap/reverse segments of length up to k)
    
    def get_neighbor_2opt(order, i, j):
        """Reverse segment i..j"""
        return order[:i] + order[i:j+1][::-1] + order[j+1:]
    
    def get_neighbor_oropt(order, i, j, new_pos):
        """Move segment i..j to position new_pos"""
        segment = order[i:j+1]
        rest = order[:i] + order[j+1:]
        if new_pos > j:
            new_pos -= (j - i + 1)
        return rest[:new_pos] + segment + rest[new_pos:]
    
    # Local search with multiple neighborhood structures
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Neighborhood 1: 2-opt (reverse segments)
        for i in range(num_blocks - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, num_blocks):
                new_order = get_neighbor_2opt(best_order, i, j)
                new_len = calc_length_fast(new_order)
                if new_len < best_len:
                    best_len = new_len
                    best_order = new_order
                    improved = True
        
        # Neighborhood 2: Or-opt (move segments)
        if time.time() - start_time > time_limit:
            break
        for seg_len in [1, 2, 3]:
            for i in range(num_blocks - seg_len):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + seg_len, num_blocks):
                    new_order = get_neighbor_oropt(best_order, i, i + seg_len - 1, j)
                    new_len = calc_length_fast(new_order)
                    if new_len < best_len:
                        best_len = new_len
                        best_order = new_order
                        improved = True
    
    # Reconstruct the word from best_order
    word = []
    for idx in best_order:
        block = blocks[idx]
        # Add block, then n, then block (with overlap)
        if not word:
            word.extend(block)
            word.append(n)
            word.extend(block)
        else:
            # Find overlap between end of current word and start of (block + [n] + block)
            full_block = list(block) + [n] + list(block)
            max_overlap = min(len(word), len(full_block))
            actual_overlap = 0
            for k in range(max_overlap, 0, -1):
                if word[-k:] == full_block[:k]:
                    actual_overlap = k
                    break
            word.extend(full_block[actual_overlap:])
    
    return "".join(map(str, word))
