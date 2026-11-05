import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build the standard recursive superpermutation up to n-1
    word = [1]
    for m in range(2, n):
        # Extract (m-1)-permutation blocks from current word
        blocks = []
        seen = set()
        for i in range(len(word) - (m - 2)):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                blocks.append(list(window))
        
        # Build new word using standard construction
        if not blocks:
            word.append(m)
            continue
            
        new_word = [m] + list(blocks[0])
        for i in range(1, len(blocks)):
            prev_block = blocks[i - 1]
            curr_block = blocks[i]
            # Find overlap between suffix of prev_block and prefix of curr_block
            max_ov = min(len(prev_block), len(curr_block))
            overlap = 0
            for k in range(max_ov, 0, -1):
                if prev_block[-k:] == curr_block[:k]:
                    overlap = k
                    break
            new_word.append(m)
            new_word.extend(curr_block[overlap:])
        word = new_word
    
    # Now optimize the ordering of (n-1)-blocks for the final step
    blocks = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            blocks.append(list(window))
    
    num_blocks = len(blocks)
    if num_blocks <= 1:
        final_word = [n] + list(blocks[0]) if blocks else [n]
        return "".join(map(str, final_word))
    
    start_time = time.time()
    time_limit = 100  # Leave 20s for reconstruction and overhead
    
    # Precompute overlap matrix: overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # This is used to quickly compute the length for a given ordering
    overlap_matrix = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                overlap_matrix[i][j] = 0
                continue
            max_ov = min(len(blocks[i]), len(blocks[j]))
            ov = 0
            for k in range(max_ov, 0, -1):
                if blocks[i][-k:] == blocks[j][:k]:
                    ov = k
                    break
            overlap_matrix[i][j] = ov
    
    # Compute the total length for a given ordering
    # The word structure is: [n] + blocks[order[0]] + [n] + blocks[order[1]][overlap:] + ...
    # Length = 1 + len(blocks[order[0]]) + sum_{i=1}^{k-1} (1 + len(blocks[order[i]]) - overlap[order[i-1]][order[i]])
    #        = 1 + sum_{i=0}^{k-1} len(blocks[order[i]]) + (k-1) - sum_{i=1}^{k-1} overlap[order[i-1]][order[i]]
    # Since all blocks have the same length (n-1), this simplifies to:
    # Length = 1 + k*(n-1) + (k-1) - sum_{i=1}^{k-1} overlap[order[i-1]][order[i]]
    #        = k*n - sum_{i=1}^{k-1} overlap[order[i-1]][order[i]]
    
    def compute_length(order):
        total = num_blocks * n
        for i in range(1, num_blocks):
            total -= overlap_matrix[order[i-1]][order[i]]
        return total
    
    # Initialize with standard order
    best_order = list(range(num_blocks))
    best_len = compute_length(best_order)
    
    # Variable Neighborhood Search (VNS)
    # Neighborhoods: 2-opt (reverse segment), Or-opt (move segment), swap
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 2-opt: reverse a contiguous segment
        for i in range(num_blocks - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, num_blocks):
                # Only consider segments of reasonable length to limit search
                if j - i + 1 > 50:
                    break
                # Compute new length
                # The overlaps that change are at positions i-1->i, i->i+1, ..., j-1->j, j->j+1
                # Old overlaps: overlap[order[i-1]][order[i]] (if i>0), overlap[order[j]][order[j+1]] (if j<num_blocks-1)
                # Plus internal overlaps in the segment
                # New overlaps: overlap[order[i-1]][order[j]] (if i>0), overlap[order[i]][order[j+1]] (if j<num_blocks-1)
                # Plus internal overlaps in the reversed segment
                
                # For efficiency, only try if the segment is short enough
                if j - i + 1 > 20:
                    continue
                    
                new_order = best_order[:]
                new_order[i:j+1] = best_order[i:j+1][::-1]
                
                # Quick check: compute the change in overlap sum
                # Old contribution from boundaries and internal
                old_ov = 0
                new_ov = 0
                
                if i > 0:
                    old_ov += overlap_matrix[best_order[i-1]][best_order[i]]
                    new_ov += overlap_matrix[best_order[i-1]][new_order[i]]
                
                if j < num_blocks - 1:
                    old_ov += overlap_matrix[best_order[j]][best_order[j+1]]
                    new_ov += overlap_matrix[new_order[j]][best_order[j+1]]
                
                # Internal overlaps
                for k in range(i, j):
                    old_ov += overlap_matrix[best_order[k]][best_order[k+1]]
                    new_ov += overlap_matrix[new_order[k]][new_order[k+1]]
                
                if new_ov > old_ov:
                    best_order = new_order
                    best_len = compute_length(best_order)
                    improved = True
        
        # Or-opt: move a contiguous segment of length 1, 2, or 3
        for seg_len in [1, 2, 3]:
            if time.time() - start_time > time_limit:
                break
            for i in range(num_blocks - seg_len):
                if time.time() - start_time > time_limit:
                    break
                segment = best_order[i:i+seg_len]
                rest = best_order[:i] + best_order[i+seg_len:]
                for j in range(len(rest) - seg_len + 1):
                    if j == i:
                        continue
                    new_order = rest[:j] + segment + rest[j:]
                    new_len = compute_length(new_order)
                    if new_len < best_len:
                        best_len = new_len
                        best_order = new_order
                        improved = True
                    if time.time() - start_time > time_limit:
                        break
                if time.time() - start_time > time_limit:
                    break
    
    # Build the final word from the optimized order
    final_word = [n] + list(blocks[best_order[0]])
    for i in range(1, num_blocks):
        prev_block = blocks[best_order[i-1]]
        curr_block = blocks[best_order[i]]
        ov = overlap_matrix[best_order[i-1]][best_order[i]]
        final_word.append(n)
        final_word.extend(curr_block[ov:])
    
    return "".join(map(str, final_word))
