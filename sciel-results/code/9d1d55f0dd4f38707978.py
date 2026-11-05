import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate the standard superpermutation for n-1 to get the blocks
    # This is the base for our optimization. We assume the standard recursive construction
    # for n-1 is sufficient to define the set of blocks and their natural order.
    prev_word = construct(n-1)
    prev_list = [int(c) for c in prev_word]
    
    # Step 2: Extract the sequence of (n-1)-permutations (blocks) from prev_word
    # In the standard construction, every window of size n-1 is a permutation, but they overlap.
    # We extract the unique permutations in the order they first appear.
    seen = set()
    blocks = []
    for i in range(len(prev_list) - (n-2)):
        window = tuple(prev_list[i:i+n-1])
        if len(set(window)) == n-1:
            if window not in seen:
                seen.add(window)
                blocks.append(window)
                
    num_blocks = len(blocks)
    block_len = n - 1
    
    # Precompute overlaps between all pairs of blocks
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # This allows O(1) cost calculation for adjacent blocks in an ordering.
    # We store the overlap length.
    overlap_matrix = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            # Since blocks are permutations of distinct digits, max overlap is usually small
            # unless they share a long suffix/prefix.
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    overlap_matrix[i][j] = k
                    break

    # Cost of a sequence of blocks B = [b1, b2, ..., bk]
    # The word is constructed as: b1, n, b1, b2, n, b2, ...
    # Wait, the standard construction logic in the seed was:
    # word = b1 n b1 b2 n b2 ...
    # Let's trace the length addition carefully.
    # Start with b1. Length = block_len.
    # Add n. Length += 1.
    # Add b1. Length += block_len.
    # Total so far: 2*block_len + 1.
    # Next block b2.
    # We append n. Length += 1.
    # We append b2. But we can overlap the suffix of the previous part with b2?
    # The previous part ends with b1.
    # The next part starts with n, then b2?
    # No, the structure is: ... b1, n, b1, n, b2 ... ?
    # Let's re-read the standard construction logic from the seed's fallback:
    # "new = []
    #  for p in perms:
    #      block = list(p) + [m] + list(p)
    #      overlap = ...
    #      new += block[overlap:]"
    #
    # This implies the segments are: S_i = p_i + [m] + p_i.
    # We concatenate S_1, S_2, ..., S_k with overlaps.
    # Overlap between S_i and S_{i+1}:
    # S_i ends with p_i.
    # S_{i+1} starts with p_{i+1}.
    # So the overlap is between p_i (suffix of S_i) and p_{i+1} (prefix of S_{i+1}).
    #
    # Let's verify the length formula:
    # L = len(S_1) + sum_{i=1}^{k-1} (len(S_{i+1}) - overlap(p_i, p_{i+1}))
    # len(S_i) = 2*block_len + 1.
    # So L = (2*block_len + 1) + sum_{i=1}^{k-1} (2*block_len + 1 - overlap_matrix[p_i_idx][p_{i+1}_idx])
    # L = k*(2*block_len + 1) - sum_{i=1}^{k-1} overlap_matrix[p_i_idx][p_{i+1}_idx]
    
    base_segment_len = 2 * block_len + 1
    
    def calc_length(order):
        # order is a list of indices into blocks
        if not order:
            return 0
        total = base_segment_len
        for i in range(len(order) - 1):
            idx1 = order[i]
            idx2 = order[i+1]
            total += base_segment_len - overlap_matrix[idx1][idx2]
        return total

    # Initial order: standard order (0 to num_blocks-1)
    current_order = list(range(num_blocks))
    current_len = calc_length(current_order)
    
    start_time = time.time()
    time_limit = 110 # Leave 10s for overhead and final reconstruction
    
    # VNS / Local Search
    # We will use 2-opt (reverse segment) and Or-opt (move segment to another position)
    # Since we need to strictly reduce length, we accept only improvements.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 2-opt: Reverse a segment [i, j]
        # We can optimize the delta calculation to avoid full recalculation,
        # but for N=720, a full recalculation is O(N) which is ~720 ops.
        # 720*720/2 ~ 250k moves per pass. 250k * 720 ops ~ 180M ops.
        # Python might be slow for 180M ops in 110s? 
        # 110s / 180M ops ~ 600ns/op. Python is typically ~50-100ns for simple ops,
        # but list slicing and function calls are slower.
        # Let's try to be smarter or limit the search space.
        
        # Alternative: Calculate delta for 2-opt [i, j]
        # Old cost contribution around i-1, i ... j, j+1:
        # Old: ... overlap(i-1, i) + sum_{k=i}^{j-1} overlap(k, k+1) + overlap(j, j+1)
        # New: ... overlap(i-1, j) + sum_{k=i}^{j-1} overlap(j-k+i, j-k+i-1) + overlap(i, j+1)
        # Note: The reversed segment is j, j-1, ..., i.
        # So the new adjacencies inside are (j, j-1), (j-1, j-2), ..., (i+1, i).
        # The overlap matrix is directed. overlap(A, B) != overlap(B, A) generally.
        
        def get_delta_2opt(i, j):
            # Reversing order[i...j]
            if i == j: return 0
            
            # Indices in current_order
            # Let the segment be indices i to j in the list `current_order`
            # Let the block indices be b_i = current_order[i], ..., b_j = current_order[j]
            
            # Old overlaps involved:
            # 1. If i > 0: overlap(current_order[i-1], current_order[i])
            # 2. If j < N-1: overlap(current_order[j], current_order[j+1])
            # 3. Internal: overlap(current_order[k], current_order[k+1]) for k in i..j-1
            
            old_delta = 0
            if i > 0:
                old_delta -= overlap_matrix[current_order[i-1]][current_order[i]]
            if j < num_blocks - 1:
                old_delta -= overlap_matrix[current_order[j]][current_order[j+1]]
            for k in range(i, j):
                old_delta -= overlap_matrix[current_order[k]][current_order[k+1]]
                
            new_delta = 0
            if i > 0:
                # New neighbor before segment is current_order[i-1], new first is current_order[j]
                new_delta -= overlap_matrix[current_order[i-1]][current_order[j]]
            if j < num_blocks - 1:
                # New neighbor after segment is current_order[j+1], new last is current_order[i]
                new_delta -= overlap_matrix[current_order[i]][current_order[j+1]]
            
            # Internal overlaps in reversed segment
            # The sequence is b_j, b_{j-1}, ..., b_i
            # Pairs: (b_j, b_{j-1}), (b_{j-1}, b_{j-2}), ..., (b_{i+1}, b_i)
            for k in range(j, i, -1):
                # Next in reversed sequence is k-1
                new_delta -= overlap_matrix[current_order[k]][current_order[k-1]]
                
            return new_delta - old_delta

        # Try 2-opt moves
        for i in range(num_blocks):
            for j in range(i+1, num_blocks):
                delta = get_delta_2opt(i, j)
                if delta < 0:
                    # Apply move
                    current_order[i:j+1] = reversed(current_order[i:j+1])
                    current_len += delta
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
            if time.time() - start_time > time_limit:
                break

    # Reconstruct the word from current_order
    # word = S_1 + S_2[overlap:] + ...
    # S_i = blocks[idx] + [n] + blocks[idx]
    
    word = []
    for k, idx in enumerate(current_order):
        block = blocks[idx]
        segment = list(block) + [n] + list(block)
        
        if k == 0:
            word = segment
        else:
            # Find overlap between end of current word and start of segment
            # Current word ends with blocks[prev_idx]
            # Segment starts with blocks[idx]
            # Overlap is overlap_matrix[prev_idx][idx]
            prev_idx = current_order[k-1]
            ov = overlap_matrix[prev_idx][idx]
            word.extend(segment[ov:])
            
    return "".join(map(str, word))
