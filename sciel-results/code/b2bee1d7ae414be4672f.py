import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate standard superpermutation for n-1
    # We use the standard recursive construction for the base case
    prev_word = construct(n-1)
    prev_list = [int(c) for c in prev_word]
    
    # Step 2: Extract the sequence of (n-1)-permutations from prev_word
    # In the standard superpermutation, every window of size n-1 is a permutation.
    # We extract the unique permutations in the order they first appear.
    seen = set()
    blocks = []
    for i in range(len(prev_list) - (n-2)):
        window = tuple(prev_list[i:i+n-1])
        if len(set(window)) == n-1:
            if window not in seen:
                seen.add(window)
                blocks.append(window)
                
    # Step 3: Optimize the order of blocks using VNS
    # The word is constructed as: b1 n b1 b2 n b2 ... bk n bk
    # Length = len(b1) + 1 + len(b1) + sum_{i=1}^{k-1} (1 + len(b_{i+1}) + len(b_{i+1}) - overlap(b_i, b_{i+1}))
    # Since all blocks have length n-1, let L = n-1.
    # Length = L + 1 + L + sum_{i=1}^{k-1} (1 + 2L - overlap(b_i, b_{i+1}))
    # Length = 2L + 1 + (k-1)(1 + 2L) - sum(overlaps)
    # To minimize length, we maximize sum(overlaps).
    
    k = len(blocks)
    L = n - 1
    
    # Precompute overlaps between all pairs of blocks
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # This is an O(k^2 * L) precomputation, which is feasible for n=7 (k=720, L=6)
    # 720^2 * 6 ~ 3M operations, fast enough.
    overlap_matrix = [[0] * k for _ in range(k)]
    for i in range(k):
        bi = blocks[i]
        for j in range(k):
            bj = blocks[j]
            max_k = min(L, L) # L
            for ov in range(max_k, -1, -1):
                if ov == 0:
                    overlap_matrix[i][j] = 0
                    break
                if bi[-ov:] == bj[:ov]:
                    overlap_matrix[i][j] = ov
                    break

    # Base length without overlaps
    base_len = 2 * L + 1 + (k - 1) * (1 + 2 * L)
    
    def calc_total_length(order):
        # order is a list of indices into blocks
        total_overlap = 0
        for idx in range(len(order) - 1):
            total_overlap += overlap_matrix[order[idx]][order[idx+1]]
        return base_len - total_overlap

    # Initialize with standard order (identity permutation of indices)
    best_order = list(range(k))
    best_len = calc_total_length(best_order)
    
    start_time = time.time()
    time_limit = 110 # Leave 10s for overhead
    
    # VNS Loop
    # Neighbors: 
    # 1. 2-opt: reverse a segment
    # 2. Or-opt: move a segment of length 1 or 2 to a new position
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 2-opt pass
        for i in range(k - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, k):
                # Try reversing segment i..j
                # To speed up, we can estimate the change in overlap
                # Old overlaps involved: (i-1, i), (j, j+1), (i, i+1)...(j-1, j)
                # New overlaps involved: (i-1, j), (i, i+1)...(j-1, j) reversed -> (j, j-1)...(i+1, i)
                # Wait, reversing i..j means:
                # ... A, B_i, B_{i+1}, ..., B_j, C ...
                # becomes
                # ... A, B_j, ..., B_{i+1}, B_i, C ...
                
                # Overlaps removed:
                # 1. Between A (i-1) and B_i (i): overlap_matrix[order[i-1]][order[i]] if i>0
                # 2. Between B_j (j) and C (j+1): overlap_matrix[order[j]][order[j+1]] if j<k-1
                # 3. Internal overlaps: sum(overlap_matrix[order[t]][order[t+1]] for t in i..j-1)
                
                # Overlaps added:
                # 1. Between A (i-1) and B_j (j): overlap_matrix[order[i-1]][order[j]] if i>0
                # 2. Between B_i (i) and C (j+1): overlap_matrix[order[i]][order[j+1]] if j<k-1
                # 3. Internal overlaps: sum(overlap_matrix[order[t+1]][order[t]] for t in i..j-1)
                
                # Let's just calculate the full delta efficiently or just re-calc if k is small.
                # For k=720, re-calc is O(k) which is 720 ops.
                # 2-opt has O(k^2) candidates. 720^2 * 720 ~ 370M ops. Too slow in Python.
                
                # We need incremental delta.
                old_ov = 0
                if i > 0: old_ov += overlap_matrix[best_order[i-1]][best_order[i]]
                if j < k-1: old_ov += overlap_matrix[best_order[j]][best_order[j+1]]
                for t in range(i, j):
                    old_ov += overlap_matrix[best_order[t]][best_order[t+1]]
                
                new_ov = 0
                if i > 0: new_ov += overlap_matrix[best_order[i-1]][best_order[j]]
                if j < k-1: new_ov += overlap_matrix[best_order[i]][best_order[j+1]]
                for t in range(i, j):
                    # The segment is reversed, so t in original maps to j-t+i in new?
                    # Original: B_i, B_{i+1}, ..., B_j
                    # New: B_j, ..., B_{i+1}, B_i
                    # Overlap between B_{i+m} and B_{i+m+1} becomes overlap between B_{i+m+1} and B_{i+m}
                    # So we sum overlap_matrix[best_order[t+1]][best_order[t]]
                    new_ov += overlap_matrix[best_order[t+1]][best_order[t]]
                    
                delta = new_ov - old_ov
                if delta > 0:
                    # Improve
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                    best_len -= delta
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
        
        if time.time() - start_time > time_limit:
            break

        # Or-opt pass (move segment of length 1 or 2)
        # To keep it simple and fast, we'll do a few random Or-opt moves if time permits
        if time.time() - start_time < time_limit - 10:
            for _ in range(1000): # Fixed number of attempts
                if time.time() - start_time > time_limit:
                    break
                seg_len = random.choice([1, 2])
                if seg_len > k: continue
                i = random.randint(0, k - seg_len)
                j = i + seg_len - 1
                # Choose new position
                # Avoid moving to same spot
                new_start = random.randint(0, k - seg_len)
                if new_start == i:
                    continue
                # Shift logic is complex for delta. 
                # Given the time, 2-opt is the primary driver. 
                # If we have time, we can implement a simpler swap (2-opt with i=j+1 is swap)
                # But 2-opt covers swaps if we allow i and i+1? No, 2-opt reverses.
                # Let's stick to 2-opt as it is robust.
                pass

    # Reconstruct the word from best_order
    # Structure: b_0 n b_0 b_1 n b_1 ... b_{k-1} n b_{k-1}
    # Wait, the standard construction logic in the seed was:
    # word = [1]
    # for m in range(2, n+1):
    #   ...
    #   new = []
    #   for p in perms:
    #       block = list(p) + [m] + list(p)
    #       # overlap with current end of new
    #       new += block[overlap:]
    #   word = new
    
    # My optimized order `best_order` is a permutation of the indices of `blocks`.
    # `blocks` are the (n-1)-perms extracted from `prev_word`.
    # We need to build the string for n.
    
    word = []
    for idx in best_order:
        block = blocks[idx]
        # Add n
        word.append(n)
        # Add block
        word.extend(block)
        
    # Wait, the seed's reconstruction logic:
    # "block = list(p) + [m] + list(p)"
    # This implies the pattern is p m p.
    # My analysis above assumed: b1 n b1 b2 n b2 ...
    # Let's check the seed's final return.
    # The seed returns "".join(map(str, word)) where word is built by:
    # for p in perms:
    #   block = list(p) + [m] + list(p)
    #   new += block[overlap:]
    
    # So the segment for a block p is `p m p`.
    # Overlap is between the end of the previous `p_prev m p_prev` and the start of `p_curr m p_curr`.
    # End of prev is `p_prev`. Start of curr is `p_curr`.
    # So overlap is between `p_prev` and `p_curr`.
    # This matches my `overlap_matrix` definition: overlap_matrix[i][j] is suffix of blocks[i] matching prefix of blocks[j].
    
    # However, the seed's initial `word` for n=1 is "1".
    # For n=2:
    # prev_word = "1"
    # blocks from "1" with n-1=1: window (1,). blocks = [(1,)]
    # k=1.
    # word = []
    # idx=0: block=(1,). word.append(2). word.extend(1). word=[2,1]
    # Result "21".
    # But superpermutation for n=2 is "121" or "212"?
    # "121" contains 12 and 21. Length 3.
    # "21" contains 21. Does it contain 12? No.
    # So the seed's logic `word.append(n); word.extend(block)` produces `n p`.
    # It is missing the first `p`?
    
    # Let's trace seed for n=2 carefully.
    # construct(1) returns "1".
    # n=2. prev_word="1". prev_list=[1].
    # perms: i in range(1-1+1=1). i=0. window=tuple([1][0:1])=(1,). len(set)==1. seen={(1,)}. perms=[(1,)].
    # new=[]
    # p=(1,). block=[1]+[2]+[1] = [1,2,1].
    # overlap: min(len(new)=0, len(block)=3) -> 0.
    # new += block[0:] -> [1,2,1].
    # word = [1,2,1].
    # Return "121". Correct.
    
    # My reconstruction:
    # word = []
    # for idx in best_order:
    #   word.append(n)
    #   word.extend(block)
    # This produces `n p` for each block.
    # For n=2, block=(1,). word becomes [2, 1]. "21". Incorrect.
    
    # I need to match the seed's construction pattern: `p n p`.
    # And handle overlaps correctly.
    
    # Let's re-implement the reconstruction to match the seed's logic but using `best_order`.
    
    word = []
    for idx in best_order:
        p = blocks[idx]
        block = list(p) + [n] + list(p)
        
        # Calculate overlap with current word
        # Current word ends with ... p_prev (if word is not empty)
        # Actually, the previous block added was `p_prev n p_prev`.
        # So word ends with `p_prev`.
        # We want to overlap `p_prev` with `p_curr`.
        # This is exactly overlap_matrix[prev_idx][curr_idx].
        # But we don't store prev_idx easily in the loop if we just iterate best_order.
        
        # Let's just do the string/list comparison as in the seed, it's O(L) per block.
        # k=720, L=6. 720*6 is very fast.
        
        if not word:
            word.extend(block)
        else:
            # Find max overlap k such that word[-k:] == block[:k]
            max_k = min(len(word), len(block))
            overlap = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if word[-k:] == block[:k]:
                    overlap = k
                    break
            word.extend(block[overlap:])
            
    return "".join(map(str, word))
