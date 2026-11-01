"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    if n == 1:
        return "1"
    
    # Step 1: Build the classic superpermutation for n-1
    # This is fast and provides the base structure and the set of permutations
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
                # Find maximum overlap with the end of the current new word
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
    # Block for perm p is p + [n] + p
    blocks = [list(p) + [n] + list(p) for p in perms]
    block_len = 2 * (n - 1) + 1
    
    # Step 4: Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # We want to maximize sum of overlaps.
    # overlap[i][j] is the length of the longest suffix of blocks[i] that is a prefix of blocks[j].
    
    # Optimization: Use string matching or simple loops.
    # Since block_len is small (13 for n=7), simple loops are fine.
    overlap_mat = [[0] * m for _ in range(m)]
    for i in range(m):
        bi = blocks[i]
        bi_str = "".join(map(str, bi))
        for j in range(m):
            bj = blocks[j]
            bj_str = "".join(map(str, bj))
            # Max possible overlap is block_len - 1
            # Actually, since they are permutations of distinct symbols,
            # the overlap is determined by the suffix/prefix match.
            max_ov = block_len - 1
            ov = 0
            for o in range(max_ov, -1, -1):
                if bi_str[-o:] == bj_str[:o]:
                    ov = o
                    break
            overlap_mat[i][j] = ov

    # Also compute overlap with the end of base_word
    base_str = "".join(map(str, base_word))
    first_ov = [0] * m
    for i in range(m):
        bi_str = "".join(map(str, blocks[i]))
        max_ov = min(len(base_str), block_len - 1)
        for o in range(max_ov, -1, -1):
            if base_str[-o:] == bi_str[:o]:
                first_ov[i] = o
                break

    # Step 5: Local Search to maximize total overlap
    # Initial order: the order in which perms appeared in base_word (recursive order)
    # This is indices 0 to m-1
    order = list(range(m))
    
    def calc_total_overlap(ord):
        total = first_ov[ord[0]]
        for k in range(1, len(ord)):
            total += overlap_mat[ord[k-1]][ord[k]]
        return total

    current_overlap = calc_total_overlap(order)
    
    # 2-Opt Local Search
    # Try reversing segments [i, j]
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for i in range(m - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, m):
                # Check if reversing order[i..j] improves overlap
                # Old segment: ... A, B, C, D ...
                # New segment: ... D, C, B, A ...
                # A = order[i-1] (if i>0 else None), B = order[i], C = order[j], D = order[j+1] (if j<m-1 else None)
                
                # Indices in order list
                idx_a = i - 1
                idx_b = i
                idx_c = j
                idx_d = j + 1
                
                old_ov = 0
                new_ov = 0
                
                # Overlap entering B
                if idx_a >= 0:
                    old_ov += overlap_mat[order[idx_a]][order[idx_b]]
                    new_ov += overlap_mat[order[idx_a]][order[idx_c]]
                else:
                    old_ov += first_ov[order[idx_b]]
                    new_ov += first_ov[order[idx_c]]
                    
                # Overlap inside segment
                # Old: B->C (direct if i==j? No, we are reversing i..j)
                # If i == j, reversal is no-op.
                if i < j:
                    # Old internal overlaps: order[i]->order[i+1] ... order[j-1]->order[j]
                    # New internal overlaps: order[j]->order[j-1] ... order[i+1]->order[i]
                    # We only need to check the boundaries and the immediate neighbors of the reversed segment?
                    # No, the internal overlaps change completely.
                    # Sum of old internal overlaps:
                    for k in range(i, j):
                        old_ov += overlap_mat[order[k]][order[k+1]]
                    # Sum of new internal overlaps:
                    for k in range(i, j):
                        new_ov += overlap_mat[order[k+1]][order[k]]
                        
                # Overlap leaving C/D
                if idx_d < m:
                    old_ov += overlap_mat[order[idx_c]][order[idx_d]]
                    new_ov += overlap_mat[order[idx_b]][order[idx_d]]
                    
                if new_ov > old_ov:
                    # Perform reversal
                    order[i:j+1] = reversed(order[i:j+1])
                    current_overlap = new_ov + (current_overlap - old_ov)
                    improved = True
                    # Restart or continue? Restarting is safer to find global max, 
                    # but continuing is faster. Let's continue.
                    
    # Step 6: Construct the final word
    final_word = list(base_word)
    # Add first block
    first_block_idx = order[0]
    first_block = blocks[first_block_idx]
    # Overlap with base_word
    ov = first_ov[first_block_idx]
    final_word += first_block[ov:]
    
    # Add remaining blocks
    for k in range(1, m):
        prev_idx = order[k-1]
        curr_idx = order[k]
        curr_block = blocks[curr_idx]
        ov = overlap_mat[prev_idx][curr_idx]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))

# EVOLVE-BLOCK-END
