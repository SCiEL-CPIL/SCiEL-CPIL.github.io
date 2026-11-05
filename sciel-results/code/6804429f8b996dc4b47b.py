import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Get the blocks (permutations of 1..n-1) in standard order
    # We need to generate the standard superpermutation for n-1 to get the block order.
    # However, for n=7, n-1=6. 6! = 720 blocks.
    # Generating S_6 takes a bit of time but is fine.
    
    def get_standard_blocks(m):
        # Returns list of blocks (tuples of ints) in the order they appear in standard S_m
        if m == 1:
            return [(1,)]
        
        prev_blocks = get_standard_blocks(m - 1)
        # The standard S_m is constructed by taking each block B from S_{m-1}
        # and forming B, m, B.
        # But we just need the list of (m-1)-permutations that serve as the base for S_m.
        # Wait, the "blocks" for S_n are the (n-1)-permutations.
        # In the standard construction, the order of these blocks is determined by S_{n-1}.
        # Specifically, if S_{n-1} is built from blocks P_1..P_k, then S_n uses P_1..P_k in that order.
        # So get_standard_blocks(m) should return the list of (m-1)-permutations used to build S_m.
        # For m=2, blocks are perms of 1: [(1,)].
        # For m=3, blocks are perms of 1,2 in order from S_2.
        # S_2 = 1 2 1. The (2-1)=1 perms are 1.
        # Wait, the blocks for S_3 are the (3-1)=2 perms.
        # The standard S_3 is 121312321.
        # The 2-perms in order of first appearance: 12, 21.
        # So for m=3, we return [(1,2), (2,1)].
        
        # Let's just generate S_{m-1} and extract the unique (m-1)-perms in order.
        if m == 2:
            return [(1,)]
            
        # Generate S_{m-1}
        # We can reuse the logic from the seed but cleaner.
        word = [1]
        for k in range(2, m):
            # Extract perms of length k-1 from word
            perms = []
            seen = set()
            for i in range(len(word) - (k - 2)):
                win = tuple(word[i:i + k - 1])
                if len(set(win)) == k - 1 and win not in seen:
                    seen.add(win)
                    perms.append(win)
            # Build new word
            new_word = []
            for p in perms:
                block = list(p) + [k] + list(p)
                if not new_word:
                    new_word = block
                else:
                    # Overlap
                    max_k = min(len(new_word), len(block))
                    overlap = 0
                    for o in range(max_k, 0, -1):
                        if new_word[-o:] == block[:o]:
                            overlap = o
                            break
                    new_word += block[overlap:]
            word = new_word
        
        # Now word is S_{m-1}. Extract (m-1)-perms
        blocks = []
        seen = set()
        for i in range(len(word) - (m - 2)):
            win = tuple(word[i:i + m - 1])
            if len(set(win)) == m - 1 and win not in seen:
                seen.add(win)
                blocks.append(win)
        return blocks

    blocks = get_standard_blocks(n)
    N = len(blocks)
    
    # Step 2: Build Overlap Matrix
    # O[i][j] = overlap length between blocks[i] and blocks[j]
    # blocks are tuples of length n-1
    L = n - 1
    overlap = [[0] * N for _ in range(N)]
    for i in range(N):
        bi = blocks[i]
        for j in range(N):
            bj = blocks[j]
            # Max overlap is L-1
            max_o = L - 1
            o = 0
            for k in range(max_o, 0, -1):
                if bi[-k:] == bj[:k]:
                    o = k
                    break
            overlap[i][j] = o
            
    # Step 3: VNS Search
    # Initial order: 0, 1, ..., N-1
    order = list(range(N))
    
    # Calculate initial score (sum of overlaps)
    def get_score(ord):
        s = 0
        for i in range(len(ord) - 1):
            s += overlap[ord[i]][ord[i + 1]]
        return s
        
    best_score = get_score(order)
    
    start_time = time.time()
    time_limit = 100 # Leave 20s for overhead and reconstruction
    
    # Neighborhoods
    # 1. 2-opt (reverse segment)
    # 2. Or-opt-1 (move 1 block)
    # 3. Or-opt-2 (move 2 blocks)
    # 4. Or-opt-3 (move 3 blocks)
    
    def try_2_opt(ord, score):
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # Reverse ord[i:j+1]
                    # Old edges: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    # Internal edges: (k, k+1) becomes (k+1, k)
                    
                    # Calculate delta
                    delta = 0
                    # Remove old internal edges
                    for k in range(i, j):
                        delta -= overlap[ord[k]][ord[k + 1]]
                    # Add new internal edges (reversed)
                    for k in range(i, j):
                        delta += overlap[ord[k + 1]][ord[k]]
                    
                    # Handle boundary edges
                    if i > 0:
                        delta -= overlap[ord[i - 1]][ord[i]]
                        delta += overlap[ord[i - 1]][ord[j]]
                    if j < N - 1:
                        delta -= overlap[ord[j]][ord[j + 1]]
                        delta += overlap[ord[i]][ord[j + 1]]
                        
                    if delta > 0:
                        # Apply
                        ord[i:j + 1] = ord[i:j + 1][::-1]
                        score += delta
                        improved = True
        return ord, score

    def try_or_opt(ord, score, segment_len):
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N - segment_len):
                # Remove segment ord[i : i + segment_len]
                # Old edges:
                # (i-1, i)
                # (i+segment_len-1, i+segment_len)
                
                for j in range(N - segment_len + 1):
                    if j >= i and j < i + segment_len:
                        continue # Same position
                    # Insert at j
                    # If j < i, indices shift
                    # Let's just simulate the change in score carefully
                    
                    # This is complex to do incrementally without modifying the list in place and backtracking.
                    # Given N=720, a simple O(N^2) check with O(1) delta calculation is preferred.
                    # But Or-opt delta calculation is tricky with index shifts.
                    # Alternative: Just do 2-opt first, which is very effective.
                    # If we have time, we can add a simpler Or-opt.
                    
                    pass
        return ord, score

    # Apply 2-opt
    order, best_score = try_2_opt(order, best_score)
    
    # If time remains, try a simple random restart or more 2-opt passes
    # Actually, 2-opt on 720 nodes is fast. Let's run it until convergence.
    
    # Step 4: Reconstruct Word
    # W = B_0, n, B_0
    # For each next B_k:
    #   Find overlap k_o with prev B
    #   Append B_k[k_o:] + [n] + B_k
    
    word = []
    first_block = blocks[order[0]]
    word.extend(first_block)
    word.append(n)
    word.extend(first_block)
    
    prev_block = first_block
    for idx in range(1, N):
        curr_block = blocks[order[idx]]
        # Find overlap
        max_o = L - 1
        o = 0
        for k in range(max_o, 0, -1):
            if prev_block[-k:] == curr_block[:k]:
                o = k
                break
        # Append
        word.extend(curr_block[o:])
        word.append(n)
        word.extend(curr_block)
        prev_block = curr_block
        
    return "".join(map(str, word))
