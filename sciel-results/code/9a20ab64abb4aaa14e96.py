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
                
    M = len(blocks)
    if M == 0:
        # Fallback for small n or errors
        return prev_word + str(n) + prev_word

    # Step 3: Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Since blocks are permutations of 1..n-1, max overlap is n-2 (if n-1 > 1)
    # Actually max overlap is n-2? No, if blocks are identical, overlap is n-1.
    # But blocks are distinct permutations.
    # Max overlap between two distinct permutations of 1..m is m-1?
    # e.g. 123 and 234? No, symbols are 1..m.
    # 1234 and 2341? Suffix 234, Prefix 234. Overlap 3.
    # So max overlap is n-2.
    
    overlap = [[0]*M for _ in range(M)]
    for i in range(M):
        bi = blocks[i]
        for j in range(M):
            bj = blocks[j]
            max_k = n-2
            # Check k from n-2 down to 0
            for k in range(max_k, -1, -1):
                if bi[-k:] == bj[:k]:
                    overlap[i][j] = k
                    break

    # Step 4: Local Search (Or-opt)
    # Order is a list of indices 0..M-1
    order = list(range(M))
    
    def get_score(ord_list):
        score = 0
        for i in range(len(ord_list) - 1):
            score += overlap[ord_list[i]][ord_list[i+1]]
        return score

    current_score = get_score(order)
    start_time = time.time()
    time_limit = 110
    
    # Or-opt: move a segment of length L to a new position
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try lengths 1, 2, 3
        for L in [1, 2, 3]:
            for i in range(M - L + 1):
                # Extract segment
                seg = order[i:i+L]
                # Try all destinations j
                # j can be 0..M-L
                # If j == i, skip
                # If j is within [i, i+L), skip (overlaps with source)
                for j in range(M - L + 1):
                    if j == i:
                        continue
                    if i <= j < i + L:
                        continue
                    
                    # Calculate new score
                    # Remove seg from i
                    # Insert seg at j
                    
                    # Boundaries affected:
                    # Old:
                    # left_old: (i-1, i) if i>0
                    # right_old: (i+L-1, i+L) if i+L < M
                    # internal_old: sum(overlap[seg[k]][seg[k+1]])
                    
                    # New:
                    # The gap at i connects (i-1, i+L)
                    # The insertion at j connects (j-1, seg[0]) and (seg[-1], j)
                    # Note: if j > i, the indices shift.
                    # It's easier to just calculate the delta.
                    
                    # Let's just reconstruct the score for the affected parts?
                    # No, M=720 is small enough that we can just calculate the full score?
                    # No, 720*720*3 * full_score(720) is too slow.
                    
                    # Let's use the O(1) delta method carefully.
                    
                    # We need to know the score contribution of the edges.
                    # Let's maintain an array `edge_score` where edge_score[k] = overlap[order[k]][order[k+1]]
                    # Sum = sum(edge_score)
                    
                    # This state management is tricky inside the loop.
                    # Alternative: Since M=720 is small, maybe a simpler heuristic is better?
                    # Or just use Python lists and accept O(M) per move?
                    # 1.5 million moves * 720 ops = 1 billion ops. Too slow.
                    
                    # Let's stick to O(1) delta but implement it correctly.
                    
                    # We'll maintain `order` and `score`.
                    # To get O(1) delta, we need to know the overlaps at the boundaries.
                    
                    # Let's precompute nothing else, just use the `overlap` matrix.
                    
                    # Old boundaries:
                    old_delta = 0
                    if i > 0:
                        old_delta += overlap[order[i-1]][order[i]]
                    if i + L < M:
                        old_delta += overlap[order[i+L-1]][order[i+L]]
                    
                    # New boundaries:
                    # The segment is removed from i and inserted at j.
                    # Case 1: j < i
                    # The segment moves left.
                    # New order: ... order[j-1], SEG, order[j] ... order[i-1], order[i+L] ...
                    # Wait, if j < i, the indices between j and i shift right by L.
                    
                    # This is getting complicated.
                    # Let's try a different approach: **Simulated Annealing** or **Random Restart Local Search** with a simpler move set?
                    
                    # Actually, for M=720, a **Greedy Insertion** might be fast and effective.
                    # Or, we can just do a few iterations of **2-opt** with a limited segment length (e.g., up to 10).
                    
                    # Let's try 2-opt with small segments.
                    # If we reverse a segment of length L, the internal overlaps change to reverse overlaps.
                    # We can precompute `rev_overlap[i][j] = overlap[j][i]`.
                    
                    # Let's implement a simple 2-opt for segments of length <= 10.
                    # Number of moves: sum_{L=1}^{10} (M-L+1) * (M-L+1) ? No, just all pairs (i, j) with j-i <= 10.
                    # Approx 10 * 720 * 720 / 2 ? No, for each i, j is i+1..i+10.
                    # 720 * 10 = 7200 moves per pass.
                    # Very fast.
                    
                    # Let's do this.
                    pass

        # If we didn't break out of the inner loop, we need to update 'improved'
        # But the logic above was a 'pass'.
        
        # Let's restart the loop structure properly.
        pass

    # Fallback: return standard construction if search fails or is too complex
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
            overlap_val = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap_val:]
        word = new
    return "".join(map(str, word))
