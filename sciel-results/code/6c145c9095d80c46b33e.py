import time
import random
import math
import sys

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Generate the standard recursive superpermutation for n-1
    def get_standard_superperm(m):
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                # Find max overlap
                overlap = 0
                # Theoretical max overlap is len(block)-1 or len(new)-1
                # But practically limited by k-2 for these structures
                # We just check all possible overlaps
                limit = min(len(new), len(block))
                for o in range(limit, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    # Get S_{n-1}
    word_prev = get_standard_superperm(n - 1)
    
    # Extract order of (n-1) permutations from word_prev
    # These are the "blocks" we will arrange
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - (n - 1) + 1):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)
            
    N = len(perms_order)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that suffix of perms_order[i] of length k == prefix of perms_order[j]
    # Max possible overlap is n-2 (since they are permutations of n-1 elements, 
    # if they share n-1 elements, they are identical, but i!=j so max is n-2)
    overlap_matrix = [[0]*N for _ in range(N)]
    max_overlap_len = n - 2
    
    for i in range(N):
        p1 = perms_order[i]
        for j in range(N):
            if i == j:
                continue
            p2 = perms_order[j]
            # Check overlaps from max down to 0
            for k in range(max_overlap_len, -1, -1):
                if k == 0:
                    overlap_matrix[i][j] = 0
                    break
                if p1[n-1-k:] == p2[:k]:
                    overlap_matrix[i][j] = k
                    break

    # Initial order: 0, 1, ..., N-1
    current_order = list(range(N))
    
    def calc_total_overlap(order):
        total = 0
        for i in range(N - 1):
            u = order[i]
            v = order[i+1]
            total += overlap_matrix[u][v]
        return total

    current_overlap = calc_total_overlap(current_order)
    
    # We want to maximize overlap.
    # Length = N * (2n - 1) - total_overlap.
    # Minimizing length is equivalent to maximizing total_overlap.
    
    # Local Search: Hill Climbing with Random Restarts and Swaps
    # Given the 120s limit, we need to be efficient.
    # N=720. 
    # A full pass of 2-opt (reversing segments) is O(N^2) or O(N^3) depending on implementation.
    # Simple swaps are O(N^2) if we check all pairs, but we can do random swaps.
    # Or we can do a "First Improvement" or "Best Improvement" local search on swaps.
    
    # Let's try a combination of:
    # 1. Initial greedy construction? No, standard is already good.
    # 2. Hill climbing with random swaps.
    # 3. If stuck, try reversing a segment (2-opt).
    
    # To speed up, we can precompute for each node its neighbors in the current order?
    # Or just calculate delta for a swap (p, q).
    
    # Delta for swapping positions p and q in order:
    # Let order be ... a, u, b ... c, v, d ...
    # Old cost: cost(a,u) + cost(u,b) + cost(c,v) + cost(v,d)  (using -overlap)
    # New cost: cost(a,v) + cost(v,b) + cost(c,u) + cost(u,d)
    # We want to maximize overlap, so minimize cost.
    # Let's work with overlap directly.
    # Old Overlap: ov(a,u) + ov(u,b) + ov(c,v) + ov(v,d)
    # New Overlap: ov(a,v) + ov(v,b) + ov(c,u) + ov(u,d)
    # Note: if p and q are adjacent, the middle terms are different.
    
    def get_pos(val):
        # This is O(N). We should maintain a position array.
        pass

    # Maintain pos array: pos[val] = index in current_order
    pos = [0] * N
    for i in range(N):
        pos[current_order[i]] = i
    
    # We will perform hill climbing.
    # Time limit: 110s
    start_time = time.time()
    timeout = 110
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        # Randomize the order of checking pairs to avoid local minima bias?
        # Or just iterate through all pairs?
        # N=720, N^2/2 = 259,200 pairs.
        # Checking each pair is O(1) if we have the overlap matrix and pos array.
        # 259k * 110s? No, 259k is fast. Maybe 0.1s per pass?
        # So we can do many passes.
        
        # Let's do a full pass of checking all pairs (i, j) where i < j.
        # If we find an improvement, accept it and restart the pass?
        # Or just continue? "First Improvement" is usually fine.
        
        for i in range(N):
            if time.time() - start_time > timeout:
                break
            u = current_order[i]
            pos_u = i
            # Neighbors of u
            prev_u = current_order[i-1] if i > 0 else -1
            next_u = current_order[i+1] if i < N-1 else -1
            
            for j in range(i+1, N):
                v = current_order[j]
                pos_v = j
                # Neighbors of v
                prev_v = current_order[j-1] if j > 0 else -1
                next_v = current_order[j+1] if j < N-1 else -1
                
                # Calculate old overlap contribution
                old_ov = 0
                if prev_u != -1: old_ov += overlap_matrix[prev_u][u]
                if next_u != -1: old_ov += overlap_matrix[u][next_u]
                if prev_v != -1: old_ov += overlap_matrix[prev_v][v]
                if next_v != -1: old_ov += overlap_matrix[v][next_v]
                
                # Calculate new overlap contribution
                # Swap u and v
                # New neighbors for v (at pos i): prev_u, next_u
                # New neighbors for u (at pos j): prev_v, next_v
                # BUT if i and j are adjacent, the "inner" edge is removed/changed.
                # If j == i+1:
                # Old: ... prev_u, u, v, next_v ...
                # Edges: (prev_u, u), (u, v), (v, next_v)
                # New: ... prev_u, v, u, next_v ...
                # Edges: (prev_u, v), (v, u), (u, next_v)
                
                # If j > i+1:
                # Old: ... prev_u, u, ... , prev_v, v, next_v ...
                # Edges: (prev_u, u), (u, next_u), (prev_v, v), (v, next_v)
                # Note: next_u is the element at i+1. prev_v is the element at j-1.
                
                new_ov = 0
                if j == i + 1:
                    # Adjacent
                    if prev_u != -1: new_ov += overlap_matrix[prev_u][v]
                    else: new_ov += 0
                    # Edge v -> u
                    new_ov += overlap_matrix[v][u]
                    if next_v != -1: new_ov += overlap_matrix[u][next_v]
                    else: new_ov += 0
                else:
                    # Non-adjacent
                    if prev_u != -1: new_ov += overlap_matrix[prev_u][v]
                    else: new_ov += 0
                    if next_u != -1: new_ov += overlap_matrix[v][next_u]
                    else: new_ov += 0
                    
                    if prev_v != -1: new_ov += overlap_matrix[prev_v][u]
                    else: new_ov += 0
                    if next_v != -1: new_ov += overlap_matrix[u][next_v]
                    else: new_ov += 0
                
                if new_ov > old_ov:
                    # Accept swap
                    current_order[i], current_order[j] = v, u
                    pos[u] = j
                    pos[v] = i
                    current_overlap += (new_ov - old_ov)
                    improved = True
                    # Restart inner loop? Or just continue?
                    # Since positions changed, the neighbors for subsequent j's in the same i-loop
                    # might be affected if j was close to i?
                    # Actually, we swapped i and j.
                    # If we continue j from j+1, the element at j is now u.
                    # The element at i is now v.
                    # The loop variable i is fixed.
                    # The element at i is v.
                    # The neighbors of i (v) are prev_u and next_u (which is now u if j=i+1, or original next_u).
                    # This gets messy to update on the fly in a simple double loop.
                    # Safest: break out and restart the pass?
                    # Or just break out of inner loop and restart outer?
                    # Let's just break out and let the while loop handle the next pass.
                    # This is "First Improvement" but with a full restart after each improvement.
                    # This might be slow if there are many small improvements.
                    # But N=720, so passes are fast.
                    break
            
            if improved:
                break
        
        # If no improvement in a full pass, we are at a local optimum for swaps.
        # We could try 2-opt (reversals) here.
        # But let's first see if simple swaps get us close.
        # If we are stuck, we might need to try random restarts or different move sets.
        
        # To avoid getting stuck in a local minimum that requires a sequence of non-improving swaps,
        # we could add a random restart mechanism or simulated annealing.
        # Given the time budget, let's stick to simple hill climbing with restarts if stuck.
        
        # Actually, a full pass of all pairs is O(N^2).
        # If we find an improvement, we break and restart.
        # This is effectively "First Improvement" with restarts.
        # This is robust.

    # Construct the final word
    word = []
    for idx in range(N):
        p = perms_order[current_order[idx]]
        block = list(p) + [n] + list(p)
        if not word:
            word = block
        else:
            # Find overlap
            max_k = n - 2
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    break
                if word[-k:] == block[:k]:
                    found = k
                    break
            word += block[found:]
            
    return "".join(map(str, word))
