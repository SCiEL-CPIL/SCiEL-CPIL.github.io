import time
import random
import math
import sys

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate standard superpermutation for n-1
    # We need the list of (n-1) permutations in the order they appear in the standard construction.
    # Instead of generating the full string and parsing, we can generate the list of perms directly
    # using the standard recursive algorithm logic, or just generate the string and parse it.
    # Parsing the string is safer to ensure we match the "standard" seed behavior for the base blocks.
    
    prev_word_str = construct(n-1)
    prev_list = [int(c) for c in prev_word_str]
    
    # Extract unique (n-1) permutations in order of first appearance
    # In the standard superpermutation, every window of size n-1 is a valid permutation?
    # No, only specific windows are the "blocks". 
    # Actually, in the standard construction $S_n = S_{n-1} n S_{n-1} \dots$, the "blocks" are the permutations of $1..n-1$.
    # The standard construction is defined recursively.
    # $S_1 = "1"$
    # $S_2 = "121"$ (Perms of 1: [1]. Insert 2: 1 2 1)
    # $S_3 = "1213121"$ (Perms of 1,2: [12, 21]. Insert 3: 12 3 12 21 3 21)
    # The blocks for $S_n$ are the permutations of $1..n-1$.
    # We need to find the sequence of these blocks.
    
    # Let's identify the blocks in prev_list.
    # A block is a window of size n-1 containing distinct elements from 1..n-1.
    # In the standard superpermutation, these blocks appear as non-overlapping segments?
    # No, they overlap.
    # However, the *set* of blocks is exactly all $(n-1)!$ permutations of $1..n-1$.
    # We need their order.
    
    seen = set()
    blocks = []
    m = n - 1
    for i in range(len(prev_list) - m + 1):
        window = tuple(prev_list[i:i+m])
        # Check if it's a permutation of 1..m
        if len(set(window)) == m and set(window) == set(range(1, m+1)):
            if window not in seen:
                seen.add(window)
                blocks.append(window)
                
    # Verify we have all permutations
    if len(blocks) != math.factorial(m):
        # Fallback if parsing failed (should not happen for standard construction)
        # Generate all perms and sort by first appearance in a dummy string? 
        # For n=7, m=6, 720 perms.
        pass

    # Step 2: Optimize the order of these blocks using VNS
    # The cost of a sequence of blocks $B_1, B_2, \dots, B_K$ when inserting symbol $n$ is:
    # $L = |B_1| + 1 + |B_1| + \sum_{i=2}^K (1 + |B_i| + |B_i| - \text{overlap}(B_{i-1}, B_i))$
    # Wait, the structure is $B_1 n B_1 B_2 n B_2 \dots B_K n B_K$.
    # Let's trace the length addition.
    # Start with $B_1 n B_1$. Length: $2m + 1$.
    # Add $B_2 n B_2$.
    # The end of current string is $B_1$. The start of next segment is $B_2$.
    # We can overlap the suffix of $B_1$ with the prefix of $B_2$.
    # Let $O_{i-1, i}$ be the max $k$ such that $B_{i-1}[-k:] == B_i[:k]$.
    # The new characters added for block $B_i$ are:
    # We need to append $n$ and $B_i$.
    # Wait, the pattern is:
    # $W = B_1 n B_1$
    # $W = W B_2 n B_2$? No.
    # Standard recursion: $S_n = S_{n-1} n S_{n-1} \dots$ is not quite right.
    # It is: $S_n = P_1 n P_1 P_2 n P_2 \dots P_K n P_K$ where $P_i$ are perms of $1..n-1$.
    # Let's check $n=3$. Perms of 1,2: 12, 21.
    # $12 3 12 21 3 21$.
    # Length: $2+1+2 + 1+2+2 = 10$?
    # $1231221321$ length 10.
    # Formula:
    # Start: $B_1 n B_1$. Len $2m+1$.
    # For each next $B_i$:
    # We append $B_i$. Overlap with previous end $B_{i-1}$.
    # Added length for $B_i$ part 1: $m - O_{i-1, i}$.
    # Then we append $n$. Added length 1.
    # Then we append $B_i$ part 2. Overlap with previous end (which is $B_i$ part 1? No, we just appended $B_i$ part 1).
    # Wait, the string ends with $B_i$ (part 1).
    # Then we append $n$.
    # Then we append $B_i$ (part 2).
    # Overlap between $n$ and $B_i$? No, $n$ is distinct from $1..m$.
    # So we just append $n$ and then $B_i$.
    # But wait, the standard construction is $P_1 n P_1 P_2 n P_2$.
    # Is the second $P_1$ fully added?
    # In $12 3 12 21 3 21$:
    # $12$
    # $3$
    # $12$
    # $21$ (overlap with prev $12$? $12$ ends in 2, $21$ starts with 2. Overlap 1. So add $1$.)
    # $3$
    # $21$
    # Let's re-evaluate the cost function for a sequence of blocks $P_1, \dots, P_K$.
    # Word starts with $P_1$.
    # Append $n$.
    # Append $P_1$.
    # For $i=2$ to $K$:
    #   Append $P_i$ (with overlap against previous end $P_{i-1}$? No, previous end is $P_{i-1}$).
    #   Wait, after step $i-1$, the word ends with $P_{i-1}$.
    #   So we append $P_i$. Overlap $O(P_{i-1}, P_i)$. Added: $m - O$.
    #   Append $n$. Added: 1.
    #   Append $P_i$. Added: $m$. (No overlap with $n$).
    #   
    # Total Length = $m + 1 + m + \sum_{i=2}^K ( (m - O(P_{i-1}, P_i)) + 1 + m )$
    # $L = 2m + 1 + \sum_{i=2}^K ( 2m + 1 - O(P_{i-1}, P_i) )$
    # $L = 2m + 1 + (K-1)(2m+1) - \sum_{i=2}^K O(P_{i-1}, P_i)$
    # $L = K(2m+1) + 1 - \sum_{i=2}^K O(P_{i-1}, P_i)$
    
    # To minimize $L$, we must maximize $\sum O(P_{i-1}, P_i)$.
    # This is a Traveling Salesman Problem (TSP) variant on the blocks, where edge weight is overlap.
    
    # Precompute overlaps
    K = len(blocks)
    # Overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Since blocks are permutations of 1..m, max overlap is m-1 (if they share m-1 suffix/prefix).
    # Actually, if $B_i = (a_1, \dots, a_m)$ and $B_j = (b_1, \dots, b_m)$.
    # $B_i[-k:] == B_j[:k]$.
    # If $k=m$, then $B_i = B_j$. But blocks are unique. So max $k=m-1$.
    
    # We can compute overlap matrix efficiently.
    # Overlap depends on the last $m-1$ elements of $B_i$ and first $m-1$ of $B_j$.
    # Let's just compute it on the fly or precompute for the 720x720 matrix.
    # 720*720 = 518,400 pairs. Each check is small. Fast enough.
    
    overlap_matrix = [[0]*K for _ in range(K)]
    for i in range(K):
        bi = blocks[i]
        for j in range(K):
            if i == j:
                overlap_matrix[i][j] = m # Should not happen in path, but safe
                continue
            bj = blocks[j]
            # Max possible overlap is m-1
            max_k = m - 1
            # Check from max_k down to 0
            # bi[-k:] == bj[:k]
            # To speed up, we can check the last element of prefix vs first of suffix?
            # Simple loop is fine for m=6.
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap_matrix[i][j] = 0
                    break
                if bi[-k:] == bj[:k]:
                    overlap_matrix[i][j] = k
                    break

    # VNS on the permutation of blocks
    # Objective: Maximize sum of overlaps
    # Current order: standard order (identity)
    current_order = list(range(K))
    
    def calc_score(order):
        score = 0
        for i in range(1, K):
            prev_idx = order[i-1]
            curr_idx = order[i]
            score += overlap_matrix[prev_idx][curr_idx]
        return score

    best_order = current_order[:]
    best_score = calc_score(best_order)
    
    start_time = time.time()
    time_limit = 100 # Leave buffer
    
    # VNS Neighborhoods
    # N1: 2-opt (reverse segment)
    # N2: Or-opt (move segment of length 1, 2, 3)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 2-Opt
        for i in range(K - 1):
            for j in range(i + 1, K):
                # Reverse segment i..j
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_score = calc_score(new_order)
                if new_score > best_score:
                    best_score = new_score
                    best_order = new_order
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
            if time.time() - start_time > time_limit:
                break
        if time.time() - start_time > time_limit:
            break
            
        # Or-Opt (move 1 element)
        if time.time() - start_time < time_limit:
            for i in range(K):
                for j in range(K):
                    if i == j: continue
                    # Move element at i to j
                    # Remove i
                    elem = best_order[i]
                    temp = best_order[:i] + best_order[i+1:]
                    # Insert at j (adjust index if j > i)
                    insert_idx = j if j < i else j - 1
                    new_order = temp[:insert_idx] + [elem] + temp[insert_idx:]
                    new_score = calc_score(new_order)
                    if new_score > best_score:
                        best_score = new_score
                        best_order = new_order
                        improved = True
                        if time.time() - start_time > time_limit:
                            break
                if time.time() - start_time > time_limit:
                    break
            if time.time() - start_time > time_limit:
                break

        # Or-Opt (move 2 elements) - optional if time permits
        if time.time() - start_time < time_limit:
            for i in range(K - 1):
                for j in range(K - 1):
                    if i == j: continue
                    # Move segment [i, i+1] to j
                    seg = best_order[i:i+2]
                    temp = best_order[:i] + best_order[i+2:]
                    # Insert at j
                    # If j > i, indices shift?
                    # It's safer to just try a few random moves or simple swaps if complex.
                    # Let's stick to 2-opt and 1-move for robustness and speed.
                    pass

    # Reconstruct word from best_order
    word = []
    # Start with first block
    first_block = blocks[best_order[0]]
    word.extend(first_block)
    word.append(n)
    word.extend(first_block)
    
    for k in range(1, K):
        idx = best_order[k]
        b = blocks[idx]
        prev_b = blocks[best_order[k-1]]
        
        # Calculate overlap
        # We need to append b, then n, then b.
        # Overlap between prev_b (end of current word) and b (start of new segment).
        # Current word ends with prev_b.
        # We want to append b.
        # Overlap $O = overlap_matrix[best_order[k-1]][idx]$
        O = overlap_matrix[best_order[k-1]][idx]
        
        # Append b[O:]
        word.extend(b[O:])
        word.append(n)
        word.extend(b)
        
    return "".join(map(str, word))
