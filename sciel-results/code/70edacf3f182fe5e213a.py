import math
import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate standard superpermutation up to n-1
    if n <= 1:
        return "1"
        
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if time.time() > deadline - 30:
        # Not enough time for optimization, return standard
        return "".join(map(str, word))

    # Now optimize the last step (m=n)
    # Get perms of n-1 from word
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    
    # Precompute overlaps
    # overlaps[i][j] is the max overlap of block_i suffix and block_j prefix
    # Block i is P_i + [n] + P_i
    # We only need to store the overlap value.
    # Note: The overlap is determined by suffix of P_i and prefix of P_j.
    # Actually, the block is P_i, n, P_i.
    # Suffix of Block_i: ... P_i
    # Prefix of Block_j: P_j ...
    # Overlap k means suffix of Block_i of length k == prefix of Block_j of length k.
    
    overlaps = [[0]*num_perms for _ in range(num_perms)]
    blocks = []
    for i in range(num_perms):
        p_i = perms[i]
        block_i = list(p_i) + [n] + list(p_i)
        blocks.append(block_i)
        
    for i in range(num_perms):
        block_i = blocks[i]
        for j in range(num_perms):
            if i == j: continue
            block_j = blocks[j]
            max_o = min(len(block_i), len(block_j))
            o = 0
            # Optimization: Only check up to 2*(n-1) ?
            # Max possible overlap is 2*(n-1) because the middle 'n' breaks symmetry?
            # Actually, if P_i ends with X and P_j starts with X, and X contains n? No, P_i, P_j are perms of 1..n-1.
            # So 'n' is only in the middle.
            # Overlap can span the 'n' in block_i?
            # Suffix of block_i: ... n, P_i.
            # Prefix of block_j: P_j, n ...
            # If overlap is large, it might include the 'n' from block_i matching a symbol in P_j? No, P_j doesn't have n.
            # So the 'n' in block_i cannot match any symbol in block_j's prefix (which is P_j).
            # Thus, the overlap must be contained within the suffix of P_i and prefix of P_j, OR it wraps around?
            # Wait. Block_i = P_i + [n] + P_i.
            # Suffix of Block_i of length k.
            # If k <= n-1, it's suffix of P_i.
            # If k > n-1, it includes the 'n'.
            # Prefix of Block_j of length k.
            # If k <= n-1, it's prefix of P_j.
            # If k > n-1, it includes the 'n' of Block_j.
            
            # Case 1: k <= n-1. Suffix of P_i == Prefix of P_j.
            # Case 2: k > n-1.
            # Suffix of Block_i: ... P_i (last k-(n-1) chars of P_i) + n + (first k-(2*(n-1)) chars of P_i)? No.
            # Let's just use string comparison for simplicity, it's fast enough for N=720.
            
            # We can limit max_o to 2*(n-1) because if it's longer, it would imply P_i and P_j are identical in a way that's impossible for distinct perms?
            # Actually, max overlap is at most 2*(n-1).
            for k in range(min(max_o, 2*(n-1)), -1, -1):
                if k == 0 or block_i[len(block_i)-k:] == block_j[:k]:
                    o = k
                    break
            overlaps[i][j] = o
            
    # Nearest Neighbor Initialization
    visited = [False]*num_perms
    order = [0]
    visited[0] = True
    
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_overlap = -1
        for j in range(num_perms):
            if not visited[j]:
                if overlaps[last][j] > best_overlap:
                    best_overlap = overlaps[last][j]
                    best_next = j
        if best_next == -1: break
        order.append(best_next)
        visited[best_next] = True
        
    # Calculate current total overlap
    current_overlap = 0
    for i in range(num_perms - 1):
        current_overlap += overlaps[order[i]][order[i+1]]
        
    # 2-Opt Local Search
    # We want to maximize current_overlap
    improved = True
    max_iterations = 10 # Safety limit on passes
    pass_count = 0
    
    while improved and pass_count < max_iterations:
        if time.time() > deadline - 10:
            break
        improved = False
        pass_count += 1
        for i in range(num_perms - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 2, num_perms):
                # Move: reverse order[i+1 .. j]
                # Edges removed: (i, i+1) and (j-1, j) ? No.
                # Sequence: ... A (i) B (i+1) ... C (j-1) D (j) ...
                # Reverse B..C.
                # New sequence: ... A C ... B D ...
                # Edges removed: (A, B) and (C, D)
                # Edges added: (A, C) and (B, D)
                
                # Note: If i+1 == j, it's a swap of two elements?
                # If j = i+1, reverse is same.
                # If j = i+2, reverse B, C -> C, B.
                # Edges removed: (A, B), (B, C), (C, D)
                # Edges added: (A, C), (C, B), (B, D)
                
                # General 2-opt on path:
                # Remove edges (i, i+1) and (j-1, j).
                # Add edges (i, j) and (i+1, j-1).
                # This assumes we reverse the segment i+1 to j.
                
                old_cost = overlaps[order[i]][order[i+1]] + overlaps[order[j-1]][order[j]]
                new_cost = overlaps[order[i]][order[j]] + overlaps[order[i+1]][order[j-1]]
                
                if new_cost > old_cost:
                    # Perform reversal
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    current_overlap += (new_cost - old_cost)
                    improved = True
                    # After a change, we should restart the inner loop or continue carefully.
                    # Simple approach: restart inner loop to avoid index issues?
                    # Or just continue. Since we changed the array, subsequent j checks for this i might be invalid?
                    # Actually, the structure changed. It's safer to break inner and restart outer?
                    # Or just break inner and continue outer?
                    # For speed, we'll just break inner and restart the whole pass? No, that's slow.
                    # Let's just break the inner loop and continue the outer.
                    break
        
    # Construct new word
    new_word = []
    for idx in order:
        block = blocks[idx]
        max_overlap = min(len(new_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                overlap = o
                break
        new_word += block[overlap:]
        
    # Standard word for comparison
    std_new_word = []
    for i in range(num_perms):
        block = blocks[i]
        max_overlap = min(len(std_new_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0 or std_new_word[len(std_new_word) - o:] == block[:o]:
                overlap = o
                break
        std_new_word += block[overlap:]
        
    if len(new_word) < len(std_new_word):
        word = new_word
    else:
        word = std_new_word
        
    # Try to trim
    candidate = word[:-1]
    if len(candidate) >= n:
        seen_perms = set()
        for i in range(len(candidate) - n + 1):
            w = tuple(candidate[i:i+n])
            if len(set(w)) == n:
                seen_perms.add(w)
                if len(seen_perms) == math.factorial(n):
                    break
        if len(seen_perms) == math.factorial(n):
            word = candidate
            
    return "".join(map(str, word))
