import itertools
import time
import random

def _overlap_len(a, b):
    """Calculate the length of the longest suffix of a that matches a prefix of b."""
    max_overlap = min(len(a), len(b))
    for o in range(max_overlap, -1, -1):
        if o == 0 or a[-o:] == b[:o]:
            return o
    return 0

def _optimize_order(perms):
    """
    Reorder permutations to maximize the sum of overlaps between consecutive blocks.
    Block for perm p is p + [new_symbol] + p. 
    The overlap between block(p_i) and block(p_j) is determined by suffix of p_i and prefix of p_j.
    Specifically, suffix of p_i + [new] + p_i and prefix of p_j + [new] + p_j.
    Since [new] is unique, the overlap cannot cross the [new].
    So the overlap is the longest suffix of p_i that matches a prefix of p_j.
    """
    n_perms = len(perms)
    if n_perms <= 1:
        return perms
    
    # Precompute overlap matrix
    # overlap[i][j] = _overlap_len(perms[i], perms[j])
    # We want to maximize sum of overlaps.
    # This is equivalent to minimizing sum of (n-1 - overlap).
    
    # Greedy Nearest Neighbor
    # Start with a random or specific start
    # To be deterministic, start with perms[0]
    used = [False] * n_perms
    order = [0]
    used[0] = True
    
    # Precompute overlaps for speed? 
    # N=720, N^2 = 500k. Storing matrix is fine.
    overlaps = [[0]*n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        pi = perms[i]
        for j in range(n_perms):
            if i == j: continue
            pj = perms[j]
            # Calculate overlap
            max_ov = min(len(pi), len(pj))
            ov = 0
            for k in range(max_ov, 0, -1):
                if pi[-k:] == pj[:k]:
                    ov = k
                    break
            overlaps[i][j] = ov
            
    # Greedy
    for _ in range(n_perms - 1):
        last = order[-1]
        best_next = -1
        best_score = -1
        for j in range(n_perms):
            if not used[j]:
                score = overlaps[last][j]
                if score > best_score:
                    best_score = score
                    best_next = j
        order.append(best_next)
        used[best_next] = True
        
    # 2-Opt Local Search
    # We want to maximize sum of overlaps.
    # Let path be order.
    # Sum = overlaps[order[0]][order[1]] + ... + overlaps[order[N-2]][order[N-1]]
    # Note: The first block doesn't have an incoming overlap in the sum, but it has a length.
    # The total length is L_first + sum(L_block - overlap).
    # L_block is constant.
    # So maximizing sum of overlaps is correct.
    
    improved = True
    max_iter = 10 # Limit iterations to ensure speed
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(n_perms - 1):
            for j in range(i + 1, n_perms):
                # Reverse segment i..j
                # Check if reversal improves sum
                # Original: ... A - B ... C - D ...
                # Where A=order[i-1], B=order[i], C=order[j], D=order[j+1]
                # New: ... A - C ... B - D ...
                
                if i > 0:
                    a = order[i-1]
                else:
                    a = None
                
                if j < n_perms - 1:
                    d = order[j+1]
                else:
                    d = None
                    
                b = order[i]
                c = order[j]
                
                old_score = 0
                if a is not None:
                    old_score += overlaps[a][b]
                if d is not None:
                    old_score += overlaps[c][d]
                
                new_score = 0
                if a is not None:
                    new_score += overlaps[a][c]
                if d is not None:
                    new_score += overlaps[b][d]
                
                # Note: The internal overlaps in the reversed segment remain the same?
                # Sum of overlaps in segment i..j:
                # Old: ov(B, next) + ... + ov(prev, C)
                # New: ov(C, next) + ... + ov(prev, B)
                # Since overlaps are symmetric? No, overlaps[i][j] != overlaps[j][i] generally.
                # But the sum of internal overlaps is the same if we consider the path?
                # Path: b -> ... -> c
                # Reversed: c -> ... -> b
                # Sum of overlaps in reversed path is sum of overlaps in original path?
                # ov(x, y) vs ov(y, x).
                # The sum is NOT necessarily the same.
                # So we should calculate the full difference.
                # For N=720, calculating full sum is O(N).
                # 2-opt is O(N^2) passes.
                # With N=720, N^2 = 500k.
                # If we do O(N) check for each pair, it's O(N^3) = 300 billion. Too slow.
                
                # We need to check if the reversal is beneficial.
                # The change is only at the boundaries A-B, C-D and the internal sum.
                # Let S_old_internal = sum(overlaps[order[k]][order[k+1]] for k in i..j-1)
                # Let S_new_internal = sum(overlaps[order[k+1]][order[k]] for k in i..j-1)
                # We can precompute prefix sums of overlaps?
                # But the direction changes.
                
                # Given time constraints, maybe skip 2-opt for N=720 or use a simpler heuristic.
                # Or just use Greedy.
                # Greedy is often good enough for this structure.
                
                # Let's try to implement a faster check or just use Greedy.
                # For now, let's just use Greedy to be safe on time.
                # If we have time, we can add 2-opt with a limit.
                
                pass # Placeholder for 2-opt logic if we decide to include it
                
    return [perms[i] for i in order]

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Optimize order of perms
        if time.time() < deadline - 10: # Leave 10s for construction and rest
            perms = _optimize_order(perms)
            
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if time.time() > deadline:
        return "".join(map(str, word))
        
    # 2. Calculate frequency of all permutations in the word
    freq = {}
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        freq[p] = freq.get(p, 0) + 1
        
    # 3. Trim from ends
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        if len(word) > n:
            first_perm = tuple(word[:n])
            if freq.get(first_perm, 0) > 1:
                freq[first_perm] -= 1
                word = word[1:]
                changed = True
                
        if len(word) > n:
            last_perm = tuple(word[-n:])
            if freq.get(last_perm, 0) > 1:
                freq[last_perm] -= 1
                word = word[:-1]
                changed = True
                
    return "".join(map(str, word))
