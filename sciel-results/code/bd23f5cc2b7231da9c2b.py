import time
import sys

def construct(n):
    if n < 7:
        # Classic recursive construction is optimal for n <= 6
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
                # Find overlap
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n != 7:
        # Fallback for other n, though task implies n=7
        # Use classic for safety
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # n == 7 specific search
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Precompute permutations and transitions
    perms = []
    perm_to_id = {}
    # Generate all permutations of 1..7
    import itertools
    for p in itertools.permutations(range(1, n+1)):
        perm_to_id[p] = len(perms)
        perms.append(p)
    
    num_perms = len(perms)
    
    # Precompute next states
    # next_trans[i] = list of (j, c) where c is the symbol added to perm i to get perm j
    next_trans = [[] for _ in range(num_perms)]
    for i, p in enumerate(perms):
        suffix = p[1:]
        for c in range(1, n+1):
            new_p = suffix + (c,)
            if new_p in perm_to_id:
                j = perm_to_id[new_p]
                next_trans[i].append((j, c))
                
    # Sort transitions by c to ensure deterministic order (smallest first)
    for i in range(num_perms):
        next_trans[i].sort(key=lambda x: x[1])

    # DFS with backtracking
    # State: current_perm_id, seen_mask, path (list of symbols)
    # We store the path as a list of ints to allow backtracking easily
    
    best_solution = None
    best_length = 5913 # Classic length
    
    # Start with perm (1, 2, 3, 4, 5, 6, 7)
    start_id = perm_to_id[(1, 2, 3, 4, 5, 6, 7)]
    
    # Stack for DFS: (current_id, seen_mask, path)
    # We use a list to simulate stack
    stack = [(start_id, 1 << start_id, [1, 2, 3, 4, 5, 6, 7])]
    
    while stack:
        if time.time() > deadline:
            break
            
        curr_id, seen_mask, path = stack.pop()
        
        # Check if we found a solution
        if seen_mask == (1 << num_perms) - 1:
            curr_len = len(path)
            if curr_len < best_length:
                best_length = curr_len
                best_solution = path[:]
                # We can stop if we found 5905 (known optimal)
                if best_length <= 5905:
                    break
            continue
            
        # Heuristic Pruning
        # Min remaining steps = num_perms - popcount(seen_mask)
        # If len(path) + min_remaining > 5905, prune
        # Note: We are looking for <= 5905
        popcount = bin(seen_mask).count('1')
        min_remaining = num_perms - popcount
        if len(path) + min_remaining > 5905:
            continue
            
        # Try next moves
        # We want to explore promising branches first.
        # Heuristic: Prefer moves that keep options open?
        # For now, try in order of c (smallest first)
        # To encourage backtracking, we push in reverse order
        
        moves = next_trans[curr_id]
        # Filter moves that lead to new perms
        valid_moves = []
        for j, c in moves:
            if not (seen_mask & (1 << j)):
                valid_moves.append((j, c))
                
        # Push in reverse so smallest c is popped first
        for j, c in reversed(valid_moves):
            new_mask = seen_mask | (1 << j)
            new_path = path + [c]
            stack.append((j, new_mask, new_path))
            
    if best_solution:
        return "".join(map(str, best_solution))
    else:
        # Fallback to classic
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
