import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n <= 4:
        # Small n, just use standard construction or brute force
        return "".join(map(str, standard_superperm(n)))
        
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    num_perms = len(perms)
    
    # Convert to strings for easier overlap checking? Or keep as tuples.
    # Strings are faster for slicing/comparison in Python sometimes.
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlaps
    # overlap[i][j] = k means perm_strs[i][-k:] == perm_strs[j][:k]
    # Max overlap is n-1.
    overlaps = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: Only calculate if time allows.
    # For n=7, num_perms=5040. 25M pairs.
    # We can parallelize? No, standard library only.
    # We can just do it.
    
    for i in range(num_perms):
        si = perm_strs[i]
        for j in range(num_perms):
            if i == j: continue
            sj = perm_strs[j]
            # Check overlaps from n-1 down to 1
            # Max possible overlap is n-1
            max_ov = n - 1
            for k in range(max_ov, 0, -1):
                if si[-k:] == sj[:k]:
                    overlaps[i][j] = k
                    break
    
    if time.time() > deadline - 20:
        # Not enough time for search, return standard
        return "".join(map(str, standard_superperm(n)))
        
    # Greedy construction with multiple restarts
    best_len = float('inf')
    best_word = None
    
    # We can run several greedy passes
    # Each pass picks a random start or the best start
    # Given the size, even one good greedy pass might be better than standard.
    
    for restart in range(5): # 5 restarts
        if time.time() > deadline - 10:
            break
            
        # Choose start
        # Standard construction starts with 12...n
        # Let's try starting with a random permutation
        start_idx = 0 if restart == 0 else time.time() % num_perms # Simple pseudo-random
        
        visited = [False] * num_perms
        visited[start_idx] = True
        current_idx = start_idx
        current_str = perm_strs[start_idx]
        
        for _ in range(num_perms - 1):
            if time.time() > deadline - 5:
                break
                
            best_next = -1
            best_ov = -1
            # Find next unvisited perm with max overlap
            # To speed this up, we could maintain a priority queue, but for 5040 items, linear scan is 5040 * 5040 = 25M ops per restart.
            # 5 restarts * 25M = 125M ops. Might be too slow in Python.
            
            # Optimization: Pre-compute "best next" for each node? No, depends on visited.
            
            # Alternative: Just do one greedy pass from the standard start.
            # The standard start is known to be good.
            # Let's just do 1 greedy pass from the standard start to see if it beats standard.
            # Actually, the standard construction IS a greedy construction with a specific tie-breaking/order.
            # If we use a "Max Overlap" greedy, it might be different.
            
            # Let's just do the linear scan. 25M ops is ~5-10s in Py.
            for j in range(num_perms):
                if visited[j]:
                    continue
                ov = overlaps[current_idx][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            
            if best_next == -1:
                break # Should not happen
                
            visited[best_next] = True
            current_str += perm_strs[best_next][best_ov:]
            current_idx = best_next
            
        if len(current_str) < best_len:
            best_len = len(current_str)
            best_word = current_str
            
    if best_word is not None:
        return best_word
    else:
        return "".join(map(str, standard_superperm(n)))

def standard_superperm(n):
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
            overlap = 0
            # Find max overlap
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word
