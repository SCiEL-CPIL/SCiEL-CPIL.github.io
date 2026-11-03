import numpy as np
from itertools import permutations
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Classic construction
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)
    
    # If n is small, classic is likely optimal or very close.
    # For n=7, we attempt TSP optimization.
    if n >= 4 and n <= 7:
        # For n=4,5,6 classic is optimal (length matches known shortest).
        # For n=7, classic is 5913, shortest is 5905.
        # TSP approach might find something better.
        if n == 7:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)
        elif n < 7:
            # Could try TSP here too, but classic is already optimal for 4,5,6?
            # Let's check.
            # n=4: 33. n=5: 153. n=6: 873.
            # If TSP finds shorter, we take it.
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)

    return best_word

def _classic_construct(n):
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _tsp_construct(n, deadline):
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map tuple to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Convert to strings for faster overlap computation
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix using a more efficient method
    # overlap[i][j] = length of longest suffix of perms[i] that is prefix of perms[j]
    # Max overlap is n-1 (if n>1)
    
    # For each position l from 1 to n-1, compute which pairs have overlap >= l
    # This allows us to build the matrix more efficiently
    
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # For each possible overlap length l (from n-1 down to 1),
    # find all pairs (i,j) where suffix of length l of perm i equals prefix of length l of perm j
    
    # Group permutations by their suffixes and prefixes
    # For each l, create a mapping from suffix string to list of indices
    # and from prefix string to list of indices
    
    for l in range(n - 1, 0, -1):
        # Only set overlap[i][j] = l if it hasn't been set to a larger value already
        # So we go from large l to small l
        
        # Build suffix groups
        suffix_groups = {}
        for i in range(N):
            s = perm_strs[i][n-l:]
            if s not in suffix_groups:
                suffix_groups[s] = []
            suffix_groups[s].append(i)
        
        # Build prefix groups
        prefix_groups = {}
        for j in range(N):
            p = perm_strs[j][:l]
            if p not in prefix_groups:
                prefix_groups[p] = []
            prefix_groups[p].append(j)
        
        # For each common string, all (i,j) pairs get overlap at least l
        for key in suffix_groups:
            if key in prefix_groups:
                for i in suffix_groups[key]:
                    for j in prefix_groups[key]:
                        if overlap[i][j] == 0:  # Only set if not already set to larger value
                            overlap[i][j] = l
        
        # Check time
        if time.time() > deadline - 30:
            return None
    
    # Cost matrix: cost[i][j] = n - overlap[i][j]
    # We want to minimize sum of costs.
    # This is TSP.
    
    # Heuristic: Nearest Neighbor + 2-Opt
    # Start at 0
    visited = [False] * N
    visited[0] = True
    path = [0]
    curr = 0
    
    for _ in range(N - 1):
        # Find next
        # costs from curr to all unvisited
        # We can vectorize this
        unvisited = np.array([k for k in range(N) if not visited[k]])
        if len(unvisited) == 0: break
        # overlap[curr][unvisited]
        ovs = overlap[curr, unvisited]
        # min cost is max overlap
        best_idx_in_unvisited = np.argmax(ovs)
        next_idx = unvisited[best_idx_in_unvisited]
        
        visited[next_idx] = True
        path.append(next_idx)
        curr = next_idx
        
        if time.time() > deadline - 20:
            return None
            
    # 2-Opt improvement
    # Iterate a few times
    improved = True
    max_iters = 50
    iter_count = 0
    while improved and iter_count < max_iters and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                # Reverse path[i:j+1]
                # Check cost change
                # Edges: (i-1, i), (j, j+1)
                # New edges: (i-1, j), (i, j+1)
                # If j == N-1, the second edge is (i, N-1) -> no, j+1 is out of bounds?
                # Path is a path, not cycle.
                # So if j == N-1, we only change edge (i-1, i) to (i-1, j).
                
                old_cost = 0
                new_cost = 0
                
                # Edge 1: i-1 -> i
                if i > 0:
                    old_cost += n - overlap[path[i-1]][path[i]]
                
                # Edge 2: j -> j+1
                if j < N - 1:
                    old_cost += n - overlap[path[j]][path[j+1]]
                
                # New Edge 1: i-1 -> j
                if i > 0:
                    new_cost += n - overlap[path[i-1]][path[j]]
                
                # New Edge 2: i -> j+1
                if j < N - 1:
                    new_cost += n - overlap[path[i]][path[j+1]]
                
                if new_cost < old_cost:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
                    
    # Build string
    # Start with perms[path[0]]
    s = perm_strs[path[0]]
    for k in range(1, N):
        p_prev = perm_strs[path[k-1]]
        p_curr = perm_strs[path[k]]
        # Find overlap
        max_ov = n - 1
        ov = 0
        for l in range(max_ov, 0, -1):
            if p_prev[-l:] == p_curr[:l]:
                ov = l
                break
        s += p_curr[ov:]
        
    # Verify validity
    # Check if all perms are in s
    # For n=7, s length ~ 6000. Number of windows ~ 6000.
    # We can just check if set of windows contains all perms.
    
    windows = set()
    for i in range(len(s) - n + 1):
        w = tuple(map(int, s[i:i+n]))
        windows.add(w)
        
    if len(windows) == N:
        return s
    else:
        return None
