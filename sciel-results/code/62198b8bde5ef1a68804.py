import itertools
import random
import time
import numpy as np

def construct(n):
    if n <= 6:
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n=7, use search
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # 2. Precompute cost matrix
    # cost_mat[i][j] = n - max_overlap(perms[i], perms[j])
    cost_mat = np.full((N, N), n, dtype=np.uint8)
    
    # For k from n-1 down to 1
    for k in range(n - 1, 0, -1):
        # Map prefix of length k to list of indices
        prefix_map = {}
        for j, p in enumerate(perms):
            prefix = p[:k]
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(j)
        
        # For each i, find suffix of length k
        for i, p in enumerate(perms):
            suffix = p[n-k:]
            if suffix in prefix_map:
                for j in prefix_map[suffix]:
                    if cost_mat[i][j] == n:
                        cost_mat[i][j] = n - k

    # 3. Greedy Construction with Randomization
    # We use the fast prefix-map greedy method
    # But we can also use the cost matrix for tie-breaking or selection
    
    def greedy_random():
        unvisited = set(range(N))
        # Build prefix sets for unvisited
        # To make it fast, we maintain these sets
        prefix_sets = {}
        for j, p in enumerate(perms):
            for k in range(1, n):
                prefix = p[:k]
                if prefix not in prefix_sets:
                    prefix_sets[prefix] = set()
                prefix_sets[prefix].add(j)
        
        # Start with random permutation
        start_idx = random.randint(0, N - 1)
        unvisited.remove(start_idx)
        for k in range(1, n):
            prefix_sets[perms[start_idx][:k]].remove(start_idx)
            
        path = [start_idx]
        curr = start_idx
        
        for _ in range(N - 1):
            next_idx = -1
            # Check overlaps from largest to smallest
            for k in range(n - 1, 0, -1):
                suffix = perms[curr][n-k:]
                candidates = prefix_sets.get(suffix, set())
                if candidates:
                    # Pick one randomly
                    next_idx = random.choice(list(candidates))
                    break
            
            if next_idx == -1:
                # No overlap, pick random unvisited
                next_idx = random.choice(list(unvisited))
            
            unvisited.remove(next_idx)
            for k in range(1, n):
                prefix_sets[perms[next_idx][:k]].remove(next_idx)
                
            path.append(next_idx)
            curr = next_idx
            
        return path

    def path_length(path):
        # Length = n + sum(cost_mat[path[i]][path[i+1]])
        # cost_mat is n - overlap, so added length is cost_mat
        return n + sum(cost_mat[path[i]][path[i+1]] for i in range(len(path) - 1))

    best_path = None
    best_len = float('inf')
    
    # Run greedy multiple times
    iterations = 0
    while time.time() - start_time < deadline - 20: # Leave 20s for 2-opt
        path = greedy_random()
        l = path_length(path)
        if l < best_len:
            best_len = l
            best_path = path
        iterations += 1
        if iterations % 100 == 0:
            # Check time
            if time.time() - start_time > deadline - 20:
                break

    # 4. 2-opt Improvement on best_path
    if best_path and time.time() - start_time < deadline - 5:
        current_path = best_path
        current_len = best_len
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            for i in range(1, N - 1):
                for j in range(i + 1, N):
                    # Reverse segment i..j
                    # New edges: (i-1, j) and (i, j+1)
                    # Old edges: (i-1, i) and (j, j+1)
                    old_cost = cost_mat[current_path[i-1]][current_path[i]] + cost_mat[current_path[j]][current_path[j+1]]
                    new_cost = cost_mat[current_path[i-1]][current_path[j]] + cost_mat[current_path[i]][current_path[j+1]]
                    
                    if new_cost < old_cost:
                        # Apply move
                        current_path[i:j+1] = reversed(current_path[i:j+1])
                        current_len += new_cost - old_cost
                        improved = True
                        # Restart loops or continue? Usually restart or continue.
                        # To be safe and fast, break and restart outer loop?
                        # Or just continue.
                        break
                if improved:
                    break
                    
    # Construct final string
    final_word = []
    for idx in best_path:
        p = perms[idx]
        if not final_word:
            final_word.extend(p)
        else:
            # Append with overlap
            # We need to find overlap between last perm and this one
            # We can use cost_mat or just recompute
            # cost_mat gives n - overlap. So overlap = n - cost
            # But we need the actual overlap value to slice
            prev_idx = best_path[best_path.index(idx) - 1] if idx != best_path[0] else None
            # This is inefficient. Let's just iterate properly.
            
    # Reconstruct properly
    final_word = list(perms[best_path[0]])
    for i in range(1, len(best_path)):
        prev_idx = best_path[i-1]
        curr_idx = best_path[i]
        p_prev = perms[prev_idx]
        p_curr = perms[curr_idx]
        
        # Find max overlap
        max_k = 0
        for k in range(n - 1, 0, -1):
            if p_prev[n-k:] == p_curr[:k]:
                max_k = k
                break
        final_word.extend(p_curr[max_k:])
        
    return "".join(map(str, final_word))

