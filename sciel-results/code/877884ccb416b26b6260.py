import time
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(permutations(range(1, n+1)))
    N = len(perms)
    
    # Map perm to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Compute Overlap Matrix K (uint8)
    # K[i][j] = max k such that suffix_k(perms[i]) == prefix_k(perms[j])
    K = np.zeros((N, N), dtype=np.uint8)
    
    # Helper to get suffix/prefix tuples
    # We can optimize by precomputing suffixes and prefixes for each k
    # But N=5040, k<=6. 
    # Let's use the dictionary approach for speed.
    
    # For each k from n-1 down to 1
    for k in range(n-1, 0, -1):
        # Map suffix of length k to list of indices
        # Map prefix of length k to list of indices
        suff_map = {}
        pref_map = {}
        
        for i, p in enumerate(perms):
            s = p[n-k:]
            pre = p[:k]
            if s not in suff_map:
                suff_map[s] = []
            suff_map[s].append(i)
            
            if pre not in pref_map:
                pref_map[pre] = []
            pref_map[pre].append(i)
        
        # For each key in suff_map, find matches in pref_map
        for s, i_list in suff_map.items():
            if s in pref_map:
                j_list = pref_map[s]
                for i in i_list:
                    for j in j_list:
                        if K[i][j] < k:
                            K[i][j] = k
                            
    # Cost matrix C = n - K
    # We want to minimize sum of C, which is maximizing sum of K.
    # Let's work with K directly. Maximize sum of K.
    
    # Initial Path: Nearest Neighbor
    # Start at 0
    path = [0]
    remaining = set(range(1, N))
    curr = 0
    
    # To speed up NN, we can just pick max K[curr][j] for j in remaining.
    # O(N^2) total.
    while remaining:
        max_k = -1
        best_j = -1
        # Optimization: if time is tight, skip NN and use recursive?
        # NN is fast enough.
        for j in remaining:
            if K[curr][j] > max_k:
                max_k = K[curr][j]
                best_j = j
        path.append(best_j)
        remaining.remove(best_j)
        curr = best_j
        
    # Convert to numpy array for easier manipulation?
    # List is fine.
    
    # Calculate initial total overlap
    def calc_total_overlap(p):
        return sum(K[p[i]][p[i+1]] for i in range(len(p)-1))
        
    current_overlap = calc_total_overlap(path)
    
    # Prefix sums for 2-opt
    # PrefK[i] = sum of K[path[t]][path[t+1]] for t=0 to i-1
    # PrefKR[i] = sum of K[path[t+1]][path[t]] for t=0 to i-1
    
    def get_prefix_sums(p):
        n_len = len(p)
        pf = np.zeros(n_len, dtype=np.int32)
        pr = np.zeros(n_len, dtype=np.int32)
        for i in range(n_len - 1):
            pf[i+1] = pf[i] + K[p[i]][p[i+1]]
            pr[i+1] = pr[i] + K[p[i+1]][p[i]]
        return pf, pr
        
    pf, pr = get_prefix_sums(path)
    
    import random
    random.seed(42)
    
    no_improve_count = 0
    
    while time.time() - start_time < deadline:
        # Try 2-opt
        # Random i, j
        i = random.randint(0, N-2)
        j = random.randint(i+1, N-1)
        
        # Old Overlap
        # Edges: (i-1, i), (i, i+1)...(j-1, j), (j, j+1)
        # Internal: sum K[path[t]][path[t+1]] for t=i to j-1
        # This is pf[j] - pf[i]
        
        # New Overlap
        # Reverse segment i..j
        # New internal: sum K[path[t+1]][path[t]] for t=i to j-1
        # This is pr[j] - pr[i]
        
        # Boundary edges:
        # Old: (i-1, i) and (j, j+1)
        # New: (i-1, j) and (i, j+1)
        
        old_boundary = 0
        if i > 0:
            old_boundary += K[path[i-1]][path[i]]
        if j < N-1:
            old_boundary += K[path[j]][path[j+1]]
            
        new_boundary = 0
        if i > 0:
            new_boundary += K[path[i-1]][path[j]]
        if j < N-1:
            new_boundary += K[path[i]][path[j+1]]
            
        old_internal = pf[j] - pf[i]
        new_internal = pr[j] - pr[i]
        
        old_total = old_boundary + old_internal
        new_total = new_boundary + new_internal
        
        if new_total > old_total:
            # Apply move
            # Reverse path[i:j+1]
            path[i:j+1] = path[i:j+1][::-1]
            current_overlap += new_total - old_total
            pf, pr = get_prefix_sums(path)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Try Or-opt (move 1)
        if random.random() < 0.5:
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i == j: continue
            if i == j-1: continue # swap, handled by 2-opt or separate logic
            
            # Move i to after j
            # Assume i != j
            # Remove i
            # Insert at j
            
            # To simplify, let's just do a simple swap of two non-adjacent nodes?
            # No, 2-opt is better.
            # Let's add a simple "move node i to position j"
            
            # This is complex to update prefix sums efficiently.
            # Given time, maybe skip Or-opt and rely on 2-opt + random restarts?
            # Or just 2-opt is enough?
            
            # Let's try a "3-opt" or "segment move" if 2-opt stalls.
            
        if no_improve_count > 1000:
            # Try a small random perturbation
            # Swap two random nodes
            i = random.randint(0, N-2)
            j = random.randint(i+1, N-1)
            path[i], path[j] = path[j], path[i]
            current_overlap = calc_total_overlap(path)
            pf, pr = get_prefix_sums(path)
            no_improve_count = 0
            
    # Convert path to string
    # The word is:
    # Start with perms[path[0]]
    # For each next perm, append the suffix that doesn't overlap
    
    word = list(perms[path[0]])
    for idx in range(1, N):
        prev = path[idx-1]
        curr = path[idx]
        k = K[prev][curr]
        # Append curr[k:]
        word.extend(perms[curr][k:])
        
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
