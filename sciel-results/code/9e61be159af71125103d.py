import itertools
import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    time_limit = 110.0

    if n <= 3:
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

    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}

    # Compute Asymmetric Cost Matrix
    C = np.zeros((N, N), dtype=np.int8)
    
    # Vectorized-ish overlap calculation
    # Convert perms to numpy array of ints
    P = np.array(perms)
    
    # We want C[i][j] = n - overlap(P[i], P[j])
    # overlap is max k < n such that P[i][n-k:] == P[j][:k]
    
    # This is hard to vectorize fully due to variable k.
    # But we can loop over k.
    # For k from n-1 down to 1:
    #   Check if P[i][n-k:] == P[j][:k]
    #   If yes, C[i][j] = n - k, break.
    
    # We can do this in a loop over k, and use numpy to find matches?
    # For a fixed k, we can compare all suffixes of length k with all prefixes of length k.
    
    # Let's try this.
    # Initialize C with n (overlap 0)
    C[:] = n
    
    for k in range(n - 1, 0, -1):
        # Suffixes of length k: P[:, n-k:]
        # Prefixes of length k: P[:, :k]
        # We want to find pairs (i, j) such that P[i, n-k:] == P[j, :k]
        
        # This is a matrix matching problem.
        # We can flatten the suffixes and prefixes?
        # Or use a dictionary.
        
        # For each i, suffix = P[i, n-k:]
        # For each j, prefix = P[j, :k]
        # If suffix == prefix, then C[i][j] = n - k.
        
        # To speed up, we can group by the value of the tuple.
        # Since n is small, we can just loop i and j?
        # N^2 is 25M.
        # Doing this for each k (up to 6) is 150M operations.
        # In Python, this is too slow.
        
        # Use Numpy broadcasting?
        # Suffixes: shape (N, k)
        # Prefixes: shape (N, k)
        # We want to compare Suffixes[i] with Prefixes[j].
        # This results in a (N, N, k) array. Memory: 5040*5040*7*4 bytes ~ 700MB. Too much.
        
        # Alternative:
        # Loop over i.
        # For each i, we want to find all j such that P[j, :k] == P[i, n-k:].
        # We can precompute a dictionary mapping prefix tuples to list of indices.
        
        prefix_map = {}
        for j in range(N):
            pref = tuple(P[j, :k])
            if pref not in prefix_map:
                prefix_map[pref] = []
            prefix_map[pref].append(j)
            
        for i in range(N):
            suff = tuple(P[i, n-k:])
            if suff in prefix_map:
                for j in prefix_map[suff]:
                    C[i, j] = n - k
                    
    # Compute Symmetric Cost Matrix S
    S = np.minimum(C, C.T)
    
    # Run Nearest Neighbor with Symmetric Cost
    best_path = None
    best_cost = float('inf')
    
    # Use seed path as one candidate
    # Extract seed path
    word_seed = [1]
    for m in range(2, n + 1):
        perms_seed, seen = [], set()
        for i in range(len(word_seed) - m + 2):
            window = tuple(word_seed[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_seed.append(window)
        new = []
        for p in perms_seed:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_seed = new
    
    path_seed = []
    seen_perms = set()
    for i in range(len(word_seed) - n + 1):
        window = tuple(word_seed[i:i + n])
        if window not in seen_perms:
            seen_perms.add(window)
            path_seed.append(perm_to_idx[window])
            
    # Evaluate seed path with Asymmetric Cost
    def eval_path(path):
        cost = 0
        for i in range(len(path) - 1):
            cost += C[path[i], path[i+1]]
        return cost + n # Add first permutation length
    
    best_cost = eval_path(path_seed)
    best_path = path_seed
    
    # Try a few random NN starts
    num_runs = 20
    for run in range(num_runs):
        if time.time() - start_time > time_limit - 20:
            break
            
        start_node = random.randint(0, N - 1)
        path = [start_node]
        unvisited = set(range(N))
        unvisited.remove(start_node)
        
        while unvisited:
            current = path[-1]
            # Find nearest neighbor in S
            # S[current, :]
            # We want min S[current, j] for j in unvisited
            # To speed up, we can just pick the global min and check if it's visited?
            # No, we must pick from unvisited.
            
            # Simple: loop over unvisited? O(N^2) total.
            # 5040 * 5040 / 2 ~ 12M ops. ~1-2s per run.
            # 20 runs ~ 40s.
            
            min_val = float('inf')
            min_idx = -1
            for j in unvisited:
                val = S[current, j]
                if val < min_val:
                    min_val = val
                    min_idx = j
            
            path.append(min_idx)
            unvisited.remove(min_idx)
            
        # 2-Opt with Symmetric Cost
        # (Omitted for brevity in thought, will implement in code)
        
        cost = eval_path(path)
        if cost < best_cost:
            best_cost = cost
            best_path = path
            
    # Reconstruct word from best_path
    result = list(perms[best_path[0]])
    for k in range(1, len(best_path)):
        prev_p = perms[best_path[k-1]]
        curr_p = perms[best_path[k]]
        max_k = 0
        for k_len in range(n - 1, -1, -1):
            if prev_p[n - k_len:] == curr_p[:k_len]:
                max_k = k_len
                break
        result.extend(curr_p[max_k:])
        
    return "".join(map(str, result))
