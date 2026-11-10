import numpy as np
import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # For small n, standard construction is optimal and fast
    if n <= 4:
        # Standard construction
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

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    P = np.array(perms)  # Shape (N, n)
    
    # Compute overlap matrix O
    # O[i, j] = max k such that P[i][-k:] == P[j][:k]
    O = np.zeros((N, N), dtype=np.int8)
    
    for k in range(n - 1, 0, -1):
        # Suffix of length k for all i
        # P[:, n-k:]
        # Prefix of length k for all j
        # P[:, :k]
        # Compare all pairs
        # (N, k, 1) vs (1, N, k)
        # Result shape (N, N)
        # We want True if P[i, n-k:] == P[j, :k]
        
        # To save memory, we can do this in chunks or just rely on NumPy broadcasting
        # N=5040, N^2 = 25M. 25M booleans is 25MB. Fine.
        
        suffixes = P[:, n-k:]  # (N, k)
        prefixes = P[:, :k]    # (N, k)
        
        # Broadcast comparison
        # suffixes[:, None, :] has shape (N, 1, k)
        # prefixes[None, :, :] has shape (1, N, k)
        # Equality gives (N, N, k)
        # All along last axis gives (N, N)
        
        # This might be memory intensive for large N, but N=5040 is okay.
        # 5040 * 5040 * 6 * 1 byte = ~150MB for the intermediate array?
        # Actually, `==` creates a boolean array.
        # Let's try to be efficient.
        
        # Alternative: Loop over k, and for each k, check matches.
        # We can use a dictionary of prefixes to find matches faster?
        # But vectorized is likely faster for N=5040.
        
        # Let's check memory:
        # The intermediate array is (N, N, k).
        # 5040 * 5040 * 6 = 152 million booleans = 152 MB.
        # This is acceptable.
        
        matches = (suffixes[:, None, :] == prefixes[None, :, :]).all(axis=-1)
        
        # Update O where matches is True and O is 0
        # We want the max k, so we iterate from high k to low k.
        # If a match is found at k, and O[i,j] is 0, we set it to k.
        # If O[i,j] is already set (from a higher k), we don't change it.
        
        mask = matches & (O == 0)
        O[mask] = k
        
        if time.time() > deadline:
            break

    # Cost matrix C = n - O
    C = n - O
    C = C.astype(np.int32)
    
    # TSP Heuristic: Nearest Neighbor
    unvisited = np.ones(N, dtype=bool)
    path = [0]
    unvisited[0] = False
    current = 0
    
    while np.any(unvisited):
        if time.time() > deadline:
            break
            
        # Find min cost to unvisited nodes
        # C[current, :] gives costs to all nodes
        # We want min among unvisited
        costs = C[current, :].copy()
        costs[~unvisited] = np.inf
        
        # Find index of min cost
        next_node = np.argmin(costs)
        
        if costs[next_node] == np.inf:
            break
            
        path.append(next_node)
        unvisited[next_node] = False
        current = next_node
        
    # If we didn't finish, we have a partial path. 
    # We should probably just use the standard construction if we can't finish TSP quickly.
    # But Nearest Neighbor is O(N^2) = 25M ops. Should be fast.
    
    # Try 2-opt improvements
    # This is more complex. Given time constraints, let's just use the NN path for now.
    # If we have time, we can add a simple local search.
    
    # Construct the word
    word = list(P[path[0]])
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        ov = O[prev, curr]
        word.extend(P[curr][ov:])
        
    # Verify validity
    # Check if all permutations are present
    # Sliding window
    found = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        found.add(w)
        
    # Check if found == set(perms)
    # set(perms) is expensive to create every time, but N=5040.
    # We can just check len(found) == N.
    if len(found) == N:
        return "".join(map(str, word))
    else:
        # Fallback to standard construction
        word_std = [1]
        for m in range(2, n + 1):
            perms_std, seen = [], set()
            for i in range(len(word_std) - m + 2):
                window = tuple(word_std[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_std.append(window)
            new = []
            for p in perms_std:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word_std = new
        return "".join(map(str, word_std))
