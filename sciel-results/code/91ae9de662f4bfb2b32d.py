import time
import random
import itertools
import sys

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        # 1231321? No.
        # 123, 132, 213, 231, 312, 321
        # 12313212?
        # Let's just use the general algorithm for n>=3 if it's fast enough.
        # 3! = 6. Very fast.
        pass

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Generate permutations
    perms = list(itertools.permutations(range(1, n+1)))
    N = len(perms)
    
    # Map perm to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}

    # 2. Build Overlap Matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Use the dictionary trick for speed
    
    # Precompute suffixes and prefixes for all k
    # Actually, we can just iterate k from n-1 down to 0
    
    overlap = [[0]*N for _ in range(N)]
    
    # For each k, group perms by suffix of length k
    # But we need to update overlap[i][j] only if it's 0 (not set yet)
    # Since we go k=n-1 down to 0, the first set is the max.
    
    # Optimization: 
    # For a given k, create dict: suffix -> list of indices
    # Then for each i, look up suffix(i, k) in dict, and for each j in list, 
    # if overlap[i][j] == 0, set to k.
    
    # However, creating the dict for each k is O(N).
    # Total O(N * n).
    # Then the inner loop: for each i, look up, iterate matches.
    # Matches for k is ~ N/n^k.
    # Total ops ~ sum_k N * (N/n^k) ~ N^2.
    
    # Let's implement this.
    
    # To speed up, we can store perms as tuples of ints.
    # Slicing tuples is fast.
    
    for k in range(n-1, -1, -1):
        if k == 0:
            # Overlap 0 is default, no need to set
            break
            
        # Build dict for this k
        # Key: suffix of length k
        # Value: list of indices having this suffix
        # Wait, we need suffix of i to match prefix of j.
        # So Dict[Suffix(i, k)] should contain j such that Prefix(j, k) == Suffix(i, k).
        # So we should build Dict[Prefix(j, k)] -> list of j.
        
        prefix_dict = {}
        for j in range(N):
            p = perms[j]
            prefix = p[:k]
            if prefix not in prefix_dict:
                prefix_dict[prefix] = []
            prefix_dict[prefix].append(j)
            
        # Now for each i, find its suffix, look up in dict
        for i in range(N):
            p_i = perms[i]
            suffix = p_i[-k:]
            if suffix in prefix_dict:
                js = prefix_dict[suffix]
                for j in js:
                    if overlap[i][j] == 0:
                        overlap[i][j] = k

    # 3. Build Cost Matrix
    # cost[i][j] = n - overlap[i][j]
    # We can just use overlap in the cost calculation to save memory/time
    # cost(i,j) = n - overlap[i][j]

    # 4. Initialization: Greedy Nearest Neighbor
    # Start with a random node
    # Always go to the unvisited node with min cost (max overlap)
    
    visited = [False] * N
    path = []
    
    # Pick random start
    start = random.randint(0, N-1)
    path.append(start)
    visited[start] = True
    current = start
    
    for _ in range(N-1):
        best_next = -1
        best_cost = float('inf')
        # Find min cost neighbor
        # We can optimize this by pre-sorting neighbors by cost?
        # But N is 5040. N^2 is 25M. 
        # Greedy NN is O(N^2). 25M ops. Fast.
        
        for j in range(N):
            if not visited[j]:
                c = n - overlap[current][j]
                if c < best_cost:
                    best_cost = c
                    best_next = j
        
        if best_next == -1:
            # Should not happen
            break
            
        path.append(best_next)
        visited[best_next] = True
        current = best_next

    # Calculate initial length
    def calc_length(path):
        if len(path) <= 1:
            return n
        total = n
        for i in range(len(path)-1):
            total += n - overlap[path[i]][path[i+1]]
        return total

    best_path = path[:]
    best_len = calc_length(best_path)
    
    # 5. Simulated Annealing
    # Move: Swap two random elements in path
    # Cost Delta:
    #   Let path be ... L, A, M, B, R ...
    #   Swap A and B.
    #   New path ... L, B, M, A, R ...
    #   Old edges: (L,A), (A,M), (M,B), (B,R)
    #   New edges: (L,B), (B,M), (M,A), (A,R)
    #   Delta = C[L,B] + C[B,M] + C[M,A] + C[A,R] - (C[L,A] + C[A,M] + C[M,B] + C[B,R])
    #   Note: M is the segment between A and B.
    #   If A and B are adjacent, M is empty.
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Random swap
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        
        # Ensure i < j for logic simplicity? No, swap is symmetric.
        if i > j:
            i, j = j, i
        
        # Get neighbors
        # Indices in path: i and j
        # Left of i: i-1 (or None if 0)
        # Right of i: i+1 (or None if N-1)
        # Left of j: j-1 (or None if 0)
        # Right of j: j+1 (or None if N-1)
        
        # Cases:
        # 1. i and j are not adjacent.
        # 2. i and j are adjacent (j = i+1).
        
        if j == i + 1:
            # Adjacent
            # Path: ... L, A, B, R ...
            # Swap: ... L, B, A, R ...
            # Old: C[L,A] + C[A,B] + C[B,R]
            # New: C[L,B] + C[B,A] + C[A,R]
            # (Handle boundaries)
            
            L = path[i-1] if i > 0 else None
            R = path[j+1] if j < N-1 else None
            A = path[i]
            B = path[j]
            
            old_cost = 0
            new_cost = 0
            
            if L is not None:
                old_cost += n - overlap[L][A]
                new_cost += n - overlap[L][B]
            
            old_cost += n - overlap[A][B]
            new_cost += n - overlap[B][A]
            
            if R is not None:
                old_cost += n - overlap[B][R]
                new_cost += n - overlap[A][R]
                
            delta = new_cost - old_cost
        else:
            # Non-adjacent
            # Path: ... L, A, ..., B, R ...
            # Let M_left = path[i+1], M_right = path[j-1]
            # Old edges: (L,A), (A, M_left), (M_right, B), (B, R)
            # New edges: (L,B), (B, M_left), (M_right, A), (A, R)
            
            L = path[i-1] if i > 0 else None
            R = path[j+1] if j < N-1 else None
            M_left = path[i+1]
            M_right = path[j-1]
            A = path[i]
            B = path[j]
            
            old_cost = 0
            new_cost = 0
            
            if L is not None:
                old_cost += n - overlap[L][A]
                new_cost += n - overlap[L][B]
            
            old_cost += n - overlap[A][M_left]
            new_cost += n - overlap[B][M_left]
            
            old_cost += n - overlap[M_right][B]
            new_cost += n - overlap[M_right][A]
            
            if R is not None:
                old_cost += n - overlap[B][R]
                new_cost += n - overlap[A][R]
                
            delta = new_cost - old_cost

        # Accept
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            path[i], path[j] = path[j], path[i]
            best_len_candidate = best_len + delta
            if best_len_candidate < best_len:
                best_len = best_len_candidate
                best_path = path[:]
        
        T *= cooling

    # 6. Construct String
    # Use best_path
    final_word = []
    for idx in best_path:
        p = perms[idx]
        if not final_word:
            final_word.extend(p)
        else:
            # Find max overlap with current end
            # We already know the overlap from the matrix?
            # Yes, but we need to find the *actual* overlap used in the construction.
            # The matrix gives the max possible overlap.
            # In the path, we assume we used the max overlap.
            # So we can just use overlap[prev_idx][idx].
            pass
    
    # Better construction:
    # Iterate through best_path
    # Keep track of current string (or just the last n-1 chars? No, we need to append)
    # Actually, we can just reconstruct.
    
    # To be safe and fast:
    # We know the length is best_len.
    # We can build the string.
    
    # Optimization:
    # The string is formed by concatenating blocks.
    # Block 1: perms[best_path[0]]
    # Block k: perms[best_path[k]] with overlap removed.
    
    # Let's just do it.
    # N=5040. String length ~6000.
    # Appending is fast.
    
    res = list(perms[best_path[0]])
    prev_idx = best_path[0]
    for k in range(1, N):
        curr_idx = best_path[k]
        p = perms[curr_idx]
        ov = overlap[prev_idx][curr_idx]
        # Append p[ov:]
        res.extend(p[ov:])
        prev_idx = curr_idx
        
    return "".join(map(str, res))

# Test for small n
# construct(1) -> "1"
# construct(2) -> "121"
# construct(3) -> 
# Perms: 123, 132, 213, 231, 312, 321
# Optimal length for n=3 is 9? 
# 12313212? No.
# 123, 231, 312, 123...
# Known shortest for n=3 is 9: 123132121? No, 1231321 is 7?
# 123 (123)
# 231 (231) -> 123231
# 312 (312) -> 1232312
# 123 (123) -> 12323123
# 213? No.
# Let's trust the algorithm.
