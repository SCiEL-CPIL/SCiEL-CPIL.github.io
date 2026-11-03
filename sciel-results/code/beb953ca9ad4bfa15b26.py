import random
import time
import sys

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1213121"
    if n == 4:
        return "121312141213121"
    
    # Generate all permutations
    import itertools
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map permutation tuple to index for quick lookup? 
    # We work with indices directly.
    
    # Precompute overlap matrix K[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # K[i][j] can be 0..n-1.
    # We want to MAXIMIZE sum(K[i][j]) to MINIMIZE length.
    # Length = n + sum(n - K[i][j]) = N*n - sum(K[i][j]).
    
    # Optimization: K[i][j] is determined by the suffix of i and prefix of j.
    # We can precompute for each perm, its prefixes and suffixes as tuples.
    # But N=5040, N^2=25M. 
    # Checking overlap in Python loop: 25M * 7 ops = 175M ops. 
    # This might take 10-20 seconds.
    # Let's try to compute it faster or on the fly if memory is tight? 
    # 25M ints in a list of lists is ~200MB+. Might be risky.
    # Alternative: Store K as a flat list or use numpy.
    
    # Let's use a flat list of bytes/shorts.
    # K[i][j] is small (0-6). 
    # We can use a list of arrays from numpy if available, or just a list of lists if we optimize the loop.
    
    # Actually, we don't need the full matrix if we use a heuristic that doesn't check all neighbors.
    # But for SA, we need fast cost delta.
    # Let's try to build the matrix efficiently.
    
    # Suffixes of i: perms[i][-k:]
    # Prefixes of j: perms[j][:k]
    
    # We can group permutations by their first symbol, second symbol, etc.
    # For a fixed i, the suffix is fixed. We need to find j such that prefix(j) == suffix(i).
    # If we have a dict: prefix_tuple -> list of indices j, we can find candidates quickly.
    
    prefix_map = {}
    for j, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(j)
            
    # K[i][j] is the max k where prefix_map[perms[i][-k:]] contains j.
    # Since we want max k, we can check k=n-1, n-2, ..., 1.
    # If found, K is that k. Else 0.
    
    # To speed up SA, we can precompute the "neighbors" for each i, sorted by K desc.
    # But in SA, we might need to move to any j.
    # Let's precompute the full K matrix as a list of lists of ints.
    # N=5040.
    # Memory: 5040 * 5040 * 1 byte (if we use bytes) = 25MB.
    # Using `array('b')` or just a list of lists of ints (bigger memory but faster access in PyPy? No, CPython is slow).
    # Let's use a list of `bytearray` or similar if we stick to stdlib.
    # Or just a flat list of size N*N? 25M ints.
    
    # Let's try to compute K on the fly for the initial greedy, and use a cached dictionary for SA?
    # SA moves are local. We only need K for neighbors involved in the move.
    # Move: swap i and i+1.
    # Delta = K[i-1][i+1] + K[i+1][i+2] - K[i-1][i] - K[i][i+1].
    # We need K for specific pairs.
    # We can compute K(i, j) in O(n) time. 7 comparisons.
    # This is fast enough. We don't need the full matrix.
    
    def get_overlap(p1, p2):
        # p1, p2 are tuples
        # Returns max k in 0..n-1 such that p1[-k:] == p2[:k]
        # Check from largest k downwards
        max_k = n - 1
        # We can optimize by checking the first character first.
        if p1[-1] != p2[0]:
            return 0
        for k in range(max_k, 0, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0
        
    # Initial Greedy Construction
    # This produces the standard 5913 length sequence.
    visited = [False] * N
    seq = [0]
    visited[0] = True
    current = perms[0]
    
    for _ in range(N - 1):
        best_idx = -1
        best_ov = -1
        # Find the unvisited permutation with max overlap with current
        # To make it deterministic and fast, we can iterate through all? 
        # 5040 * 5040 = 25M checks * 7 ops = 175M. ~10s.
        # We can do better.
        # We want max k.
        for k in range(n - 1, 0, -1):
            suffix = current[-k:]
            candidates = prefix_map.get(suffix, [])
            for idx in candidates:
                if not visited[idx]:
                    best_idx = idx
                    best_ov = k
                    break
            if best_idx != -1:
                break
        
        if best_idx == -1:
            # No overlap, pick any unvisited
            for i in range(N):
                if not visited[i]:
                    best_idx = i
                    best_ov = 0
                    break
        
        visited[best_idx] = True
        seq.append(best_idx)
        current = perms[best_idx]
        
    # Calculate initial length
    def calc_length(seq):
        length = n
        for i in range(len(seq) - 1):
            p1 = perms[seq[i]]
            p2 = perms[seq[i+1]]
            ov = get_overlap(p1, p2)
            length += n - ov
        return length
        
    current_len = calc_length(seq)
    
    # Simulated Annealing
    # Time limit 120s. 
    # Greedy took ~?s. Let's assume 5-10s.
    # We have ~110s for SA.
    
    start_time = time.time()
    time_limit = 110
    
    # Temperature
    T = 5.0 # Start high
    T_min = 0.001
    cooling = 0.99999 # Very slow cooling
    
    best_seq = seq[:]
    best_len = current_len
    
    # Move generation:
    # 1. Swap adjacent
    # 2. Move element to adjacent position
    # 3. 2-opt (reverse segment) - might be too complex to compute delta quickly without full matrix?
    #    Delta for 2-opt is larger.
    # Let's stick to Swap and Move.
    
    while time.time() - start_time < time_limit:
        # Pick a random move
        if random.random() < 0.7:
            # Swap adjacent
            i = random.randint(0, N - 2)
            j = i + 1
            
            # Calculate delta
            # Old edges: (i-1, i), (i, j), (j, i+1)
            # New edges: (i-1, j), (j, i), (i, i+1)
            
            # Get indices
            idx_i = seq[i]
            idx_j = seq[j]
            
            if i > 0:
                idx_im1 = seq[i-1]
            else:
                idx_im1 = -1
                
            if j < N - 1:
                idx_jp1 = seq[j+1]
            else:
                idx_jp1 = -1
                
            # Old Cost parts
            old_cost = 0
            if i > 0:
                old_cost += n - get_overlap(perms[idx_im1], perms[idx_i])
            old_cost += n - get_overlap(perms[idx_i], perms[idx_j])
            if j < N - 1:
                old_cost += n - get_overlap(perms[idx_j], perms[idx_jp1])
                
            # New Cost parts
            new_cost = 0
            if i > 0:
                new_cost += n - get_overlap(perms[idx_im1], perms[idx_j])
            new_cost += n - get_overlap(perms[idx_j], perms[idx_i])
            if j < N - 1:
                new_cost += n - get_overlap(perms[idx_i], perms[idx_jp1])
                
            delta = new_cost - old_cost
        else:
            # Move element i to i+1 (or i-1)
            # Let's just do swap for simplicity and speed.
            # Or move: take element at i, insert at i+1.
            # This is effectively a swap if we view it as (i, i+1) -> (i+1, i).
            # So swap covers adjacent moves.
            # Let's add a "long jump" move: move element i to random position k.
            # This is O(N) to compute delta if we don't have full matrix.
            # Too slow. Stick to adjacent swaps.
            
            # Let's try a 2-element reversal (swap) but on a wider window?
            # No, just swap.
            
            # If we want more diversity, we can do a "3-opt" or "rotation" of 3 elements.
            # i, i+1, i+2 -> i+1, i+2, i
            # Let's implement a 3-rotation.
            if random.random() < 0.5:
                i = random.randint(0, N - 3)
                j = i + 1
                k = i + 2
                
                a = seq[i]
                b = seq[j]
                c = seq[k]
                
                # Old: (i-1, a), (a, b), (b, c), (c, k+1)
                # New: (i-1, b), (b, c), (c, a), (a, k+1)
                
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    idx_im1 = seq[i-1]
                    old_cost += n - get_overlap(perms[idx_im1], perms[a])
                    new_cost += n - get_overlap(perms[idx_im1], perms[b])
                else:
                    # Start of sequence
                    # No prefix cost, but we assume we just start with the perm.
                    # The length calculation adds n for the first perm.
                    # In delta calculation, we only look at edges.
                    # The first element doesn't have a left neighbor edge in the sum.
                    pass
                    
                old_cost += n - get_overlap(perms[a], perms[b])
                old_cost += n - get_overlap(perms[b], perms[c])
                
                new_cost += n - get_overlap(perms[b], perms[c]) # This edge stays? No.
                # Old edges: a-b, b-c
                # New edges: b-c, c-a
                # Wait.
                # Sequence: ... x, a, b, c, y ...
                # Edges: x-a, a-b, b-c, c-y
                # Rotate: x, b, c, a, y
                # Edges: x-b, b-c, c-a, a-y
                # Edge b-c is in both.
                # Old: ov(x,a) + ov(a,b) + ov(b,c) + ov(c,y)
                # New: ov(x,b) + ov(b,c) + ov(c,a) + ov(a,y)
                # Delta = ov(x,b) + ov(c,a) + ov(a,y) - ov(x,a) - ov(a,b) - ov(c,y)
                # Cost = n - ov.
                # Delta Cost = -Delta Ov.
                
                new_cost += n - get_overlap(perms[c], perms[a])
                
                if k < N - 1:
                    idx_kp1 = seq[k+1]
                    old_cost += n - get_overlap(perms[c], perms[idx_kp1])
                    new_cost += n - get_overlap(perms[a], perms[idx_kp1])
                else:
                    pass
                
                delta = new_cost - old_cost
                
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    seq[i] = b
                    seq[j] = c
                    seq[k] = a
                    current_len += delta
                    if current_len < best_len:
                        best_len = current_len
                        best_seq = seq[:]
            else:
                # Simple swap
                i = random.randint(0, N - 2)
                j = i + 1
                a = seq[i]
                b = seq[j]
                
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    idx_im1 = seq[i-1]
                    old_cost += n - get_overlap(perms[idx_im1], perms[a])
                    new_cost += n - get_overlap(perms[idx_im1], perms[b])
                    
                old_cost += n - get_overlap(perms[a], perms[b])
                new_cost += n - get_overlap(perms[b], perms[a])
                
                if j < N - 1:
                    idx_jp1 = seq[j+1]
                    old_cost += n - get_overlap(perms[b], perms[idx_jp1])
                    new_cost += n - get_overlap(perms[a], perms[idx_jp1])
                
                delta = new_cost - old_cost
                
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    seq[i] = b
                    seq[j] = a
                    current_len += delta
                    if current_len < best_len:
                        best_len = current_len
                        best_seq = seq[:]
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Reconstruct string from best_seq
    result = list(perms[best_seq[0]])
    for i in range(1, len(best_seq)):
        p_prev = perms[best_seq[i-1]]
        p_curr = perms[best_seq[i]]
        ov = get_overlap(p_prev, p_curr)
        result.extend(p_curr[ov:])
        
    return "".join(map(str, result))
