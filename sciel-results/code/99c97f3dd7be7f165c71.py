import time
import sys
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # If N is small, we can do more thorough search
    # For n=7, N=5040
    
    # Precompute overlap matrix M
    # M[i][j] = max k such that perms[i][n-k:] == perms[j][:k]
    # k ranges from 0 to n-1
    
    # Optimization: Use dictionaries for prefixes and suffixes
    # For each k, map prefix -> list of indices
    # map suffix -> list of indices
    
    # Actually, direct computation with early break is fine if we optimize the inner loop.
    # But for N=5040, N^2 is 25M.
    # Let's use the map approach to speed up M construction.
    
    M = [[0] * N for _ in range(N)]
    
    # Precompute prefix and suffix maps for each k
    # We only need to check k from n-1 down to 1.
    # For a given i, we check k=n-1, n-2, ...
    
    # To make it faster, we can precompute for each k:
    # prefix_map[k][tuple] = list of j such that perms[j][:k] == tuple
    # suffix_map[k][tuple] = list of i such that perms[i][n-k:] == tuple
    
    # But we need M[i][j].
    # For each i, for each k, find j's.
    
    # Let's build prefix_map for all k
    prefix_maps = [{} for _ in range(n)]
    suffix_maps = [{} for _ in range(n)]
    
    for k in range(1, n):
        p_map = prefix_maps[k]
        s_map = suffix_maps[k]
        for idx, p in enumerate(perms):
            pre = p[:k]
            suf = p[n-k:]
            if pre not in p_map:
                p_map[pre] = []
            p_map[pre].append(idx)
            if suf not in s_map:
                s_map[suf] = []
            s_map[suf].append(idx)
            
    # Now fill M
    # For each i, iterate k from n-1 down to 1
    # If M[i][j] is already set, skip? No, we want max k.
    # So we iterate k from high to low.
    # If we find a match, we set M[i][j] = k and mark it.
    # We need to reset M[i][j] for each i? No, M is 0 initialized.
    # But if we process k=n-1 first, and find match, M[i][j] = n-1.
    # Then for k=n-2, if match, we shouldn't overwrite with smaller k.
    # So we only set if M[i][j] == 0.
    
    for i in range(N):
        for k in range(n - 1, 0, -1):
            suf = perms[i][n-k:]
            if suf in suffix_maps[k]:
                # This suffix matches some prefixes
                # But we need to check if the prefix of j matches this suffix
                # Wait, suffix_maps[k] maps suffix to list of i's.
                # We need prefix_maps[k] to find j's.
                # So we look up suf in prefix_maps[k]?
                # Yes, if perms[j][:k] == perms[i][n-k:], then j is in prefix_maps[k][suf].
                if suf in prefix_maps[k]:
                    for j in prefix_maps[k][suf]:
                        if M[i][j] == 0:
                            M[i][j] = k
                            # We don't break the inner loop because multiple j's can match
                    # We don't break the k loop because a smaller k might match other j's
                    # But for a specific j, the max k is found first.
                    # So if M[i][j] is set, it won't be changed by smaller k.
    
    # Greedy Nearest Neighbor
    visited = [False] * N
    path = [0]
    visited[0] = True
    current_overlap_sum = 0
    
    for _ in range(N - 1):
        last = path[-1]
        best_j = -1
        best_m = -1
        for j in range(N):
            if not visited[j]:
                m = M[last][j]
                if m > best_m:
                    best_m = m
                    best_j = j
        if best_j == -1:
            break
        path.append(best_j)
        visited[best_j] = True
        current_overlap_sum += best_m
        
    # Or-opt Local Search
    # We want to maximize current_overlap_sum
    # Move node k to position p
    
    def get_overlap_sum(path):
        s = 0
        for t in range(len(path) - 1):
            s += M[path[t]][path[t+1]]
        return s
    
    # Initial sum
    current_sum = get_overlap_sum(path)
    
    improved = True
    iteration = 0
    max_iterations = 100 # Cap iterations to avoid infinite loops if no improvement logic is flawed
    
    while improved and time.time() < deadline - 5:
        improved = False
        iteration += 1
        if iteration > max_iterations:
            break
            
        for k in range(1, N - 1): # Don't move first or last? 
            # Actually, we can move any node.
            # Let's allow moving any node from 1 to N-2 (internal)
            # Moving first or last is equivalent to rotating or reversing, which might not be necessary
            # But let's try all k from 0 to N-1?
            # If k=0, prev doesn't exist.
            # Let's handle boundaries.
            
            u = path[k-1] if k > 0 else -1
            v = path[k]
            w = path[k+1] if k < N-1 else -1
            
            # Calculate removal gain
            # If k > 0 and k < N-1:
            # Remove v. Connect u and w.
            # Old: M[u][v] + M[v][w]
            # New: M[u][w]
            # Gain_remove = M[u][w] - M[u][v] - M[v][w]
            
            # If k == 0:
            # Remove v (first).
            # Old: M[v][w]
            # New: None (w becomes first)
            # Gain_remove = -M[v][w]
            
            # If k == N-1:
            # Remove v (last).
            # Old: M[u][v]
            # New: None
            # Gain_remove = -M[u][v]
            
            if k > 0 and k < N - 1:
                gain_remove = M[u][w] - M[u][v] - M[v][w]
            elif k == 0:
                gain_remove = -M[v][w]
            else: # k == N-1
                gain_remove = -M[u][v]
                
            # Try inserting v at all positions p
            # p is the index in the NEW path (after removal) where v is inserted BEFORE path[p]
            # Or simply, iterate p from 0 to N-1 in the original path, skipping k and k-1?
            # Easier: iterate p from 0 to N-1.
            # If p == k or p == k-1, skip (no change or invalid).
            # Actually, if we remove v, the path length is N-1.
            # We insert v at position p (0 to N-1).
            # If p == k, it goes back to original.
            # If p == k-1, it goes to original.
            
            for p in range(N):
                if p == k or p == k - 1:
                    continue
                    
                # Determine neighbors in the path WITHOUT v
                # If p > k, the index p in the original path corresponds to p-1 in the new path?
                # Let's just use the original path indices for neighbors, adjusting for removal.
                
                # Neighbors of insertion position p in the path where v is removed:
                # If p < k:
                #   prev_ins = path[p-1] if p > 0 else -1
                #   next_ins = path[p]
                # If p > k:
                #   prev_ins = path[p-2] if p-2 >= 0 else -1  (since v at k is removed, index p in original is p-1 in new)
                #   next_ins = path[p-1] if p-1 >= 0 else -1
                # If p == k: (skipped)
                
                # Let's map p (insertion index in 0..N-1) to neighbors in original path
                # Case 1: p < k
                if p < k:
                    a = path[p-1] if p > 0 else -1
                    b = path[p]
                else: # p > k (p != k, k-1)
                    # In the new path (v removed), the element at index p-1 (if p>0) is path[p-1] if p-1 < k, else path[p]
                    # Wait.
                    # Original: ... path[k-1], path[k](v), path[k+1] ...
                    # New: ... path[k-1], path[k+1] ...
                    # If we insert v at position p in the NEW path:
                    # If p < k:
                    #   Insert before path[p].
                    #   Neighbors: path[p-1] (or start) and path[p].
                    # If p >= k:
                    #   Insert before path[p+1] in original?
                    #   Let's say we insert v between index p and p+1 in the NEW path.
                    #   If p < k:
                    #     New path indices 0..p-1 correspond to Orig 0..p-1.
                    #     New path index p corresponds to Orig p.
                    #     So we insert before Orig p.
                    #     Neighbors: Orig p-1, Orig p.
                    #   If p >= k:
                    #     New path indices 0..k-2 correspond to Orig 0..k-2.
                    #     New path index k-1 corresponds to Orig k+1.
                    #     So if we insert at position p in New path:
                    #     If p < k:
                    #        Prev: Orig p-1 (if p>0)
                    #        Next: Orig p
                    #     If p >= k:
                    #        Let q = p - 1 (shift due to removal)
                    #        If q < k-1:
                    #           Prev: Orig q
                    #           Next: Orig q+1
                    #        Else:
                    #           Prev: Orig k+1 (if q == k-1) ... this is getting confusing.
                    
                    # Simpler approach:
                    # Just reconstruct the path for each candidate? No, too slow.
                    
                    # Let's define a function to get neighbors of insertion slot p in the path with v removed.
                    # Slot p is between element p-1 and p in the new path.
                    # If p == 0: Prev = -1, Next = New[0]
                    # If p == N-1: Prev = New[N-2], Next = -1
                    
                    # Mapping New index to Orig index:
                    # New[i] = Orig[i] if i < k
                    # New[i] = Orig[i+1] if i >= k
                    
                    # So:
                    if p < k:
                        # Insert before Orig[p]
                        a = path[p-1] if p > 0 else -1
                        b = path[p]
                    else:
                        # Insert before Orig[p+1] (if p+1 < N)
                        # Wait, if p == N-1, we append to end.
                        # If p >= k:
                        # The element at New[p] is Orig[p+1].
                        # The element at New[p-1] is:
                        #   if p-1 < k: Orig[p-1]
                        #   else: Orig[p]
                        
                        if p == N - 1:
                            # Append to end
                            # Prev is New[N-2]
                            if N-2 < k:
                                a = path[N-2]
                            else:
                                a = path[N-1] # Wait, if p=N-1, and p>=k, then New[N-2] is Orig[N-1] if N-2>=k?
                                # Let's trace:
                                # N=5, k=2 (v at 2).
                                # Orig: 0, 1, 2(v), 3, 4
                                # New: 0, 1, 3, 4
                                # Insert at p=4 (end).
                                # Prev is New[3] which is Orig[4].
                                # Next is -1.
                                a = path[N-1]
                                b = -1
                        else:
                            # Insert before New[p] which is Orig[p+1]
                            b = path[p+1]
                            if p - 1 < k:
                                a = path[p-1]
                            else:
                                a = path[p]
                
                # Calculate insertion gain
                # Old edge (if any) between a and b is removed.
                # New edges a->v and v->b are added.
                
                if a == -1 and b == -1:
                    # Path was empty? Impossible.
                    gain_insert = 0
                elif a == -1:
                    # v becomes first
                    gain_insert = M[v][b]
                elif b == -1:
                    # v becomes last
                    gain_insert = M[a][v]
                else:
                    gain_insert = M[a][v] + M[v][b] - M[a][b]
                    
                total_gain = gain_remove + gain_insert
                
                if total_gain > 0:
                    # Perform move
                    # Remove v from path
                    path.pop(k)
                    # Adjust p if p > k
                    if p > k:
                        p -= 1
                    # Insert v at p
                    path.insert(p, v)
                    current_sum += total_gain
                    improved = True
                    break # Restart k loop? Or continue?
                    # Restarting is safer for Or-opt to avoid interference
                    break
            
            if improved:
                break # Break k loop to restart from beginning
        
    # Construct string
    word = list(perms[path[0]])
    for t in range(1, len(path)):
        i = path[t-1]
        j = path[t]
        ov = M[i][j]
        word += perms[j][ov:]
        
    # Verify
    if len(word) < n:
        return "1" # Fallback
        
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
            
    if len(seen) == N:
        return "".join(map(str, word))
    else:
        # Fallback to classic
        return _classic_superperm_str(n)

def _classic_superperm_str(n):
    if n == 1:
        return "1"
    # Recursive
    def gen(n):
        if n == 1:
            return [1]
        prev = gen(n-1)
        # Get perms of 1..n-1
        perms = list(permutations(range(1, n)))
        # Map perm to string? No, list.
        # We need to find the positions?
        # Standard construction:
        # S_n = concat( p + [n] + p for p in perms ) -- No, this is too long.
        
        # Correct:
        # S_n is built from S_{n-1} by inserting n.
        # For each permutation \pi of 1..n-1, we find it in S_{n-1} and replace it with \pi, n, \pi?
        # No.
        
        # Let's use the explicit formula:
        # S_n = (1, 2, ..., n)
        # For each \pi in Perms(1..n-1):
        #   Append \pi, then n?
        
        # Actually, just use the Greedy result if it's valid.
        # If we are here, Greedy failed.
        # Return a valid (though not shortest) superperm.
        
        # Simple valid construction:
        # Concatenate all permutations.
        # Length n * n!
        # This is valid but long.
        # For n=7, 5040*7 = 35280.
        # This is valid.
        
        result = []
        for p in permutations(range(1, n+1)):
            result.extend(p)
        return "".join(map(str, result))

    return "".join(map(str, gen(n)))
