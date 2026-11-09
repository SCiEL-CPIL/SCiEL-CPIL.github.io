import itertools
import random
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # ov[i][j] = max k such that perms[i][n-k:] == perms[j][:k]
    # We want to maximize sum of overlaps.
    # Overlap 0 is always possible.
    
    # Optimization: Store perms as lists for faster slicing? Tuples are fine.
    # Let's precompute suffixes and prefixes?
    # For each perm p, p_suffix[k] = p[n-k:] for k=1..n-1
    # p_prefix[k] = p[:k] for k=1..n-1
    
    # Actually, simple check is faster for n=7.
    # ov[i][j]
    ov = [[0]*N for _ in range(N)]
    
    # We can parallelize this? No, standard library only.
    # Let's just loop.
    # To speed up, only compute if last char of i matches first char of j.
    
    # Pre-extract first and last chars
    first_chars = [p[0] for p in perms]
    last_chars = [p[-1] for p in perms]
    
    # Build adjacency list for potential overlaps > 0?
    # Or just compute full matrix.
    # 5040 * 5040 = 25.4 million entries.
    # In Python, a 2D list of ints is memory heavy (~200MB+).
    # We can use a flat list or array of bytes (max overlap 6).
    # Use array('b') or just a list of lists of ints.
    # Let's use a list of lists.
    
    # Time check: if N is large, skip precomputation and do on-the-fly?
    # For n=7, N=5040.
    # Let's try to compute.
    
    # To save memory and time, we can compute overlaps on the fly during greedy?
    # Greedy needs to find max overlap among unvisited nodes.
    # If we compute on the fly, we do N checks per step. N steps. N^2 total.
    # Same complexity.
    # But precomputation allows fast local search.
    
    # Let's use a dictionary for sparse storage?
    # Most overlaps are 0 or 1.
    # Let's just use a list of arrays of bytes.
    import array
    
    ov = [array.array('b', [0]*N) for _ in range(N)]
    
    # Compute overlaps
    # Iterate i
    for i in range(N):
        pi = perms[i]
        li = last_chars[i]
        # Iterate j
        # Optimization: only check j where first_chars[j] == li
        # We can pre-group indices by first char.
        # But let's just do the full loop first to see speed.
        # If too slow, we will optimize.
        
        # Actually, let's group by first char to speed up precomputation.
        # indices_by_first_char = {c: [j for j, p in enumerate(perms) if p[0]==c]}
        
    # Let's implement the grouped version for precomputation.
    indices_by_first = {c: [] for c in range(1, n+1)}
    for j, p in enumerate(perms):
        indices_by_first[p[0]].append(j)
        
    for i in range(N):
        pi = perms[i]
        li = last_chars[i]
        candidates = indices_by_first[li]
        # For each candidate j, compute overlap
        for j in candidates:
            pj = perms[j]
            # Check max k
            # k can be 1 to n-1
            # We know k=1 matches because last_chars[i] == first_chars[j]
            max_k = 1
            # Check larger k
            # pi[n-k:] == pj[:k]
            # Try k=n-1 down to 2
            for k in range(n-1, 1, -1):
                if pi[n-k:] == pj[:k]:
                    max_k = k
                    break
            ov[i][j] = max_k
            
    # Check time
    if time.time() > deadline - 20:
        # If precomputation took too long, just return a simple construction?
        # Or proceed with what we have?
        # For n=7, it should be fast enough.
        pass
        
    # Greedy Nearest Neighbor
    def greedy_start(start_idx):
        visited = [False] * N
        path = [start_idx]
        visited[start_idx] = True
        curr = start_idx
        
        # Pre-compute list of unvisited?
        # Just iterate through all and pick best unvisited.
        # N=5040. 5040 iterations * 5040 checks = 25M ops.
        # This will be slow in Python if done naively 5040 times.
        
        # Optimization: Keep a list of unvisited nodes.
        unvisited = list(range(N))
        unvisited.remove(start_idx)
        
        while unvisited:
            best_j = -1
            best_ov = -1
            # Iterate unvisited
            # We can filter unvisited by first char matching last char of curr?
            # last_curr = last_chars[curr]
            # candidates = [j for j in unvisited if first_chars[j] == last_curr]
            # This reduces search space significantly.
            
            last_curr = last_chars[curr]
            # Filter unvisited
            # This is O(N) per step. Total O(N^2).
            # But the constant factor is smaller.
            
            for j in unvisited:
                if first_chars[j] != last_curr:
                    continue
                ov_val = ov[curr][j]
                if ov_val > best_ov:
                    best_ov = ov_val
                    best_j = j
            
            if best_j == -1:
                # No overlap > 0 found? Should not happen as ov=0 is always possible
                # But our loop only checks candidates with matching first char.
                # If no match, we must pick an arbitrary unvisited node.
                # Actually, if no candidate has matching first char, we pick any.
                # But ov=0.
                # Let's just pick the first unvisited.
                best_j = unvisited[0]
                best_ov = 0
            else:
                # If multiple have same best_ov, pick one randomly?
                # For now, pick the first one found with max.
                pass
            
            # Remove best_j from unvisited
            unvisited.remove(best_j)
            path.append(best_j)
            curr = best_j
            
        return path

    # Local Search (2-opt style swap)
    def local_search(path):
        # path is list of indices
        # Calculate total overlap
        def get_total_overlap(p):
            total = 0
            for i in range(len(p)-1):
                total += ov[p[i]][p[i+1]]
            return total
        
        curr_path = path[:]
        curr_ov = get_total_overlap(curr_path)
        improved = True
        while improved:
            improved = False
            # Try swapping adjacent elements
            for i in range(len(curr_path)-1):
                # Swap curr_path[i] and curr_path[i+1]
                a = curr_path[i]
                b = curr_path[i+1]
                
                # Calculate change in overlap
                # Old: ov[prev][a] + ov[a][b] + ov[b][next]
                # New: ov[prev][b] + ov[b][a] + ov[a][next]
                
                prev_idx = curr_path[i-1] if i > 0 else -1
                next_idx = curr_path[i+2] if i+2 < len(curr_path) else -1
                
                old_val = 0
                new_val = 0
                
                if i > 0:
                    old_val += ov[prev_idx][a]
                    new_val += ov[prev_idx][b]
                
                old_val += ov[a][b]
                new_val += ov[b][a]
                
                if i+2 < len(curr_path):
                    old_val += ov[b][next_idx]
                    new_val += ov[a][next_idx]
                
                if new_val > old_val:
                    # Swap
                    curr_path[i], curr_path[i+1] = b, a
                    curr_ov += new_val - old_val
                    improved = True
                    # Restart loop or continue?
                    # Continuing might miss improvements elsewhere, but restarting is safer.
                    # Given time constraints, let's just break and restart the while loop.
                    break
            if time.time() > deadline - 5:
                break
                
        return curr_path, curr_ov

    # Main Logic
    best_path = None
    best_len = float('inf')
    
    # Run multiple greedy restarts
    num_restarts = 50 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        start_idx = random.randint(0, N-1)
        path = greedy_start(start_idx)
        
        # Calculate length
        # Length = n + sum(n - ov)
        # = n * N - sum(ov)
        total_ov = 0
        for i in range(len(path)-1):
            total_ov += ov[path[i]][path[i+1]]
        
        curr_len = n * N - total_ov
        
        if curr_len < best_len:
            best_len = curr_len
            best_path = path[:]
            
        # Local search on this path
        if time.time() > deadline - 10:
            break
            
        ls_path, ls_ov = local_search(path)
        ls_len = n * N - ls_ov
        
        if ls_len < best_len:
            best_len = ls_len
            best_path = ls_path[:]

    # Construct final string
    if best_path is None:
        # Fallback to simple construction if something went wrong
        word = [1]
        for m in range(2, n + 1):
            perms_m, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_m.append(window)
            new = []
            for p in perms_m:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Build string from best_path
    # Start with first permutation
    res = list(perms[best_path[0]])
    for i in range(1, len(best_path)):
        prev_p = perms[best_path[i-1]]
        curr_p = perms[best_path[i]]
        # Append overlap
        ov_val = ov[best_path[i-1]][best_path[i]]
        # Append curr_p[ov_val:]
        res.extend(curr_p[ov_val:])
        
    return "".join(map(str, res))
