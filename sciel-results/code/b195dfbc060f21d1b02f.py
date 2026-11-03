<<<<<<< SEARCH
def _tsp_construct(n, deadline):
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to strings for faster overlap checks
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix efficiently
    # overlap[i][j] = longest suffix of perm_strs[i] matching prefix of perm_strs[j]
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # For each i, compute overlaps with all j
    # Use the fact that max overlap is n-1
    max_ov = n - 1
    for i in range(N):
        si = perm_strs[i]
        # Get suffixes of si of lengths 1..max_ov
        # For each j, check if prefix of perm_strs[j] matches
        for j in range(N):
            sj = perm_strs[j]
            for l in range(max_ov, 0, -1):
                if si[-l:] == sj[:l]:
                    overlap[i][j] = l
                    break
    
    if time.time() > deadline - 15:
        return None
    
    # Nearest Neighbor with efficient unvisited tracking
    visited = np.zeros(N, dtype=bool)
    visited[0] = True
    path = [0]
    curr = 0
    
    # Track unvisited count for early termination
    unvisited_count = N - 1
    for step in range(N - 1):
        # Get overlaps from curr to all nodes
        row = overlap[curr]
        # Mask visited nodes with -1
        row_masked = row.copy()
        row_masked[visited] = -1
        next_idx = int(np.argmax(row_masked))
        if next_idx == 0 and visited[0]:
            # All visited
            break
        visited[next_idx] = True
        path.append(next_idx)
        curr = next_idx
        unvisited_count -= 1
        
        if step % 500 == 0 and time.time() > deadline - 20:
            return None
    
    if len(path) != N:
        return None
    
    # 2-Opt improvement - limited passes, early termination
    # For N=5040, full 2-opt is O(N^2) per pass. Limit to a few passes.
    improved = True
    max_passes = 5
    for p in range(max_passes):
        if time.time() > deadline - 10:
            break
        improved = False
        # Only do local 2-opt: check reversals of segments of length up to some limit
        # Full 2-opt is too slow. Use Or-opt style: try moving small segments.
        # Or do limited 2-opt with early break.
        # Let's do a simplified version: for each position i, try reversing path[i:j] for small j-i
        for i in range(1, N - 1):
            if time.time() > deadline - 8:
                break
            # Try a few j values
            for j in range(i + 1, min(i + 50, N)):
                # Compute cost change
                old_cost = 0
                new_cost = 0
                
                # Edge (i-1, i)
                old_cost += n - overlap[path[i-1]][path[i]]
                # Edge (j, j+1) if j < N-1
                if j < N - 1:
                    old_cost += n - overlap[path[j]][path[j+1]]
                
                # New edge (i-1, j)
                new_cost += n - overlap[path[i-1]][path[j]]
                # New edge (i, j+1) if j < N-1
                if j < N - 1:
                    new_cost += n - overlap[path[i]][path[j+1]]
                
                if new_cost < old_cost:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
                    i += 1  # Skip ahead after modification
                    if i >= N - 1:
                        break
        
        if not improved:
            break
    
    # Build string
    s = perm_strs[path[0]]
    for k in range(1, N):
        p_prev = perm_strs[path[k-1]]
        p_curr = perm_strs[path[k]]
        ov = overlap[path[k-1]][path[k]]
        s += p_curr[ov:]
    
    # Verify validity
    if len(s) - n + 1 < N:
        return None
    
    windows = set()
    for i in range(len(s) - n + 1):
        w = s[i:i+n]
        windows.add(w)
    
    if len(windows) == N:
        return s
    else:
        return None
=======
def _tsp_construct(n, deadline):
    """
    Solve the superpermutation problem as a TSP on n! permutations.
    Edge weight from i to j = n - overlap[i][j] where overlap is the
    longest suffix of perm_i matching prefix of perm_j.
    Uses multi-start NN + LNS (large neighborhood search).
    """
    start_time = time.time()
    time_budget = deadline - start_time - 10  # safety margin
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to strings
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Fast overlap computation using integer encoding
    # Encode each permutation as a tuple of ints
    perm_tuples = [tuple(p) for p in perms]
    
    # Precompute suffix keys and prefix keys for fast lookup
    # For overlap[i][j]: longest l in 1..n-1 such that perm_tuples[i][-l:] == perm_tuples[j][:l]
    # Build a dict: for each length l, map suffix of length l -> list of indices that have that suffix
    # And prefix of length l -> list of indices that have that prefix
    
    # Actually, more efficient: for each i, compute overlap to all j by checking
    # suffixes of i against prefixes of j.
    # Use a dict: prefix_key -> list of (index, length)
    
    max_ov = n - 1
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # For each possible overlap length l from max_ov down to 1:
    #   Build a dict: prefix of length l -> list of j that have this prefix
    #   For each i, look up suffix of i of length l in the dict
    #   This gives all j where overlap[i][j] >= l
    #   Process from longest to shortest to get exact overlap
    
    # Precompute prefixes and suffixes
    prefixes = []  # prefixes[l][j] = tuple(perm_tuples[j][:l])
    suffixes = []  # suffixes[l][i] = tuple(perm_tuples[i][-l:])
    for l in range(1, max_ov + 1):
        pref = [tuple(pt[:l]) for pt in perm_tuples]
        suff = [tuple(pt[-l:]) for pt in perm_tuples]
        prefixes.append(pref)
        suffixes.append(suff)
    
    # Process from longest overlap to shortest
    for l in range(max_ov, 0, -1):
        # Build dict: prefix tuple -> list of indices
        prefix_dict = {}
        for j in range(N):
            key = prefixes[l-1][j]  # l-1 because 0-indexed
            if key not in prefix_dict:
                prefix_dict[key] = []
            prefix_dict[key].append(j)
        
        # For each i, check if suffix matches any prefix
        for i in range(N):
            key = suffixes[l-1][i]
            if key in prefix_dict:
                for j in prefix_dict[key]:
                    if overlap[i][j] == 0:  # Only set if not already set (longer overlap found)
                        overlap[i][j] = l
    
    if time.time() > deadline - 15:
        return None
    
    # Build cost matrix: cost[i][j] = n - overlap[i][j]
    # We want to minimize total cost = sum of (n - overlap)
    # Equivalent to maximizing sum of overlaps
    # Total length = n + sum_{k=1}^{N-1} (n - overlap[path[k-1]][path[k]])
    #             = n + (N-1)*n - sum(overlaps)
    #             = N*n - sum(overlaps)
    # So maximizing sum of overlaps minimizes length.
    
    def path_cost(path):
        """Compute total overlap for a path (higher is better)."""
        total = 0
        for k in range(len(path) - 1):
            total += overlap[path[k]][path[k+1]]
        return total
    
    def path_length(path):
        """Compute word length for a path."""
        return N * n - path_cost(path)
    
    # Multi-start with nearest neighbor
    best_path = None
    best_cost = -1
    best_len = float('inf')
    
    # Number of restarts based on time budget
    # Each NN is O(N^2) with numpy, roughly 0.01-0.05s for N=5040
    # Let's do multiple restarts
    max_restarts = 20
    restart_time_budget = time_budget / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random start node
        start_node = restart if restart < N else int(np.random.randint(N))
        
        # Nearest neighbor
        visited = np.zeros(N, dtype=bool)
        visited[start_node] = True
        path = [start_node]
        curr = start_node
        
        for step in range(N - 1):
            row = overlap[curr].copy()
            row[visited] = -1
            next_idx = int(np.argmax(row))
            if next_idx == 0 and visited[0] and not visited[next_idx]:
                # Fallback: find first unvisited
                unvisited = np.where(~visited)[0]
                if len(unvisited) == 0:
                    break
                next_idx = int(unvisited[0])
            visited[next_idx] = True
            path.append(next_idx)
            curr = next_idx
            
            if step % 1000 == 0 and time.time() > deadline - 25:
                break
        
        if len(path) != N:
            continue
        
        # LNS improvement
        path = _lns_improve(path, overlap, N, n, deadline - 15)
        
        cost = path_cost(path)
        plen = path_length(path)
        
        if cost > best_cost:
            best_cost = cost
            best_len = plen
            best_path = path[:]
        
        if time.time() > deadline - 20:
            break
    
    if best_path is None:
        return None
    
    # Build string
    s = perm_strs[best_path[0]]
    for k in range(1, N):
        ov = overlap[best_path[k-1]][best_path[k]]
        s += perm_strs[best_path[k]][ov:]
    
    # Verify validity
    if len(s) - n + 1 < N:
        return None
    
    windows = set()
    for i in range(len(s) - n + 1):
        w = s[i:i+n]
        windows.add(w)
    
    if len(windows) == N:
        return s
    else:
        return None


def _lns_improve(path, overlap, N, n, deadline):
    """
    Large Neighborhood Search: repeatedly destroy a portion of the path
    and reconstruct it optimally (or near-optimally).
    """
    current_path = path[:]
    current_cost = 0
    for k in range(len(current_path) - 1):
        current_cost += overlap[current_path[k]][current_path[k+1]]
    
    best_path = current_path[:]
    best_cost = current_cost
    
    # LNS parameters
    destroy_fraction = 0.05  # destroy 5% of the path
    min_destroy = 20
    max_destroy = 200
    
    while time.time() < deadline:
        # Destroy: pick a random segment and remove it
        seg_len = max(min_destroy, min(max_destroy, int(N * destroy_fraction)))
        if seg_len >= N:
            seg_len = N // 2
        
        start_pos = int(np.random.randint(0, N - seg_len))
        end_pos = start_pos + seg_len
        
        # Extract the removed segment
        removed = current_path[start_pos:end_pos]
        # The remaining path
        remaining = current_path[:start_pos] + current_path[end_pos:]
        
        # Reconstruct: insert the removed nodes back optimally
        # Use a greedy insertion: for each node in removed, find best position
        new_path = remaining[:]
        
        for node in removed:
            # Find best position to insert node
            best_pos = 0
            best_gain = -1
            for pos in range(len(new_path) + 1):
                # Cost of inserting at pos
                gain = 0
                if pos > 0:
                    gain += overlap[new_path[pos-1]][node]
                if pos < len(new_path):
                    gain += overlap[node][new_path[pos]]
                # Subtract the cost that was there before (if pos is in middle)
                if 0 < pos < len(new_path):
                    gain -= overlap[new_path[pos-1]][new_path[pos]]
                
                if gain > best_gain:
                    best_gain = gain
                    best_pos = pos
            
            new_path.insert(best_pos, node)
        
        # Compute new cost
        new_cost = 0
        for k in range(len(new_path) - 1):
            new_cost += overlap[new_path[k]][new_path[k+1]]
        
        if new_cost > best_cost:
            best_cost = new_cost
            best_path = new_path[:]
            current_path = new_path[:]
            current_cost = new_cost
            # Increase destroy fraction slightly
            destroy_fraction = min(destroy_fraction * 1.1, 0.15)
        else:
            # Reset destroy fraction
            destroy_fraction = 0.05
    
    return best_path
>>>>>>> REPLACE

Now let me also update the `construct` function to handle timing better:

