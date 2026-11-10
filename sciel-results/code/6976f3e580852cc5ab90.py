import time

def construct():
    start_time = time.time()
    deadline = 90.0  # Safety margin
    
    # 1. Precompute modular inverses for 1..10 modulo 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)
        
    # 2. Generate all non-empty subsets of {1..10}
    # Each subset is represented by a bitmask 1..1023
    subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + invs[x]) % 101
        subsets.append((s, terms, len(terms)))
        
    # 3. Group subsets by length to facilitate incremental checking
    # subsets_by_len[k] contains list of (residue, terms) for subsets of length k
    subsets_by_len = [[] for _ in range(11)]
    for res, terms, length in subsets:
        subsets_by_len[length].append((res, terms))
        
    # We will try to find a perfect matching using subsets of size <= k
    # for k = 1, 2, ..., 10
    # We use a simple DFS-based bipartite matching (Kuhn's algorithm)
    
    def find_matching(max_len):
        # Build adjacency list for residues 0..100
        # adj[r] = list of subset indices (into a temporary list) that match r
        # We need to map subset indices to terms.
        # Let's collect all eligible subsets (length <= max_len)
        eligible_subsets = []
        for l in range(1, max_len + 1):
            for res, terms in subsets_by_len[l]:
                eligible_subsets.append((res, terms))
                
        # To speed up matching, we can group eligible subsets by residue
        # adj[r] = list of indices in eligible_subsets that have residue r
        adj = [[] for _ in range(101)]
        for idx, (res, terms) in enumerate(eligible_subsets):
            adj[res].append(idx)
            
        # match_subset[j] = residue i that subset j is matched to, or -1
        match_subset = [-1] * len(eligible_subsets)
        # match_res[i] = subset j that residue i is matched to, or -1
        match_res = [-1] * 101
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_subset[v] == -1 or dfs(match_subset[v], visited):
                    match_subset[v] = u
                    match_res[u] = v
                    return True
            return False
            
        matched_count = 0
        # Standard Kuhn's algorithm: iterate over all left nodes (residues)
        for u in range(101):
            visited = [False] * len(eligible_subsets)
            if dfs(u, visited):
                matched_count += 1
                
        if matched_count == 101:
            # Perfect matching found
            # Reconstruct the result
            result = [None] * 101
            for u in range(101):
                v = match_res[u]
                result[u] = eligible_subsets[v][1]
            return result
        return None

    # Check for increasing max length
    for k in range(1, 11):
        if time.time() > deadline - 5:
            break
        res = find_matching(k)
        if res is not None:
            return res
            
    # Fallback: if no matching found (should not happen as k=10 covers all), 
    # return the greedy first-found solution from seed
    representations = {}
    for res, terms, length in subsets:
        if res not in representations:
            representations[res] = terms
    return [representations[r] for r in range(101)]
