import time
import itertools

def construct():
    start_time = time.time()
    LIMIT = 55.0
    
    p = 101
    denominators = list(range(1, 11))
    n = 10
    
    # Precompute modular inverses
    inverses = [0] * 11
    for d in denominators:
        inverses[d] = pow(d, -1, p)
        
    # Generate all non-empty subsets and their residues
    # subsets_by_size[size] = list of (residue, subset_tuple)
    subsets_by_size = {i: [] for i in range(1, n + 1)}
    
    for size in range(1, n + 1):
        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p
            subsets_by_size[size].append((residue, combo))
            
    # Function to check if a valid assignment exists for max_size K
    def find_assignment(max_size):
        # Collect all candidate subsets
        candidates = []
        for sz in range(1, max_size + 1):
            candidates.extend(subsets_by_size[sz])
        
        # Build adjacency list for bipartite matching
        # Left side: residues 0..100
        # Right side: candidate indices 0..len(candidates)-1
        adj = [[] for _ in range(p)]
        for idx, (res, combo) in enumerate(candidates):
            adj[res].append(idx)
            
        # match_to_subset[i] = residue matched to candidate i, or -1
        match_to_subset = [-1] * len(candidates)
        # match_to_residue[r] = candidate index matched to residue r, or -1
        match_to_residue = [-1] * p
        
        # DFS for Kuhn's algorithm
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_to_subset[v] == -1 or dfs(match_to_subset[v], visited):
                    match_to_subset[v] = u
                    match_to_residue[u] = v
                    return True
            return False
            
        # Try to find a perfect matching
        for u in range(p):
            visited = [False] * len(candidates)
            if not dfs(u, visited):
                return None
                
        # If we reach here, a perfect matching exists
        result = [None] * p
        for u in range(p):
            v = match_to_residue[u]
            result[u] = list(candidates[v][1])
        return result

    # Iterate K from 1 to n
    for K in range(1, n + 1):
        if time.time() - start_time > LIMIT:
            break
        assignment = find_assignment(K)
        if assignment is not None:
            # Verify the solution
            for r in range(p):
                combo = assignment[r]
                if not combo:
                    raise ValueError("Empty combo")
                if len(set(combo)) != len(combo):
                    raise ValueError("Duplicate denominators in combo")
                s = sum(inverses[d] for d in combo)
                if s % p != r:
                    raise ValueError(f"Residue mismatch for r={r}")
                if not all(1 <= d <= 10 for d in combo):
                    raise ValueError("Denominator out of range")
            return assignment

    # Fallback if no matching found (should not happen)
    # Use greedy as a last resort
    assignments = {}
    for size in range(1, n + 1):
        for res, combo in subsets_by_size[size]:
            if res not in assignments:
                assignments[res] = combo
            if len(assignments) == p:
                break
        if len(assignments) == p:
            break
            
    result = []
    for r in range(p):
        if r in assignments:
            result.append(list(assignments[r]))
        else:
            result.append(list(denominators))
    return result
