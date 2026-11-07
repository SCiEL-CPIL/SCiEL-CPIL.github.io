"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_clique(subset, adj):
    s = list(subset)
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            if s[j] not in adj[s[i]]:
                return False
    return True

def _is_independent(subset, adj):
    s = list(subset)
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            if s[j] in adj[s[i]]:
                return False
    return True

def _is_valid_part(subset, adj):
    if len(subset) <= 1:
        return True
    return _is_clique(subset, adj) or _is_independent(subset, adj)

def _chromatic_number(adj):
    """Compute chromatic number via subset DP for 14 vertices."""
    n = P['n']
    full = (1 << n) - 1
    # Precompute which subsets are independent sets
    is_indep = [False] * (1 << n)
    is_indep[0] = True
    for mask in range(1, 1 << n):
        # check if mask is independent set
        valid = True
        for i in range(n):
            if mask & (1 << i):
                # check i has no neighbors in mask
                if adj[i] & mask:
                    valid = False
                    break
        is_indep[mask] = valid
    
    # DP: dp[mask] = min number of colors to color vertices in mask
    INF = 999
    dp = [INF] * (1 << n)
    dp[0] = 0
    for mask in range(1, 1 << n):
        # try all independent subsets of mask
        sub = mask
        while sub:
            if is_indep[sub]:
                dp[mask] = min(dp[mask], dp[mask ^ sub] + 1)
            sub = (sub - 1) & mask
    return dp[full]

def _cochromatic_lower_bound(adj):
    """Lower bound: max(chromatic number, clique number of complement = chromatic of complement)."""
    n = P['n']
    # Chromatic number of G
    chi = _chromatic_number(adj)
    # Chromatic number of complement
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    # Compute chromatic of complement
    full = (1 << n) - 1
    is_indep = [False] * (1 << n)
    is_indep[0] = True
    for mask in range(1, 1 << n):
        valid = True
        for i in range(n):
            if mask & (1 << i):
                if comp_adj[i] & mask:
                    valid = False
                    break
        is_indep[mask] = valid
    INF = 999
    dp = [INF] * (1 << n)
    dp[0] = 0
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if is_indep[sub]:
                dp[mask] = min(dp[mask], dp[mask ^ sub] + 1)
            sub = (sub - 1) & mask
    chi_comp = dp[full]
    return max(chi, chi_comp)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    adj = _build_adj()
    n = P['n']
    vertices = list(range(n))
    
    # Compute lower bound
    lb = _cochromatic_lower_bound(adj)
    
    # Try to find partition with exactly lb parts
    # Use backtracking: assign each vertex to one of k parts
    # Each part must be either a clique or independent set
    
    # Precompute for each pair whether they can be in same clique part or same indep part
    # For a part to be a clique: all pairs must be edges
    # For a part to be an independent set: no pairs can be edges
    
    # Strategy: try k from lb upward
    for k in range(lb, n + 1):
        if time.time() > deadline:
            break
        
        # Backtracking search
        parts = [[] for _ in range(k)]
        part_type = [None] * k  # None=undetermined, 'clique', 'independent'
        
        def can_add_clique(part, v):
            for u in part:
                if v not in adj[u]:
                    return False
            return True
        
        def can_add_indep(part, v):
            for u in part:
                if v in adj[u]:
                    return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = vertices[idx]
            
            # Try adding to each existing part
            for p in range(k):
                t = part_type[p]
                if t is None:
                    # Part is empty, can start as clique or independent
                    parts[p].append(v)
                    part_type[p] = 'clique'
                    if backtrack(idx + 1):
                        return True
                    parts[p].pop()
                    part_type[p] = 'independent'
                    if backtrack(idx + 1):
                        return True
                    parts[p].pop()
                    part_type[p] = None
                elif t == 'clique':
                    if can_add_clique(parts[p], v):
                        parts[p].append(v)
                        if backtrack(idx + 1):
                            return True
                        parts[p].pop()
                elif t == 'independent':
                    if can_add_indep(parts[p], v):
                        parts[p].append(v)
                        if backtrack(idx + 1):
                            return True
                        parts[p].pop()
            
            # Try starting a new part (if we have empty parts left)
            for p in range(k):
                if not parts[p]:
                    parts[p].append(v)
                    part_type[p] = 'clique'
                    if backtrack(idx + 1):
                        return True
                    parts[p].pop()
                    part_type[p] = 'independent'
                    if backtrack(idx + 1):
                        return True
                    parts[p].pop()
                    part_type[p] = None
                    break  # only need to try one empty part
            
            return False
        
        if backtrack(0):
            # Extract non-empty parts
            result = []
            for p in range(k):
                if parts[p]:
                    result.append(list(parts[p]))
            # Convert to ensure plain Python types
            result = [list(part) for part in result]
            return result
    
    # Fallback: greedy (shouldn't reach here for n=14)
    parts = []
    for v in vertices:
        placed = False
        for part in parts:
            trial = part + [v]
            if _is_valid_part(trial, adj):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
