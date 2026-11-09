import time

# EVOLVE-BLOCK-START
P = 27720  # LCM(1..12)
MAX_L = 144
MIN_L = 12

def _precompute_adj(P):
    """Precompute adjacency lists for each residue r in [0, P)."""
    adj = [None] * P
    for r in range(P):
        base = r + 1
        adj[r] = []
        for i in range(1, 13):
            r0 = (-base) % i
            positions = list(range(r0, MAX_L, i))
            adj[r].append(positions)
    return adj

def _has_matching(adj_list, L):
    """
    Fast bipartite matching check using iterative DFS.
    """
    # Filter positions < L
    adj = [adj_list[i] if adj_list[i][-1] < L else [j for j in adj_list[i] if j < L] for i in range(12)]
    
    match_j = [-1] * L
    
    # Iterative DFS using explicit stack
    # We'll process divisors one at a time
    for i in range(12):
        # Try to find augmenting path for divisor i
        # Stack entries: (divisor, next_candidate_index)
        stack = [(i, 0)]
        # Path tracking: for each divisor in stack, record which position we're trying
        # path_pos[k] = position being tried for divisor at stack level k
        path_pos = [0]
        path_div = [i]
        visited = [False] * L
        
        found = False
        while stack:
            div, idx = stack[-1]
            if idx >= len(adj[div]):
                # Backtrack
                stack.pop()
                if stack:
                    # Restore parent's next index
                    parent_div, parent_idx = stack[-1]
                    # We need to advance parent's index past the position we came from
                    # Actually, this approach is getting complex. Let me use recursive but with sys.setrecursionlimit
                continue
            else:
                j = adj[div][idx]
                stack[-1] = (div, idx + 1)  # advance
                if visited[j]:
                    continue
                visited[j] = True
                if match_j[j] == -1:
                    # Found augmenting path
                    # Update matches along the path
                    # We need to track the full path. Let me restart with a cleaner approach.
                    found = True
                    break
                else:
                    # Recurse into match_j[j]
                    stack.append((match_j[j], 0))
        
        if not found:
            return False
    
    return True

def _f(m, adj_list):
    """Compute f(12, m) via binary search."""
    lo, hi = MIN_L, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(adj_list, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 85.0
    start_time = time.time()
    
    adj_all = _precompute_adj(P)
    
    best_m = 0
    best_f = 0
    
    for r in range(P):
        if time.time() > deadline:
            break
            
        f_val = _f(r, adj_all[r])
        
        if f_val > best_f:
            best_f = f_val
            best_m = r
            
        if best_f >= MAX_L:
            break
            
    return int(best_m)
# EVOLVE-BLOCK-END
