import math
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst)

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _lcm_list(_DIVISORS)
_MAX = 1000000

def _has_perfect_matching(adj):
    """
    adj: list of lists, adj[i] contains indices of right-side nodes connected to left-side node i.
    Returns a list of matches for left-side nodes, or None if no perfect matching.
    Left side size is n. Right side size is m.
    """
    n = len(adj)
    if n == 0:
        return []
    
    # Standard DFS-based bipartite matching (Kuhn's algorithm)
    match_right = {} # right_node -> left_node
    match_left = [-1] * n # left_node -> right_node
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_right or dfs(match_right[v], visited):
                match_right[v] = u
                match_left[u] = v
                return True
        return False
    
    for u in range(n):
        visited = set()
        if not dfs(u, visited):
            return None
            
    return match_left

def _check_feasibility(L, divisors, R, max_val):
    """
    Check if there exists a window of length L (i.e., range L) 
    within [1, max_val] that contains distinct multiples for all divisors.
    Returns list of values if feasible, else None.
    """
    n = len(divisors)
    
    # Iterate over all possible starting positions modulo R
    for X in range(1, R + 1):
        # Build adjacency list
        adj = []
        right_nodes = []
        val_to_idx = {}
        
        for i, d in enumerate(divisors):
            k_start = (X + d - 1) // d
            k_end = (X + L) // d
            
            adj_i = []
            for k in range(k_start, k_end + 1):
                val = k * d
                if val not in val_to_idx:
                    idx = len(right_nodes)
                    right_nodes.append(val)
                    val_to_idx[val] = idx
                adj_i.append(val_to_idx[val])
            adj.append(adj_i)
            
        if any(len(a) == 0 for a in adj):
            continue
            
        # Check if valid shift k exists
        k_min = math.ceil((1 - X) / R)
        k_max = math.floor((max_val - X - L) / R)
        
        if k_min > k_max:
            continue
            
        match = _has_perfect_matching(adj)
        if match is not None:
            rel_vals = [right_nodes[match[i]] for i in range(n)]
            k = k_min
            abs_vals = [v + k * R for v in rel_vals]
            return abs_vals
            
    return None

def construct():
    divisors = _DIVISORS
    R = _LCM
    max_val = _MAX
    
    lo = 0
    hi = max_val - 1
    best_vals = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        res = _check_feasibility(mid, divisors, R, max_val)
        if res is not None:
            best_vals = res
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_vals is None:
        return divisors[:]
        
    return [int(v) for v in best_vals]
# EVOLVE-BLOCK-END
