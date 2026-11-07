import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst)

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _lcm_list(_DIVISORS)
_MAX = 1000000

def _has_perfect_matching(adj):
    n = len(adj)
    if n == 0:
        return []
    match_right = {}
    match_left = [-1] * n
    
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

def construct():
    divisors = _DIVISORS
    R = _LCM
    max_val = _MAX
    n = len(divisors)
    
    # Lower bound for L
    # Max gap between multiples of d is d.
    # To fit 10 distinct numbers, L must be at least ...
    # Let's start from a reasonable lower bound.
    # The smallest divisor is 6.
    # We need 10 distinct values.
    # Just start from 1.
    
    for L in range(1, 200):  # Cap at 200, should be enough
        # Iterate v0, the multiple of divisors[0]
        # v0 can be any multiple of divisors[0]
        # We can restrict v0 to [1, R] because of periodicity
        d0 = divisors[0]
        
        found = False
        # We can optimize by only checking v0 that are "valid" start points
        # But let's just iterate all multiples of d0 in [1, R]
        # There are R/d0 = 360360/6 = 60060 multiples.
        
        for k0 in range(1, R // d0 + 1):
            v0 = k0 * d0
            
            # Build adjacency
            adj = []
            right_nodes = []
            val_to_idx = {}
            
            # For each divisor i, find multiples in [v0-L, v0+L]
            # Note: We must ensure v_i != v_j
            # The matching will handle distinctness if we map to unique values
            
            for i, d in enumerate(divisors):
                # Multiples of d in [v0-L, v0+L]
                k_start = (v0 - L + d - 1) // d
                k_end = (v0 + L) // d
                
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
                
            match = _has_perfect_matching(adj)
            if match is not None:
                rel_vals = [right_nodes[match[i]] for i in range(n)]
                
                # Check if we can shift to [1, max_val]
                min_v = min(rel_vals)
                max_v = max(rel_vals)
                
                # We can add j * R to all values
                # We need 1 <= min_v + j*R and max_v + j*R <= max_val
                # j*R >= 1 - min_v  => j >= ceil((1-min_v)/R)
                # j*R <= max_val - max_v => j <= floor((max_val-max_v)/R)
                
                j_min = math.ceil((1 - min_v) / R)
                j_max = math.floor((max_val - max_v) / R)
                
                if j_min <= j_max:
                    j = j_min
                    abs_vals = [v + j * R for v in rel_vals]
                    # Verify distinctness and bounds
                    if len(set(abs_vals)) == n and all(1 <= v <= max_val for v in abs_vals):
                        return [int(v) for v in abs_vals]
                        
        # If no solution found for this L, try larger L
        
    # Fallback
    return divisors[:]
