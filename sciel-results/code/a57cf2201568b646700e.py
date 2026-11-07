"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    P = 27720  # LCM(1, ..., 12)
    n = 12
    
    # Precompute valid candidates for each divisor i
    # We need to find min L for each residue r
    # f(12, m) is min L s.t. matching exists in (m, m+L]
    # We want max over m of this min L.
    
    # For each residue r in 0..P-1, find min L
    # We can simulate the window expanding.
    
    max_L_found = 0
    best_r = 0
    
    # Helper: Check if bipartite matching exists using efficient adjacency
    # Left: 1..12, Right: numbers in (r, r+L]
    
    def has_matching(r, L):
        # Build adjacency efficiently using arithmetic progression
        adj = [[] for _ in range(n)]
        
        for i in range(n):
            divisor = i + 1
            # First value in (r, r+L] divisible by divisor
            rem = r % divisor
            if rem == 0:
                first_val = r + divisor
            else:
                first_val = r + 1 + (divisor - 1 - rem)
            
            if first_val > r + L:
                adj[i] = []
                continue
                
            # Values: first_val, first_val+divisor, first_val+2*divisor, ... <= r+L
            val = first_val
            while val <= r + L:
                j = val - (r + 1)  # index in [0, L-1]
                if 0 <= j < L:
                    adj[i].append(j)
                val += divisor
        
        # Bipartite matching - augmenting path with bitset optimization
        match_right = [-1] * L  # match_right[j] = i if right node j is matched to left i
        
        # Use iterative DFS to avoid recursion overhead
        def find_augmenting_path(start_u):
            # BFS/DFS to find augmenting path
            parent = [-1] * n  # parent[u] = v that led to u
            visited_u = [False] * n
            visited_v = [False] * L
            
            # Simple DFS
            stack = [start_u]
            visited_u[start_u] = True
            parent[start_u] = -1
            
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not visited_v[v]:
                        visited_v[v] = True
                        if match_right[v] == -1:
                            # Found augmenting path
                            # Trace back
                            cur_v = v
                            while cur_v != -1:
                                cur_u = parent.get(cur_v, None) if isinstance(parent, dict) else None
                                # This approach is getting complex, let's use simpler DFS
                            return True
                        else:
                            matched_u = match_right[v]
                            if not visited_u[matched_u]:
                                visited_u[matched_u] = True
                                parent[matched_u] = v
                                stack.append(matched_u)
            return False
        
        # Simpler: use the original DFS but with early termination
        def dfs(u, visited_v):
            for v in adj[u]:
                if not visited_v[v]:
                    visited_v[v] = True
                    if match_right[v] == -1 or dfs(match_right[v], visited_v):
                        match_right[v] = u
                        return True
            return False
        
        matching_count = 0
        for u in range(n):
            visited_v = [False] * L
            if dfs(u, visited_v):
                matching_count += 1
            if matching_count == n:
                return True
        return False

    # Iterate through all residues
    for r in range(P):
        if time.time() > deadline:
            break
            
        # Find minimal L for this r
        # We want the MINIMAL L that WORKS.
        # Start L=1, check if matching exists. If yes, f(12, m) = L.
        # If no, L=2, etc.
        
        found_L = None
        # Upper bound for L
        for L in range(1, 200):
            if has_matching(r, L):
                found_L = L
                break
        
        if found_L is not None:
            if found_L > max_L_found:
                max_L_found = found_L
                best_r = r
                if max_L_found >= 144:
                    break

    # Find m <= 1000000 such that m % P == best_r
    m = best_r
    if m > 1000000:
        m = 0
        
    return int(m)

# EVOLVE-BLOCK-END
