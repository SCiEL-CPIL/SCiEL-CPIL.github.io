import time
import sys
from itertools import combinations
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin for safety
    
    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    def time_check():
        return time.time() > deadline

    def max_clique():
        """Find maximum clique using Bron-Kerbosch with pivot."""
        best_clique = []
        
        def bk(R, P, X):
            nonlocal best_clique
            if time_check():
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            
            # Pivot selection: choose vertex in P U X with max degree in P
            pivot = None
            max_deg = -1
            for v in P:
                deg = len(P & adj_set[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            if pivot is None:
                for v in X:
                    deg = len(P & adj_set[v])
                    if deg > max_deg:
                        max_deg = deg
                        pivot = v
            
            if pivot is not None:
                # Iterate over P \ N(pivot)
                candidates = [v for v in P if v not in adj_set[pivot]]
            else:
                candidates = list(P)
                
            for v in candidates:
                # New P = P \ N(v)
                new_P = set(P)
                new_P.discard(v)
                for u in adj_list[v]:
                    new_P.discard(u)
                new_P = P & new_P
                
                # New X = X \ N(v)
                new_X = X - set(adj_list[v])
                
                bk(R | {v}, new_P, new_X)
                
                P.discard(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return best_clique

    def find_paths_for_branches(branches):
        """Verify and construct paths for a set of branches."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Check if it's a clique
        is_clique = True
        for u, v in pairs:
            if u not in adj_set[v]:
                is_clique = False
                break
        
        if is_clique:
            return { (u, v): [u, v] for u, v in pairs }
        
        # If not a clique, try to find disjoint paths
        # This is the expensive part, so we only do it if needed
        # For now, return None if not a clique to prioritize clique finding
        # We can enhance this later if clique size is small
        
        # Simple check: if any pair has no common neighbor, it's invalid
        for u, v in pairs:
            if u not in adj_set[v]:
                common = set(adj_list[u]) & set(adj_list[v])
                common -= branch_set
                if not common:
                    return None
        
        # If we get here, it might be a valid subdivision, but we need to ensure disjointness
        # Given the time constraint and the goal to beat score 3, and knowing a 4-clique exists,
        # we will just return None for non-cliques to force the search to focus on cliques first.
        # If the max clique is < 4, we will need to implement this.
        return None

    def solve():
        # 1. Find Maximum Clique
        clique = max_clique()
        
        if len(clique) >= 2:
            paths = find_paths_for_branches(clique)
            if paths is not None:
                return {
                    'branches': [int(b) for b in clique],
                    'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                }
        
        # Fallback: try to find any valid 2-branch path
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
            
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    return solve()
# EVOLVE-BLOCK-END
