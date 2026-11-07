import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80

    # Known 7-edge solution
    # Hub 11, connected to 1, 4, 7, 15, 18, 21, 23
    fallback = [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]
    
    # Function to check if a set of edges is valid
    def is_valid(added):
        # Check triangle free
        # Build adj
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in added:
            if len(adj[u] & adj[v]) > 0:
                return False
            adj[u].add(v)
            adj[v].add(u)
            
        # Check diameter <= 4
        from collections import deque
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
        return True

    # Search for 6-edge solution
    # Strategy: Try all centers c from 0 to 23
    # For each c, find minimal dominating sets for L and R
    # Combine them and check validity
    
    def get_dominating_sets(interval, c, min_dist):
        """
        Get all sets of vertices that dominate the interval,
        are at distance >= min_dist from c,
        and are mutually non-adjacent.
        interval is (start, end) inclusive.
        """
        start, end = interval
        if start > end:
            return [()]
            
        # Candidates must be in [start-1, end+1] to dominate
        # But must be in [0, 23]
        cand_start = max(0, start - 1)
        cand_end = min(n - 1, end + 1)
        
        candidates = []
        for v in range(cand_start, cand_end + 1):
            if abs(v - c) < min_dist:
                continue
            candidates.append(v)
            
        # We need to cover [start, end]
        # Use backtracking or simple generation if small
        # Since interval is small, we can try all subsets of size 3 or 4?
        # Actually, we can just generate all subsets of candidates of size 3 and 4
        # and check if they dominate.
        
        valid_sets = []
        # Try size 3
        for subset in combinations(candidates, 3):
            # Check mutual non-adjacency
            ok = True
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if abs(subset[i] - subset[j]) == 1:
                        ok = False
                        break
                if not ok: break
            if not ok: continue
            
            # Check domination
            dominated = set()
            for v in subset:
                dominated.update([v-1, v, v+1])
            if all(x in dominated for x in range(start, end+1)):
                valid_sets.append(subset)
                
        # Try size 4
        for subset in combinations(candidates, 4):
            ok = True
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if abs(subset[i] - subset[j]) == 1:
                        ok = False
                        break
                if not ok: break
            if not ok: continue
            
            dominated = set()
            for v in subset:
                dominated.update([v-1, v, v+1])
            if all(x in dominated for x in range(start, end+1)):
                valid_sets.append(subset)
                
        return valid_sets

    best_solution = fallback
    
    for c in range(n):
        if time.time() > deadline:
            break
            
        # L: [0, c-3]
        # R: [c+3, 23]
        
        # If c-3 < 0, L is empty
        # If c+3 > 23, R is empty
        
        L_interval = (0, c-3) if c-3 >= 0 else None
        R_interval = (c+3, 23) if c+3 <= 23 else None
        
        L_sets = [()] if L_interval is None else get_dominating_sets(L_interval, c, 3)
        R_sets = [()] if R_interval is None else get_dominating_sets(R_interval, c, 3)
        
        # We want total edges <= 6
        # So |L_set| + |R_set| <= 6
        
        for L_set in L_sets:
            for R_set in R_sets:
                if len(L_set) + len(R_set) > 6:
                    continue
                    
                edges = []
                for v in L_set:
                    edges.append((c, v))
                for v in R_set:
                    edges.append((c, v))
                    
                # Check if this specific combination is valid
                # The domination logic ensures diameter <= 4 IF the hub logic holds
                # But we should still verify with is_valid to be safe
                # is_valid is expensive, so we only call it for promising candidates
                # But there are few candidates.
                
                if is_valid(edges):
                    if len(edges) < len(best_solution):
                        best_solution = edges
                        
    return [[int(u), int(v)] for u, v in best_solution]
