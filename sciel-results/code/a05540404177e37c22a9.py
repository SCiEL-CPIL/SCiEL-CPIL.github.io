# EVOLVE-BLOCK-START
import sys
import time
from collections import defaultdict
from random import Random

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges):
        n_edges = len(edges)
        adj_orig = defaultdict(set)
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            
        conflict = [set() for _ in range(n_edges)]
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if u2 in adj_orig[u1] or v2 in adj_orig[u1] or u2 in adj_orig[v1] or v2 in adj_orig[v1]:
                        conflict[i].add(j)
                        conflict[j].add(i)
        return conflict

    def solve_coloring(conflict):
        n = len(conflict)
        if n == 0:
            return []
        
        # Heuristic ordering: Smallest Last
        # We will process vertices in a fixed order determined by heuristic
        # But for B&B, we usually pick dynamically.
        
        # We want to find min k such that coloring with k colors exists.
        # Lower bound: max degree + 1
        max_deg = max(len(adj) for adj in conflict)
        lb = max_deg + 1
        
        # Upper bound: Greedy coloring
        def greedy_coloring():
            colors = [-1] * n
            used = [set() for _ in range(n)]
            # Order by degree desc
            order = sorted(range(n), key=lambda x: len(conflict[x]), reverse=True)
            for v in order:
                c = 0
                while c in used[v]:
                    c += 1
                colors[v] = c
                for u in conflict[v]:
                    used[u].add(c)
            return max(colors) + 1 if colors else 0
            
        ub = greedy_coloring()
        
        best_colors = [0] * n # Placeholder
        
        # B&B Function
        # We try to color with k colors.
        # Returns True if possible, and fills 'colors'
        
        def can_color_with_k(k):
            colors = [-1] * n
            # Available colors for each vertex
            # avail[v] is a bitmask or set of colors available
            # Using sets for simplicity, might be slow.
            # For n<=240 and k<=50, sets are okay.
            
            avail = [set(range(k)) for _ in range(n)]
            
            # Pre-compute neighbors
            neighbors = [list(adj) for adj in conflict]
            
            # Heuristic: Pick vertex with fewest available colors (MRV)
            # Tie-breaker: Highest degree in remaining graph?
            
            def backtrack(uncolored_count):
                if uncolored_count == 0:
                    return True
                
                # Select vertex with minimum available colors
                min_avail_size = k + 1
                best_v = -1
                for v in range(n):
                    if colors[v] == -1:
                        sz = len(avail[v])
                        if sz == 0:
                            return False # Prune
                        if sz < min_avail_size:
                            min_avail_size = sz
                            best_v = v
                            if sz == 1:
                                break # Can't get better than 1
                
                if best_v == -1:
                    return False # Should not happen if uncolored_count > 0
                
                v = best_v
                # Try each color
                # Sort colors by least constraining? 
                # Or just iterate.
                for c in list(avail[v]):
                    # Assign v -> c
                    colors[v] = c
                    # Update neighbors
                    removed = []
                    valid = True
                    for u in neighbors[v]:
                        if colors[u] == -1:
                            if c in avail[u]:
                                avail[u].remove(c)
                                removed.append(u)
                                if len(avail[u]) == 0:
                                    valid = False
                                    break
                    
                    if valid and backtrack(uncolored_count - 1):
                        return True
                    
                    # Undo
                    colors[v] = -1
                    for u in removed:
                        avail[u].add(c)
                    # Note: if valid was False, we broke early, so we only undo those we removed.
                    # But if we broke early, we didn't process all neighbors.
                    # So we must only undo those in 'removed'.
                    # The 'removed' list contains all u where we successfully removed c.
                    # If we broke, valid is False, but we still need to undo.
                    
                return False
                
            return backtrack(n)
        
        # Binary search or linear search for min k?
        # Linear from LB to UB is safer and usually fast enough for small gaps.
        for k in range(lb, ub + 1):
            # Reset state for new k
            # We need to re-initialize avail in the function or pass it
            # Let's wrap it
            
            def try_k(k_val):
                colors = [-1] * n
                avail = [set(range(k_val)) for _ in range(n)]
                neighbors = [list(adj) for adj in conflict]
                
                def bt(uncolored):
                    if uncolored == 0:
                        return True
                    min_sz = k_val + 1
                    best_v = -1
                    for v in range(n):
                        if colors[v] == -1:
                            sz = len(avail[v])
                            if sz == 0:
                                return False
                            if sz < min_sz:
                                min_sz = sz
                                best_v = v
                                if sz == 1:
                                    break
                    if best_v == -1:
                        return False
                    v = best_v
                    # Iterate colors
                    for c in list(avail[v]):
                        colors[v] = c
                        removed = []
                        valid = True
                        for u in neighbors[v]:
                            if colors[u] == -1:
                                if c in avail[u]:
                                    avail[u].remove(c)
                                    removed.append(u)
                                    if len(avail[u]) == 0:
                                        valid = False
                                        break
                        if valid and bt(uncolored - 1):
                            return True
                        colors[v] = -1
                        for u in removed:
                            avail[u].add(c)
                    return False
                    
                if bt(n):
                    return colors
                return None
            
            res = try_k(k)
            if res is not None:
                return res
                
        # If B&B fails (shouldn't happen as UB is valid), return greedy
        return greedy_coloring_result()

    def greedy_coloring_result():
        # Re-run greedy to get the actual colors if B&B fails
        # But we need to return colors, not just count.
        # The above try_k returns colors.
        # If we reach here, it means k > ub, which is impossible.
        # So we just return a valid coloring using greedy logic again.
        pass # This function is a placeholder, we should have stored the greedy result.

    # Refactor to store greedy result
    def solve_instance(spec):
        edges = instance(*spec)
        conflict = build_conflict_graph(edges)
        n = len(edges)
        
        # Greedy for initial UB and fallback
        def greedy():
            colors = [-1] * n
            used = [set() for _ in range(n)]
            order = sorted(range(n), key=lambda x: len(conflict[x]), reverse=True)
            for v in order:
                c = 0
                while c in used[v]:
                    c += 1
                colors[v] = c
                for u in conflict[v]:
                    used[u].add(c)
            return colors
            
        greedy_colors = greedy()
        ub = max(greedy_colors) + 1 if n > 0 else 0
        max_deg = max(len(adj) for adj in conflict) if n > 0 else 0
        lb = max_deg + 1
        
        if lb >= ub:
            return greedy_colors
            
        # B&B
        neighbors = [list(adj) for adj in conflict]
        
        def bt(k_val):
            colors = [-1] * n
            avail = [set(range(k_val)) for _ in range(n)]
            
            def rec(uncolored):
                if uncolored == 0:
                    return True
                min_sz = k_val + 1
                best_v = -1
                for v in range(n):
                    if colors[v] == -1:
                        sz = len(avail[v])
                        if sz == 0:
                            return False
                        if sz < min_sz:
                            min_sz = sz
                            best_v = v
                            if sz == 1:
                                break
                if best_v == -1:
                    return False
                v = best_v
                for c in list(avail[v]):
                    colors[v] = c
                    removed = []
                    valid = True
                    for u in neighbors[v]:
                        if colors[u] == -1:
                            if c in avail[u]:
                                avail[u].remove(c)
                                removed.append(u)
                                if len(avail[u]) == 0:
                                    valid = False
                                    break
                    if valid and rec(uncolored - 1):
                        return True
                    colors[v] = -1
                    for u in removed:
                        avail[u].add(c)
                return False
                
            if rec(n):
                return colors
            return None
            
        for k in range(lb, ub + 1):
            res = bt(k)
            if res is not None:
                return res
                
        return greedy_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        out.append(solve_instance(spec))
        
    return out
# EVOLVE-BLOCK-END
