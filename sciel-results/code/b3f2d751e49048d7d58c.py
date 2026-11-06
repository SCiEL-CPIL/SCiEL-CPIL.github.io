import itertools
import random
import sys

def construct():
    """
    Constructs a C4-free graph on up to 43 vertices with independence number <= 10.
    Starts with the polarity graph of PG(2,5) and uses backtracking to add vertices.
    """
    q = 5
    
    # 1. Generate base graph (Polarity graph of PG(2,5))
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n_base = len(points)
    
    edges = []
    # Adjacency list
    adj = [set() for _ in range(n_base)]
    
    for i in range(n_base):
        for j in range(i + 1, n_base):
            u = points[i]
            v = points[j]
            dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    current_n = n_base
    current_adj = [set(s) for s in adj] # Copy
    
    # Helper: Check if a set S is C4-compatible in the current graph
    def is_c4_compatible(S):
        S_list = list(S)
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i+1, len(S_list)):
                v = S_list[j]
                if current_adj[u].intersection(current_adj[v]):
                    return False
        return True

    # Helper: Find Maximum Independent Set size in induced subgraph on vertices V_set
    def max_independent_set_size(V_set):
        if not V_set:
            return 0
        
        V_list = list(V_set)
        best = 0
        
        def dfs(remaining, count):
            nonlocal best
            if not remaining:
                if count > best:
                    best = count
                return
            if count + len(remaining) <= best:
                return
            
            # Pick vertex with max degree in 'remaining'
            max_deg = -1
            chosen_v = None
            for v in remaining:
                deg = len(current_adj[v].intersection(remaining))
                if deg > max_deg:
                    max_deg = deg
                    chosen_v = v
            
            if chosen_v is None:
                 return

            # Option 1: Include chosen_v
            neighbors_in_rem = current_adj[chosen_v].intersection(remaining)
            new_remaining = remaining - neighbors_in_rem - {chosen_v}
            dfs(new_remaining, count + 1)
            
            # Option 2: Exclude chosen_v
            dfs(remaining - {chosen_v}, count)

        dfs(set(V_list), 0)
        return best

    # Helper: Try to add a new vertex with neighbor set S
    def try_add_vertex(S):
        nonlocal current_n, current_adj, edges
        if not is_c4_compatible(S):
            return False
        
        V_set = set(range(current_n)) - set(S)
        if max_independent_set_size(V_set) > 9:
            return False
            
        new_v = current_n
        current_adj.append(set(S))
        for u in S:
            current_adj[u].add(new_v)
            edges.append([u, new_v])
        current_n += 1
        return True

    # Expansion Strategy:
    # Find a maximal independent set S in the conflict graph H (where u~v if N(u)∩N(v)≠∅)
    # such that alpha(G \ S) <= 9.
    
    for _ in range(43 - current_n):
        added = False
        
        # Build conflict graph H
        H_adj = [set() for _ in range(current_n)]
        for i in range(current_n):
            for j in range(i+1, current_n):
                if current_adj[i].intersection(current_adj[j]):
                    H_adj[i].add(j)
                    H_adj[j].add(i)
        
        # 1. Try random maximal independent sets
        for _ in range(50):
            remaining = set(range(current_n))
            S = set()
            while remaining:
                v = random.choice(list(remaining))
                S.add(v)
                remaining = remaining - {v} - H_adj[v]
            if try_add_vertex(S):
                added = True
                break
        
        if added:
            continue
            
        # 2. If random fails, try backtracking search for a valid S
        # We search for a maximal independent set in H that satisfies the alpha condition.
        nodes_explored = 0
        limit = 5000
        
        def backtrack(candidates, S_current):
            nonlocal nodes_explored
            if nodes_explored > limit:
                return None
            nodes_explored += 1
            
            if not candidates:
                # Maximal independent set found
                if try_add_vertex(S_current):
                    return S_current
                return None
            
            # Pick a vertex to branch on
            # Choose vertex with max degree in H induced by candidates?
            # Or just first one.
            v = next(iter(candidates))
            
            # Branch 1: Include v
            new_candidates = candidates - {v} - H_adj[v]
            res = backtrack(new_candidates, S_current | {v})
            if res:
                return res
                
            # Branch 2: Exclude v
            res = backtrack(candidates - {v}, S_current)
            if res:
                return res
                
            return None

        res = backtrack(set(range(current_n)), set())
        if res:
            added = True
            
        if not added:
            break

    return {'n': current_n, 'edges': edges}
