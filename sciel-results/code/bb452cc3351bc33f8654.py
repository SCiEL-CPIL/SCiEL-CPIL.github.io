import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try multiple random constructions
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Random greedy construction of C4-free graph
        n = 43
        adj = [set() for _ in range(n)]
        edges = []
        
        # Build graph by adding edges randomly while maintaining C4-free
        # Use a strategy: for each pair, add edge with some probability
        # but only if it doesn't create C4
        
        random.seed(attempt)
        
        # Try to build a dense C4-free graph
        # Strategy: process vertices in random order, for each vertex connect to 
        # random other vertices that don't create C4
        
        vertex_order = list(range(n))
        random.shuffle(vertex_order)
        
        for i, u in enumerate(vertex_order):
            # Try to connect u to some vertices that come later in the order
            # or to vertices that are already processed
            candidates = [v for v in vertex_order[i+1:] if v not in adj[u]]
            random.shuffle(candidates)
            
            # Also consider connecting to already processed vertices
            processed = [v for v in vertex_order[:i] if v not in adj[u]]
            random.shuffle(processed)
            candidates.extend(processed)
            
            # Try to add edges to a random subset of candidates
            # Limit degree to keep graph manageable
            max_degree = 15
            added = 0
            for v in candidates:
                if added >= max_degree:
                    break
                if time.time() > deadline - 5.0:
                    break
                    
                # Check if adding edge (u,v) creates C4
                if v in adj[u]:
                    continue
                    
                # Common neighbors of u and v
                common = adj[u].intersection(adj[v])
                
                # Adding edge (u,v) creates C4 if there exists a vertex w
                # such that w is adjacent to both u and v (already counted in common)
                # Actually, C4 is u-x-v-y-u where x,y are distinct.
                # This happens if there exist x,y such that:
                # x in N(u) \cap N(v) and y in N(u) \cap N(v) with x != y
                # OR more precisely, we need to check that the new edge doesn't
                # complete a 4-cycle.
                
                # The condition for no C4 after adding (u,v):
                # For any x in N(u) and y in N(v), we need x != y and 
                # not (x in N(v) and y in N(u)) unless x==y
                # Actually, simpler: after adding (u,v), the graph has C4 iff
                # there exist distinct x,y such that:
                # x in N(u) \ {v}, y in N(v) \ {u}, x in N(v), y in N(u)
                # This is equivalent to: |N(u) \cap N(v) \ {u,v}| >= 2
                # Wait, that's not right either.
                
                # Let me think again. A C4 is a cycle of length 4.
                # Adding edge (u,v) creates a C4 if there exist vertices a,b such that:
                # u-a-v-b-u is a cycle, i.e., a in N(u)\N(v)∪... 
                # Actually: u-a, a-v, v-b, b-u all edges.
                # So a must be in N(u) ∩ N(v) and b must be in N(u) ∩ N(v)
                # with a != b, and a,b != u,v.
                
                # So the condition is: |N(u) ∩ N(v)| <= 1 (before adding the edge)
                # Because after adding (u,v), u and v become adjacent, and if they
                # already had 2 common neighbors, those form a C4 with u,v.
                
                if len(common) <= 1:
                    # Safe to add
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
                    added += 1
        
        # Now check if alpha < 11
        alpha = independence_number(adj, n)
        
        if alpha < 11:
            if n > best_n:
                best_n = n
                best_result = {'n': n, 'edges': edges}
        
        # If we found a good result, we can try to improve
        if best_n >= 39 and time.time() > deadline - 10.0:
            break
    
    if best_result is None:
        # Fallback: return a valid small graph
        # 10 disjoint triangles = 30 vertices, alpha = 10 (one from each triangle)
        n = 30
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': n, 'edges': edges}
    
    return best_result

def independence_number(adj, n):
    """Compute the independence number of the graph."""
    # Use a branch and bound approach
    # Greedy coloring upper bound for pruning
    
    def greedy_coloring(vertices):
        """Upper bound on independence number using greedy coloring."""
        if not vertices:
            return 0
        color = {}
        next_color = 0
        for v in vertices:
            used = set()
            for u in adj[v]:
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            next_color = max(next_color, c + 1)
        return next_color
    
    best = 0
    found = [False]
    
    def dfs(start, current_set, depth):
        nonlocal best
        if found[0]:
            return
        if depth > best:
            best = depth
        if len(current_set) + (n - start) <= best:
            return
        
        # Upper bound using greedy coloring on remaining vertices
        remaining = list(range(start, n))
        # Filter to only vertices not in current_set and independent
        valid_remaining = []
        for v in remaining:
            if all(v not in adj[u] for u in current_set):
                valid_remaining.append(v)
        
        if not valid_remaining:
            return
        
        ub = greedy_coloring(valid_remaining)
        if depth + ub <= best:
            return
        
        for i in range(start, n):
            if found[0]:
                return
            # Check if i is independent from current_set
            if all(i not in adj[u] for u in current_set):
                dfs(i + 1, current_set + [i], depth + 1)
                if depth + 1 + (n - i - 1) <= best:
                    break
    
    dfs(0, [], 0)
    return best

# EVOLVE-BLOCK-END
