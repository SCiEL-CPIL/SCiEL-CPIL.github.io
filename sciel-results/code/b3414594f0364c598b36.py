import time
import sys
from collections import deque

DEADLINE_MARGIN = 10.0  # seconds before hard limit to start returning
HARD_LIMIT = 150.0      # seconds

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and (w in adj[u]):
                    return False
    return True

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with bitmasks."""
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    colors = [None] * n
    
    def backtrack(idx):
        if time.time() > DEADLINE - 2:
            return None
        if idx == n:
            return True
        
        v = order[idx]
        forbidden = 0
        for nb in adj[v]:
            c = colors[nb]
            if c is not None:
                forbidden |= (1 << c)
        
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = None
        
        return False
    
    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    def build_grotzsch():
        """Build the Grötzsch graph (Mycielski of C5) on vertices 0-10."""
        adj = [set() for _ in range(n)]
        
        def add_edge(u, v):
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        
        # C5: 0-1-2-3-4-0
        for i in range(5):
            add_edge(i, (i + 1) % 5)
        
        # Mycielski layer: 5-9 where 5+i connects to neighbors of i in C5
        for i in range(5):
            neighbors = [(i + 1) % 5, (i - 1) % 5]
            for nb in neighbors:
                add_edge(5 + i, nb)
        
        # Universal vertex 10 connects to all 5-9
        for i in range(5):
            add_edge(10, 5 + i)
        
        return adj
    
    def can_add_edge(adj, u, v):
        if u in adj[v]:
            return False
        common = adj[u] & adj[v]
        if common:
            return False
        return True
    
    def add_edge_to_adj(adj, u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_count(adj):
        return sum(len(adj[u]) for u in range(n)) // 2
    
    def greedy_add_edges(adj, vertex_order):
        """Add edges greedily in the given order of vertex pairs."""
        for u, v in vertex_order:
            if can_add_edge(adj, u, v):
                add_edge_to_adj(adj, u, v)
        return adj
    
    def generate_edge_orderings():
        """Generate different orderings of potential edges to try."""
        # All possible edges not in the base graph
        base = build_grotzsch()
        potential = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in base[u]:
                    potential.append((u, v))
        
        # Multiple orderings: by degree difference, random-ish, etc.
        orderings = []
        
        # Ordering 1: sort by sum of degrees in base
        def deg_sum(e):
            return len(base[e[0]]) + len(base[e[1]])
        orderings.append(sorted(potential, key=deg_sum, reverse=True))
        
        # Ordering 2: sort by min degree
        def deg_min(e):
            return min(len(base[e[0]]), len(base[e[1]]))
        orderings.append(sorted(potential, key=deg_min, reverse=True))
        
        # Ordering 3: natural order
        orderings.append(potential)
        
        # Ordering 4: reverse natural
        orderings.append(list(reversed(potential)))
        
        # Ordering 5: prioritize edges involving new vertices 11-15
        def new_vertex_priority(e):
            u, v = e
            score = 0
            if u >= 11: score += 10
            if v >= 11: score += 10
            return score
        orderings.append(sorted(potential, key=new_vertex_priority, reverse=True))
        
        return orderings
    
    # Generate orderings
    orderings = generate_edge_orderings()
    
    best_adj = None
    best_count = 0
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        adj = build_grotzsch()
        adj = greedy_add_edges(adj, ordering)
        
        # Verify triangle-free
        if not _is_triangle_free(adj):
            continue
        
        # Verify not 3-colorable
        if _is_3_colorable(adj, n):
            continue
        
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
    
    # If we found a good solution, try to improve it with local search
    if best_adj is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if can_add_edge(best_adj, u, v):
                        # Try adding
                        add_edge_to_adj(best_adj, u, v)
                        if _is_triangle_free(best_adj) and not _is_3_colorable(best_adj, n):
                            improved = True
                        else:
                            # Remove
                            best_adj[u].discard(v)
                            best_adj[v].discard(u)
    
    # If no valid solution found, fall back to simple greedy
    if best_adj is None:
        adj = build_grotzsch()
        ordering = orderings[0]
        adj = greedy_add_edges(adj, ordering)
        best_adj = adj
        best_count = edge_count(adj)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
