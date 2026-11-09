"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses Tomita's algorithm for maximum clique with greedy coloring bound.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # Build adjacency
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            l = (i * j) // gcd(i, j)
            if l <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Initial greedy clique for lower bound
    best_clique = []
    best_size = 0
    
    current_clique = []
    remaining = set(nodes)
    while remaining:
        max_deg_node = max(remaining, key=lambda x: len(adj[x] & remaining))
        current_clique.append(max_deg_node)
        remaining = remaining & adj[max_deg_node]
    
    best_clique = current_clique[:]
    best_size = len(best_clique)
    
    def greedy_color(candidates):
        """
        Greedy coloring. Returns (color_of, order, num_colors).
        order: list of vertices, lower color first.
        """
        if not candidates:
            return {}, [], 0
        
        # Order by degree in candidate subgraph (descending)
        degs = {u: len(adj[u] & candidates) for u in candidates}
        sorted_cands = sorted(candidates, key=lambda x: degs[x], reverse=True)
        
        color_of = {}
        color_classes = []
        
        for u in sorted_cands:
            used_colors = set()
            for v in adj[u]:
                if v in candidates and v in color_of:
                    used_colors.add(color_of[v])
            
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[u] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(u)
        
        order = []
        for c in range(len(color_classes)):
            order.extend(color_classes[c])
        
        return color_of, order, len(color_classes)
    
    def expand(clique, order, color_of):
        """
        Tomita expansion: process from end of order to beginning.
        """
        nonlocal best_clique, best_size
        
        n = len(order)
        i = n - 1
        
        while i >= 0:
            if time.time() > deadline:
                return
            
            u = order[i]
            i -= 1
            
            # Prune: if even taking u and all before it can't beat best
            if len(clique) + color_of[u] + 1 <= best_size:
                return
            
            # Include u
            clique.append(u)
            
            # New candidates: neighbors of u in order[0..i]
            new_cand_list = [order[j] for j in range(i + 1) if order[j] in adj[u]]
            
            if new_cand_list:
                new_cand_set = set(new_cand_list)
                new_color_of, new_order, _ = greedy_color(new_cand_set)
                expand(clique, new_order, new_color_of)
            
            clique.pop()
    
    # Start search
    all_candidates = set(nodes)
    color_of, order, _ = greedy_color(all_candidates)
    expand([], order, color_of)
    
    return sorted(best_clique)

# EVOLVE-BLOCK-END
