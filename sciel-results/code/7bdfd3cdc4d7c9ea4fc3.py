"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from collections import defaultdict, deque
    import random
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are connected by an edge."""
        m = len(edges)
        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check shared vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict
    
    def find_max_independent_set(conflict, remaining):
        """
        Find a large independent set in the conflict graph restricted to 'remaining' vertices.
        Uses greedy min-degree selection followed by local search improvement.
        """
        rem_set = set(remaining)
        rem_list = list(remaining)
        rem_to_local = {i: k for k, i in enumerate(rem_list)}
        n_rem = len(rem_list)
        
        if n_rem == 0:
            return set()
        
        # Build local conflict graph
        local_conflict = [set() for _ in range(n_rem)]
        for ki, i in enumerate(rem_list):
            for j in conflict[i]:
                if j in rem_set:
                    kj = rem_to_local[j]
                    local_conflict[ki].add(kj)
        
        # Phase 1: Greedy min-degree selection
        selected = set()
        active = set(range(n_rem))
        
        while active:
            # Find vertex with minimum degree in active subgraph
            min_deg = float('inf')
            chosen = None
            for v in active:
                deg = len(local_conflict[v] & active)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            
            if chosen is None:
                break
                
            selected.add(chosen)
            # Remove chosen and its neighbors from active
            active.discard(chosen)
            for nb in local_conflict[chosen]:
                active.discard(nb)
        
        # Phase 2: Local search - try to improve by swapping
        # For each pair (in_selected, not_in_selected), check if swapping improves size
        improved = True
        while improved:
            improved = False
            selected_list = list(selected)
            non_selected = [v for v in range(n_rem) if v not in selected]
            
            for s in selected_list:
                for ns in non_selected:
                    # Try swapping: remove s, add ns
                    # Check if ns is independent from (selected - {s})
                    if ns in local_conflict[s]:
                        continue  # ns conflicts with s, can't add
                    # Check ns against all other selected vertices
                    conflict_with_selected = False
                    for s2 in selected:
                        if s2 != s and ns in local_conflict[s2]:
                            conflict_with_selected = True
                            break
                    if conflict_with_selected:
                        continue
                    
                    # Swapping is valid: new set has same size, but we want to grow
                    # Actually, swapping keeps size same. We need a different approach.
                    # Let's try: remove s, then see if we can add multiple non-selected vertices
                    pass
        
        # Phase 2 (revised): Try to add more vertices by removing one and adding multiple
        improved = True
        while improved:
            improved = False
            selected_list = list(selected)
            
            for s in selected_list:
                # Temporarily remove s
                temp_selected = selected - {s}
                # Find all non-selected vertices that are independent from temp_selected
                candidates = []
                for v in range(n_rem):
                    if v in temp_selected:
                        continue
                    # Check if v is independent from all of temp_selected
                    independent = True
                    for s2 in temp_selected:
                        if v in local_conflict[s2]:
                            independent = False
                            break
                    if independent:
                        candidates.append(v)
                
                # From candidates, find a maximal independent set among them
                # (they may conflict with each other)
                cand_selected = set()
                cand_active = set(candidates)
                while cand_active:
                    min_deg = float('inf')
                    chosen = None
                    for v in cand_active:
                        deg = len(local_conflict[v] & cand_active)
                        if deg < min_deg:
                            min_deg = deg
                            chosen = v
                    if chosen is None:
                        break
                    cand_selected.add(chosen)
                    cand_active.discard(chosen)
                    for nb in local_conflict[chosen]:
                        cand_active.discard(nb)
                
                # If we can add more than 1 vertex by removing s, do it
                if len(cand_selected) > 1:
                    selected = temp_selected | cand_selected
                    improved = True
                    break
        
        # Convert back to global indices
        return {rem_list[i] for i in selected}
    
    def greedy_coloring_improved(conflict, m):
        """
        Improved coloring: iteratively find a large induced matching (independent set
        in conflict graph), assign it a new color, remove those edges, repeat.
        """
        color = [-1] * m
        remaining = set(range(m))
        color_idx = 0
        
        while remaining:
            # Find a large independent set in the conflict graph restricted to remaining
            selected = find_max_independent_set(conflict, remaining)
            
            # Assign color to selected edges
            for global_idx in selected:
                color[global_idx] = color_idx
                remaining.discard(global_idx)
            
            color_idx += 1
        
        return color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        conflict = build_conflict_graph(edges, adj)
        
        # Greedy coloring with improved independent set finding
        color = greedy_coloring_improved(conflict, m)
        
        out.append(color)
    
    return out
# EVOLVE-BLOCK-END
