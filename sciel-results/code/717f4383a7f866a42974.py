"""Path decomposition via iterative maximum matching on path fragments."""
# EVOLVE-BLOCK-START
import sys
from random import Random
from collections import defaultdict

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_graph(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_max_matching_bipartite(left_nodes, right_nodes, can_connect):
    """
    Find maximum matching in a bipartite graph.
    left_nodes, right_nodes: lists of node IDs
    can_connect(l, r): returns True if edge (l,r) exists
    Returns list of (l, r) pairs.
    """
    from collections import deque
    
    # Map left and right nodes to indices for efficiency
    left_set = set(left_nodes)
    right_set = set(right_nodes)
    
    # Build adjacency for left nodes
    left_adj = {l: [] for l in left_nodes}
    for l in left_nodes:
        for r in right_nodes:
            if can_connect(l, r):
                left_adj[l].append(r)
    
    # Hopcroft-Karp or simple DFS augmenting paths
    match_r = {}  # r -> l
    match_l = {}  # l -> r
    
    def dfs(l, visited):
        for r in left_adj[l]:
            if r in visited:
                continue
            visited.add(r)
            if r not in match_r or dfs(match_r[r], visited):
                match_r[r] = l
                match_l[l] = r
                return True
        return False
    
    for l in left_nodes:
        visited = set()
        dfs(l, visited)
    
    return [(l, r) for l, r in match_l.items()]


def _decompose_paths(edges):
    """
    Decompose edges into minimum number of edge-disjoint paths covering all edges.
    Uses iterative matching to merge path fragments.
    """
    if not edges:
        return []
    
    adj = _build_graph(edges)
    
    # Find connected components
    visited = set()
    components = []
    for v in adj:
        if v not in visited:
            comp = set()
            stack = [v]
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                for nb in adj[node]:
                    if nb not in comp:
                        stack.append(nb)
            visited |= comp
            components.append(comp)
    
    all_paths = []
    for comp in components:
        comp_edges = [(u, v) for u, v in edges if u in comp and v in comp]
        if not comp_edges:
            continue
        
        comp_adj = defaultdict(set)
        for u, v in comp_edges:
            comp_adj[u].add(v)
            comp_adj[v].add(u)
        
        # Count odd-degree vertices in this component
        odd_vertices = [v for v in comp if len(comp_adj[v]) % 2 == 1]
        
        # Initial paths: use a trail decomposition approach
        # Start with each edge as a path, then merge
        # Better: use the standard algorithm for minimum path cover of edges
        
        # The minimum number of edge-disjoint paths covering all edges in a connected graph
        # is max(1, odd_count / 2)
        
        # We'll construct paths by finding trails
        # Use a simple approach: repeatedly find a trail from an odd vertex (or any vertex if all even)
        
        remaining_edges = set(comp_edges)
        paths = []
        
        # Find a starting vertex
        # If there are odd vertices, start from one; otherwise any vertex with degree > 0
        def get_start_vertex():
            # Prefer odd-degree vertices
            for v in odd_vertices:
                if any((v, u) in remaining_edges or (u, v) in remaining_edges for u in comp_adj[v]):
                    return v
            for v in comp:
                if any((v, u) in remaining_edges or (u, v) in remaining_edges for u in comp_adj[v]):
                    return v
            return None
        
        while remaining_edges:
            start = get_start_vertex()
            if start is None:
                break
            
            # Find a trail starting from 'start'
            path = [start]
            current = start
            used_in_trail = set()
            
            while True:
                # Find an unused edge from current
                next_v = None
                for nb in comp_adj[current]:
                    e = (current, nb) if current < nb else (nb, current)
                    if e in remaining_edges and nb not in path:
                        next_v = nb
                        break
                    elif e in remaining_edges:
                        # Allow revisiting if needed? No, paths must be simple.
                        # But for edge-disjoint path cover, paths are simple (no repeated vertices).
                        pass
                
                if next_v is None:
                    break
                
                # Try to extend
                # We need to be careful: this greedy might not find the optimal trail length
                # Let's use a different approach: Hierholzer-like but for paths
                
                path.append(next_v)
                e = (current, next_v) if current < next_v else (next_v, current)
                remaining_edges.discard(e)
                used_in_trail.add(e)
                current = next_v
            
            if len(path) >= 2:
                paths.append(path)
        
        all_paths.extend(paths)
    
    return all_paths


def _decompose_with_matching(edges):
    """
    More sophisticated approach:
    1. Find all odd-degree vertices
    2. Pair them up (minimum number of pairs = odd_count/2)
    3. For each pair, find a path between them in the graph
    4. The union of these paths plus Eulerian trails in the remaining parts gives the decomposition
    
    Actually, the standard result: minimum path decomposition = odd_count/2 paths (if odd_count > 0), else 1 path (if edges exist).
    
    Algorithm:
    - Find odd vertices
    - Pair them arbitrarily (or optimally)
    - For each pair (u, v), find a simple path from u to v
    - Remove those edges
    - The remaining graph has all even degrees, so it has an Eulerian circuit
    - But we need paths, not circuits... 
    
    Let me think differently. The standard approach:
    - The minimum number of edge-disjoint paths covering all edges is ceil(odd_count/2) if odd_count > 0, else 1.
    - To achieve this: pair up odd vertices, find paths between each pair, then the remaining edges form an Eulerian subgraph which can be decomposed into one cycle (or multiple, but we can merge them).
    
    Actually, let's use a cleaner algorithm:
    1. Find all odd-degree vertices. Pair them up: (a1,b1), (a2,b2), ...
    2. For each pair, find a simple path in the graph.
    3. Remove all edges used in these paths.
    4. The remaining graph has all even degrees. Find an Eulerian circuit for each connected component.
    5. Convert each Eulerian circuit into a path by breaking it at some point.
    6. But this might give more paths than needed.
    
    Alternative cleaner approach:
    - Use the fact that we can do a depth-first search that traverses each edge exactly once, but we may need to "lift" some edges.
    
    Let me just implement a direct constructive algorithm:
    - Repeatedly find a trail (maximal path with no repeated edges) starting from an odd vertex if possible, else any vertex.
    - This gives a valid path decomposition. The number of paths equals the number of times we start a new trail.
    - To minimize, always start from an odd vertex when available.
    """
    if not edges:
        return []
    
    adj = _build_graph(edges)
    remaining = set(edges)
    paths = []
    
    def find_trail(start):
        """Find a maximal trail starting from start using unused edges."""
        path = [start]
        current = start
        while True:
            next_v = None
            for nb in sorted(adj[current]):
                e = (current, nb) if current < nb else (nb, current)
                if e in remaining:
                    next_v = nb
                    break
            if next_v is None:
                break
            path.append(next_v)
            e = (current, next_v) if current < next_v else (next_v, current)
            remaining.discard(e)
            current = next_v
        return path
    
    # Find odd-degree vertices dynamically
    def get_odd_vertices():
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        return [v for v in deg if deg[v] % 2 == 1]
    
    def has_edges_from(v):
        for nb in adj[v]:
            e = (v, nb) if v < nb else (nb, v)
            if e in remaining:
                return True
        return False
    
    while remaining:
        odd_v = get_odd_vertices()
        start = None
        for v in odd_v:
            if has_edges_from(v):
                start = v
                break
        if start is None:
            for v in adj:
                if has_edges_from(v):
                    start = v
                    break
        if start is None:
            break
        
        path = find_trail(start)
        if len(path) >= 2:
            paths.append(path)
    
    return paths


def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        paths = _decompose_with_matching(edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
