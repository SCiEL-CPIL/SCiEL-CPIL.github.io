import time
import random
from collections import deque, defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        n, p, seed = spec
        edges = generate_graph(n, p, seed)
        
        best_decomp = None
        best_count = float('inf')
        
        if n <= 20:
            max_restarts = 300
        elif n <= 25:
            max_restarts = 150
        else:
            max_restarts = 80

        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(r * 1000 + seed)
            decomp = decompose_with_local_search(edges, n, rng, deadline)
            count = len(decomp)
            
            if count < best_count:
                best_count = count
                best_decomp = decomp
                if count == 1:
                    break

        if best_decomp is None:
            best_decomp = [[u, v] for u, v in edges]
        
        results.append(best_decomp)

    return results

def generate_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i+1]
        edges.add((min(u,v), max(u,v)))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges

def get_connected_components(edges, n):
    """Find connected components in the graph."""
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    visited = set()
    components = []
    
    for node in range(n):
        if node not in visited and adj[node]:
            component = set()
            queue = deque([node])
            visited.add(node)
            while queue:
                u = queue.popleft()
                component.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            components.append(component)
    
    return components

def get_component_edges(edges, component):
    """Get edges that have both endpoints in the component."""
    comp_set = set(component)
    return set(e for e in edges if e[0] in comp_set and e[1] in comp_set)

def get_2_core(edges, n):
    """Returns the set of edges in the 2-core of the graph."""
    if not edges:
        return set()
    
    degree = {i: 0 for i in range(n)}
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    q = deque([i for i in range(n) if degree[i] < 2])
    removed = set()
    
    while q:
        u = q.popleft()
        if u in removed:
            continue
        if degree[u] >= 2:
            continue
            
        removed.add(u)
        for v in list(adj[u]):
            if v in removed:
                continue
            degree[v] -= 1
            adj[v].discard(u)
            adj[u].discard(v)
            if degree[v] < 2:
                q.append(v)
                
    core_edges = set()
    for u, v in edges:
        if u not in removed and v not in removed:
            core_edges.add((u, v))
            
    return core_edges

def find_long_cycle_dfs(edges, n, rng, max_steps=8000):
    """Find a long cycle using DFS with randomization."""
    if not edges:
        return None
        
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    active_nodes = [i for i in range(n) if adj[i]]
    if not active_nodes:
        return None
        
    for node in active_nodes:
        adj[node] = list(adj[node])
        rng.shuffle(adj[node])
        
    best_cycle = None
    best_len = 0
    steps_used = 0
    
    for start in active_nodes:
        if steps_used >= max_steps:
            break
            
        def dfs(u, path, visited):
            nonlocal steps_used, best_cycle, best_len
            
            if steps_used >= max_steps:
                return False
            
            steps_used += 1
            
            if len(path) > 2 and start in adj[u]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)
            
            if len(path) + (n - len(visited)) <= best_len:
                return False
            
            for v in adj[u]:
                if v == start:
                    if len(path) > 2:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                    continue
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    if not dfs(v, path, visited):
                        path.pop()
                        visited.remove(v)
                        return False
                    path.pop()
                    visited.remove(v)
                    
            return True

        dfs(start, [start], {start})
        
        if best_len == n:
            break
    
    return best_cycle

def decompose_component(edges, n, rng, deadline):
    """Decompose a single component using greedy cycle extraction."""
    remaining = set(edges)
    pieces = []
    
    while remaining:
        if time.time() > deadline:
            break
            
        core_edges = get_2_core(remaining, n)
        
        if not core_edges:
            for e in remaining:
                pieces.append([e[0], e[1]])
            break
        
        cycle = find_long_cycle_dfs(core_edges, n, rng, max_steps=5000)
        
        if cycle:
            pieces.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.remove(e)
        else:
            e = next(iter(remaining))
            pieces.append([e[0], e[1]])
            remaining.remove(e)
            
    return pieces

def local_search_merge(pieces, edges_set, n, rng, deadline, max_iterations=200):
    """
    Local search: try to merge single-edge pieces into cycles.
    A piece is a cycle if len > 2, or a single edge if len == 2.
    We try to find sets of single-edge pieces that form a cycle.
    """
    pieces = [list(p) for p in pieces]
    
    # Build edge -> piece_index mapping
    edge_to_piece = {}
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            e = (min(u, v), max(u, v))
            edge_to_piece[e] = i
    
    # Build adjacency from single-edge pieces only
    single_edge_indices = [i for i, p in enumerate(pieces) if len(p) == 2]
    
    if not single_edge_indices:
        return pieces
    
    # Build graph of single edges
    adj = defaultdict(set)
    for i in single_edge_indices:
        u, v = pieces[i]
        adj[u].add(v)
        adj[v].add(u)
    
    # Try to find cycles among single-edge pieces
    merged = set()
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Find a cycle in the single-edge subgraph
        cycle = find_cycle_in_subgraph(adj, n, rng, max_steps=3000)
        
        if not cycle or len(cycle) < 3:
            break
        
        # Check if all edges in this cycle are single-edge pieces
        cycle_edges = []
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in edge_to_piece:
                idx = edge_to_piece[e]
                if idx in merged:
                    valid = False
                    break
                cycle_edges.append((e, idx))
            else:
                valid = False
                break
        
        if not valid:
            # Try removing this cycle from consideration for now
            # Just break to avoid infinite loop
            break
        
        # Merge these single-edge pieces into one cycle piece
        new_piece = list(cycle)
        
        # Remove the old single-edge pieces
        indices_to_remove = sorted([idx for _, idx in cycle_edges], reverse=True)
        for idx in indices_to_remove:
            del pieces[idx]
            # Rebuild edge_to_piece after deletion is complex; 
            # instead, mark as merged and skip
            for e in cycle_edges:
                if e[1] == idx:
                    edge_to_piece.pop(e[0], None)
        
        # Add the new cycle piece
        pieces.append(new_piece)
        merged.update(indices_to_remove)
        
        # Rebuild edge_to_piece for remaining single edges
        edge_to_piece = {}
        for i, piece in enumerate(pieces):
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                if i not in merged:
                    edge_to_piece[e] = i
        
        # Rebuild adj
        adj = defaultdict(set)
        for i, piece in enumerate(pieces):
            if len(piece) == 2 and i not in merged:
                u, v = piece
                adj[u].add(v)
                adj[v].add(u)
    
    return pieces

def find_cycle_in_subgraph(adj, n, rng, max_steps=3000):
    """Find a cycle in a subgraph given by adjacency."""
    active_nodes = [i for i in range(n) if adj[i]]
    if not active_nodes:
        return None
    
    for node in active_nodes:
        adj[node] = list(adj[node])
        rng.shuffle(adj[node])
    
    best_cycle = None
    best_len = 0
    steps = 0
    
    for start in active_nodes:
        if steps >= max_steps:
            break
        
        def dfs(u, path, visited):
            nonlocal steps, best_cycle, best_len
            if steps >= max_steps:
                return False
            steps += 1
            
            if len(path) > 2 and start in adj[u]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)
            
            if len(path) + (n - len(visited)) <= best_len:
                return False
            
            for v in adj[u]:
                if v == start:
                    if len(path) > 2:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                    continue
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    if not dfs(v, path, visited):
                        path.pop()
                        visited.remove(v)
                        return False
                    path.pop()
                    visited.remove(v)
            return True
        
        dfs(start, [start], {start})
        if best_len == n:
            break
    
    return best_cycle

def decompose_with_local_search(edges, n, rng, deadline):
    """Decompose using component-wise greedy + local search merge."""
    # Get connected components
    components = get_connected_components(edges, n)
    
    all_pieces = []
    
    for comp in components:
        comp_edges = get_component_edges(edges, comp)
        if not comp_edges:
            continue
        
        # Decompose this component
        pieces = decompose_component(comp_edges, n, rng, deadline)
        all_pieces.extend(pieces)
    
    # Local search to merge single-edge pieces into cycles
    edges_set = set(edges)
    all_pieces = local_search_merge(all_pieces, edges_set, n, rng, deadline)
    
    return all_pieces
