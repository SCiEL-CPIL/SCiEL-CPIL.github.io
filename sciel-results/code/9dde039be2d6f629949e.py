"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 145.0  # Safety margin from 150s limit


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycle_dfs(start, adj, remaining_edges, n, max_depth=None):
    """
    Find a simple cycle containing 'start' where all edges are in remaining_edges.
    Uses iterative DFS. Returns the cycle as a list of vertices, or None.
    """
    if max_depth is None:
        max_depth = n
    
    # Check if start has at least 2 neighbors in remaining graph
    neighbors_in_rem = []
    for v in adj[start]:
        edge = (min(start, v), max(start, v))
        if edge in remaining_edges:
            neighbors_in_rem.append(v)
    
    if len(neighbors_in_rem) < 2:
        return None
    
    # Iterative DFS
    path = [start]
    on_path = {start}
    stack = [(start, iter(sorted(adj[start])))]
    
    while stack:
        node, it = stack[-1]
        found = False
        for v in it:
            edge = (min(node, v), max(node, v))
            if edge not in remaining_edges:
                continue
            if v == start:
                if len(path) >= 3:
                    return path[:]
                continue
            if v not in on_path:
                on_path.add(v)
                path.append(v)
                if len(path) > max_depth:
                    path.pop()
                    on_path.discard(v)
                    break
                stack.append((v, iter(sorted(adj[v]))))
                found = True
                break
        if not found:
            stack.pop()
            if path:
                path.pop()
                on_path.discard(node)
    
    return None


def _find_cycle_bfs(start, adj, remaining_edges, n):
    """
    Find a short cycle containing 'start' using BFS.
    Returns the cycle as a list of vertices, or None.
    """
    neighbors_in_rem = []
    for v in adj[start]:
        edge = (min(start, v), max(start, v))
        if edge in remaining_edges:
            neighbors_in_rem.append(v)
    
    if len(neighbors_in_rem) < 2:
        return None
    
    # BFS from start to find shortest path back to start
    from collections import deque
    queue = deque()
    for v in neighbors_in_rem:
        queue.append((v, [start, v]))
    
    visited = {v: [start, v] for v in neighbors_in_rem}
    
    while queue:
        node, path = queue.popleft()
        for v in adj[node]:
            edge = (min(node, v), max(node, v))
            if edge not in remaining_edges:
                continue
            if v == start:
                if len(path) >= 3:
                    return path[:]
                continue
            if v not in visited:
                visited[v] = path + [v]
                queue.append((v, path + [v]))
    
    return None


def _find_longest_cycle(adj, remaining_edges, n, time_limit, start_time, prefer_long=True):
    """
    Try to find a long cycle in the graph.
    Returns the cycle or None.
    """
    best_cycle = None
    best_len = 0
    
    # Try vertices with highest degree first
    candidates = [v for v in range(n) if len(adj[v]) >= 2]
    if prefer_long:
        candidates.sort(key=lambda v: -len(adj[v]))
    else:
        candidates.sort(key=lambda v: len(adj[v]))
    
    for start in candidates:
        if time.time() - start_time > time_limit:
            break
        if len(adj[start]) < 2:
            continue
        
        cycle = _find_cycle_dfs(start, adj, remaining_edges, n)
        if cycle and len(cycle) > best_len:
            best_len = len(cycle)
            best_cycle = cycle
            if best_len >= n:  # Hamiltonian cycle found
                break
    
    return best_cycle


def _decompose_greedy_longest(edges, n, time_limit):
    """
    Decompose graph by repeatedly finding the longest cycle.
    """
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining_edges:
        if time.time() - start_time > time_limit:
            break
        
        cycle = _find_longest_cycle(adj, remaining_edges, n, 
                                     time_limit - (time.time() - start_time), 
                                     start_time, prefer_long=True)
        
        if cycle:
            # Verify all edges in cycle still exist
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    valid = False
                    break
            
            if valid:
                pieces.append(cycle)
                # Remove cycle edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                continue
        else:
            # No more cycles, add remaining edges as single-edge pieces
            for edge in remaining_edges:
                pieces.append(list(edge))
            remaining_edges = set()
            break
    
    # Add any leftover edges
    for edge in remaining_edges:
        pieces.append(list(edge))
    
    return pieces


def _decompose_greedy_shortest(edges, n, time_limit):
    """
    Decompose graph by repeatedly finding the shortest cycle.
    This can sometimes lead to more cycles overall.
    """
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining_edges:
        if time.time() - start_time > time_limit:
            break
        
        # Find shortest cycle
        best_cycle = None
        best_len = float('inf')
        
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        candidates.sort(key=lambda v: len(adj[v]))
        
        for start in candidates:
            if time.time() - start_time > time_limit:
                break
            if len(adj[start]) < 2:
                continue
            
            cycle = _find_cycle_bfs(start, adj, remaining_edges, n)
            if cycle and len(cycle) < best_len:
                best_len = len(cycle)
                best_cycle = cycle
                if best_len == 3:  # Triangle found
                    break
        
        if best_cycle:
            # Verify all edges in cycle still exist
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    valid = False
                    break
            
            if valid:
                pieces.append(best_cycle)
                # Remove cycle edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                continue
        else:
            # No more cycles, add remaining edges as single-edge pieces
            for edge in remaining_edges:
                pieces.append(list(edge))
            remaining_edges = set()
            break
    
    # Add any leftover edges
    for edge in remaining_edges:
        pieces.append(list(edge))
    
    return pieces


def _decompose_with_local_search(edges, n, time_limit):
    """
    Improved decomposition using greedy + local search.
    """
    if not edges:
        return []
    
    edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    # Phase 1: Greedy decomposition
    start_time = time.time()
    greedy_time = min(time_limit * 0.5, 30.0)
    pieces = _decompose_greedy_longest(edges, n, greedy_time)
    
    # Phase 2: Local search to improve
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        # Reconstruct adj and remaining_edges from pieces
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        used_edges = set()
        cycles = []
        single_edges = []
        
        for piece in pieces:
            if len(piece) >= 3:
                cycles.append(piece)
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    edge = (min(u, v), max(u, v))
                    used_edges.add(edge)
            else:
                single_edges.append(piece)
                used_edges.add((min(piece[0], piece[1]), max(piece[0], piece[1])))
        
        remaining_edges = edges_set - used_edges
        
        # Try to find more cycles
        improved_cycles = cycles[:]
        while time.time() - start_time < time_limit * 0.9:
            new_cycle = _find_longest_cycle(adj, remaining_edges, n, 
                                             (time_limit - (time.time() - start_time)) * 0.1, 
                                             start_time)
            if not new_cycle:
                break
            
            # Check if this cycle uses only remaining edges
            valid = True
            for i in range(len(new_cycle)):
                u = new_cycle[i]
                v = new_cycle[(i + 1) % len(new_cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    valid = False
                    break
            
            if valid:
                # Add this cycle
                improved_cycles.append(new_cycle)
                # Update remaining edges and adj
                for i in range(len(new_cycle)):
                    u = new_cycle[i]
                    v = new_cycle[(i + 1) % len(new_cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        # Reconstruct pieces
        final_pieces = improved_cycles[:]
        # Add back single edges that weren't part of any cycle
        cycle_edges = set()
        for cycle in improved_cycles:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                cycle_edges.add((min(u, v), max(u, v)))
        
        for edge in edges_set:
            if edge not in cycle_edges:
                final_pieces.append(list(edge))
        
        pieces = final_pieces
    
    return pieces


def _decompose_multi_strategy(edges, n, time_limit):
    """
    Try multiple strategies and pick the best.
    """
    if not edges:
        return []
    
    edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    best_pieces = None
    best_count = float('inf')
    
    start_time = time.time()
    
    # Strategy 1: Greedy longest cycle
    if time.time() - start_time < time_limit * 0.3:
        pieces1 = _decompose_greedy_longest(edges, n, time_limit * 0.25)
        if len(pieces1) < best_count:
            best_count = len(pieces1)
            best_pieces = pieces1
    
    # Strategy 2: Greedy shortest cycle
    if time.time() - start_time < time_limit * 0.5:
        pieces2 = _decompose_greedy_shortest(edges, n, time_limit * 0.2)
        if len(pieces2) < best_count:
            best_count = len(pieces2)
            best_pieces = pieces2
    
    # Strategy 3: Greedy with random restarts
    for trial in range(5):
        if time.time() - start_time > time_limit * 0.7:
            break
        
        # Shuffle vertex order
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges_set)
        local_start = time.time()
        
        while remaining_edges and time.time() - local_start < time_limit * 0.08:
            # Randomly pick a vertex with degree >= 2
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            
            start = random.choice(candidates)
            cycle = _find_cycle_dfs(start, adj, remaining_edges, n)
            
            if cycle:
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = (min(u, v), max(u, v))
                    if edge not in remaining_edges:
                        valid = False
                        break
                
                if valid:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = (min(u, v), max(u, v))
                        remaining_edges.discard(edge)
                        adj[u].discard(v)
                        adj[v].discard(u)
                else:
                    continue
            else:
                break
        
        for edge in remaining_edges:
            pieces.append(list(edge))
        
        if len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces
    
    # Strategy 4: Use local search on the best so far
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10 and best_pieces:
        # Extract cycles from best_pieces
        cycles = [p for p in best_pieces if len(p) >= 3]
        if cycles:
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            used_edges = set()
            for cycle in cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    used_edges.add((min(u, v), max(u, v)))
            
            remaining_edges = edges_set - used_edges
            
            improved = cycles[:]
            while time.time() - start_time < time_limit * 0.95:
                new_cycle = _find_longest_cycle(adj, remaining_edges, n, 
                                                 (time_limit - (time.time() - start_time)) * 0.1, 
                                                 start_time)
                if not new_cycle:
                    break
                
                # Check if this cycle uses only remaining edges
                valid = True
                for i in range(len(new_cycle)):
                    u = new_cycle[i]
                    v = new_cycle[(i + 1) % len(new_cycle)]
                    edge = (min(u, v), max(u, v))
                    if edge not in remaining_edges:
                        valid = False
                        break
                
                if valid:
                    # Add this cycle
                    improved.append(new_cycle)
                    # Update remaining edges and adj
                    for i in range(len(new_cycle)):
                        u = new_cycle[i]
                        v = new_cycle[(i + 1) % len(new_cycle)]
                        edge = (min(u, v), max(u, v))
                        remaining_edges.discard(edge)
                        adj[u].discard(v)
                        adj[v].discard(u)
            
            final_pieces = improved[:]
            cycle_edges = set()
            for cycle in improved:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add((min(u, v), max(u, v)))
            
            for edge in edges_set:
                if edge not in cycle_edges:
                    final_pieces.append(list(edge))
            
            if len(final_pieces) < best_count:
                best_count = len(final_pieces)
                best_pieces = final_pieces
    
    return best_pieces if best_pieces else [[u, v] for u, v in edges]


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            # Fallback: just output all edges as single-edge pieces
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        # Use multi-strategy decomposition
        graph_time = min(remaining_time * 0.9, 45.0)
        
        pieces = _decompose_multi_strategy(edges, n, graph_time)
        
        # Validate: ensure pieces are edge-disjoint and cover all edges
        used_edges = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                # Check for duplicate vertices in cycle
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
        
        if not valid or used_edges != edges_set:
            # Fallback to simple edge decomposition
            results.append([[u, v] for u, v in edges])
        else:
            results.append(pieces)

    return results
