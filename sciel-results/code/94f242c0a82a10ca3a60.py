"""Rewritten program: improved cycle extraction with better heuristics and local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_graph(edges, n, time_budget=45):
        """Decompose graph into minimum pieces (cycles + edges) using iterative improvement."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        start_time = time.time()
        
        # Multiple randomized restarts with different strategies
        max_restarts = 300
        rng = random.Random(12345)
        
        for restart in range(max_restarts):
            if time.time() > start_time + time_budget - 2:
                break
            
            # Build adjacency
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining = set(edge_set)
            pieces = []
            
            # Iteratively extract cycles using multiple strategies
            while remaining:
                if time.time() > start_time + time_budget - 1:
                    break
                
                cycle = None
                
                # Strategy 1: Find longest cycle via randomized DFS from high-degree vertices
                degs = {}
                for u in list(adj.keys()):
                    degs[u] = len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining])
                
                if degs:
                    verts = sorted(degs.keys(), key=lambda u: (-degs[u], rng.random()))
                    
                    # Try top vertices
                    for start in verts[:min(12, len(verts))]:
                        if degs[start] < 2:
                            continue
                        
                        # Multiple DFS attempts from this start
                        for _ in range(15):
                            c = _find_cycle_randomized(start, adj, remaining, rng)
                            if c and len(c) >= 3:
                                if cycle is None or len(c) > len(cycle):
                                    cycle = c
                                if len(c) >= 10:
                                    break
                        
                        if cycle and len(cycle) >= 10:
                            break
                
                # Strategy 2: Greedy cycle extension
                c2 = _find_cycle_greedy_extended(adj, remaining, rng)
                if c2 and len(c2) >= 3:
                    if cycle is None or len(c2) > len(cycle):
                        cycle = c2
                
                # Strategy 3: Find any cycle (fallback)
                if not cycle:
                    c3 = _find_any_cycle(adj, remaining, rng)
                    if c3 and len(c3) >= 3:
                        cycle = c3
                
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    # No cycle found; take a single edge
                    e = tuple(sorted(rng.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))
            
            # Local search: try to merge adjacent single edges into cycles
            pieces = _local_search_merge(pieces, edge_set, adj, rng, time_budget, start_time)
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
                
                if best_count <= 5:  # Very good result
                    break
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
        
        return best_pieces
    
    def _find_cycle_randomized(start, adj, remaining, rng):
        """Find a cycle through start using randomized DFS with backtracking."""
        best_cycle = None
        
        # Try multiple depth limits
        for depth_limit in [6, 10, 14, 18, 24, 32, 45]:
            if best_cycle and len(best_cycle) >= depth_limit:
                break
            
            found = _dfs_cycle(start, adj, remaining, rng, depth_limit)
            if found and len(found) >= 3:
                if best_cycle is None or len(found) > len(best_cycle):
                    best_cycle = found
        
        return best_cycle
    
    def _dfs_cycle(start, adj, remaining, rng, depth_limit):
        """DFS to find a cycle back to start within depth_limit."""
        path = [start]
        visited = {start}
        
        def dfs(current, depth):
            if depth > depth_limit:
                return None
            
            neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
            
            if not neighbors:
                return None
            
            # Try to close cycle
            if start in neighbors and len(path) >= 3:
                return path[:]
            
            # Randomize neighbor order
            unvisited = [nb for nb in neighbors if nb not in visited]
            if not unvisited:
                return None
            
            rng.shuffle(unvisited)
            
            for nb in unvisited:
                visited.add(nb)
                path.append(nb)
                
                result = dfs(nb, depth + 1)
                if result:
                    return result
                
                path.pop()
                visited.discard(nb)
            
            return None
        
        return dfs(start, 0)
    
    def _find_cycle_greedy_extended(adj, remaining, rng):
        """Find a long cycle using greedy extension with look-ahead."""
        degs = {}
        for u in list(adj.keys()):
            degs[u] = len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining])
        
        if not degs:
            return None
        
        start_candidates = sorted(degs.keys(), key=lambda u: (-degs[u], rng.random()))
        best_cycle = None
        
        for start in start_candidates[:min(8, len(start_candidates))]:
            if degs[start] < 2:
                continue
            
            for _ in range(10):
                path = [start]
                visited = {start}
                current = start
                found = False
                
                for _ in range(80):
                    neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                    if not neighbors:
                        break
                    
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    
                    if unvisited:
                        # Prefer neighbors with higher remaining degree
                        unvisited_sorted = sorted(unvisited, key=lambda nb: (-degs.get(nb, 0), rng.random()))
                        next_node = unvisited_sorted[0]
                    elif start in neighbors and len(path) >= 3:
                        next_node = start
                        found = True
                    else:
                        break
                    
                    if found:
                        if len(path) >= 3:
                            if best_cycle is None or len(path) > len(best_cycle):
                                best_cycle = path[:]
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if found and len(path) >= 3:
                    if best_cycle is None or len(path) > len(best_cycle):
                        best_cycle = path[:]
        
        return best_cycle
    
    def _find_any_cycle(adj, remaining, rng):
        """Find any cycle in the graph (fallback)."""
        # Try from each vertex
        for u in list(adj.keys()):
            if len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining]) < 2:
                continue
            
            # BFS to find a cycle
            visited = {u}
            queue = [(u, [u])]
            
            while queue:
                current, path = queue.pop(0)
                neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                
                for nb in neighbors:
                    if nb == u and len(path) >= 3:
                        return path[:]
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
        
        return None
    
    def _local_search_merge(pieces, edge_set, adj, rng, time_budget, start_time):
        """Try to merge single edges into cycles or extend existing cycles."""
        # Identify single edges and cycles
        single_edges = []
        cycles = []
        for piece in pieces:
            if len(piece) == 2:
                single_edges.append(tuple(sorted(piece)))
            else:
                cycles.append(piece)
        
        if not single_edges:
            return pieces
        
        # Try to form cycles from single edges
        # Build a graph of single edges
        se_adj = defaultdict(set)
        for u, v in single_edges:
            se_adj[u].add(v)
            se_adj[v].add(u)
        
        # Find cycles in the single-edge graph
        merged = False
        for _ in range(50):
            if time.time() > start_time + time_budget - 1:
                break
            
            # Try to find a cycle in single_edges
            for u in list(se_adj.keys()):
                if len(se_adj[u]) < 2:
                    continue
                
                # DFS to find cycle
                path = [u]
                visited = {u}
                stack = [(u, 0)]
                found_cycle = None
                
                while stack:
                    node, depth = stack.pop()
                    if depth > 10:
                        continue
                    
                    neighbors = [nb for nb in se_adj[node] if (min(node, nb), max(node, nb)) in single_edges]
                    if not neighbors:
                        continue
                    
                    if u in neighbors and len(path) >= 3:
                        found_cycle = path[:]
                        break
                    
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    rng.shuffle(unvisited)
                    
                    for nb in unvisited[:3]:
                        visited.add(nb)
                        path.append(nb)
                        stack.append((nb, depth + 1))
                
                if found_cycle and len(found_cycle) >= 3:
                    # Verify all edges exist in original graph
                    valid = True
                    for i in range(len(found_cycle)):
                        a, b = found_cycle[i], found_cycle[(i+1) % len(found_cycle)]
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            valid = False
                            break
                        if e not in single_edges:
                            valid = False
                            break
                    
                    if valid:
                        # Remove these edges from single_edges
                        for i in range(len(found_cycle)):
                            a, b = found_cycle[i], found_cycle[(i+1) % len(found_cycle)]
                            single_edges.remove(tuple(sorted((a, b))))
                        se_adj.clear()
                        for u, v in single_edges:
                            se_adj[u].add(v)
                            se_adj[v].add(u)
                        
                        # Replace the single edges with a cycle
                        pieces = [p for p in pieces if len(p) != 2]
                        pieces.append(found_cycle)
                        merged = True
                        break
            
            if not merged:
                break
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Allocate time budget per graph
        remaining_time = deadline - time.time()
        if remaining_time < 15:
            # Not enough time; use simple decomposition
            pieces = [list(e) for e in edges]
        else:
            # Allocate roughly 1/3 of remaining time to each graph
            time_budget = min(remaining_time / 3 - 2, 45)
            pieces = decompose_graph(edges, n, time_budget=max(5, time_budget))
        
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
