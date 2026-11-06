"""Decompose random graphs into cycles and edges, minimizing piece count."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges_list, time_budget):
    """Decompose edges into cycles (longer preferred) and single edges."""
    if not edges_list:
        return []
    
    # Build adjacency
    adj = {}
    edge_set = set()
    for u, v in edges_list:
        edge_set.add((u, v))
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
    
    # Remove vertices with no edges from consideration
    active_edges = set(edge_set)
    pieces = []
    
    # Greedy: repeatedly find the longest cycle in the remaining graph
    # Use DFS-based cycle finding, trying to maximize cycle length
    deadline = time.time() + time_budget
    
    while active_edges:
        if time.time() > deadline:
            # Fallback: remaining edges as single edges
            for e in active_edges:
                pieces.append([e[0], e[1]])
            break
        
        # Find a vertex with degree >= 2 to start cycle search
        start = None
        for e in list(active_edges):
            u, v = e
            if len(adj.get(u, [])) >= 2 or len(adj.get(v, [])) >= 2:
                start = u
                break
        
        if start is None:
            # All remaining edges are isolated (no vertex has degree >= 2)
            for e in active_edges:
                pieces.append([e[0], e[1]])
            break
        
        # Try to find a long cycle starting from 'start' using DFS
        # We'll do multiple DFS attempts with different neighbor orderings
        best_cycle = None
        best_cycle_edges = None
        
        # Sort neighbors to have deterministic but varied exploration
        def find_cycle(start_vertex, max_depth=50):
            """DFS to find a cycle back to start_vertex."""
            # State: (current, path, visited_edges, visited_vertices_in_path)
            # Use iterative DFS with backtracking
            stack = [(start_vertex, [start_vertex], set(), {start_vertex})]
            found = False
            
            # Limit search for efficiency
            search_count = 0
            max_search = 20000
            
            while stack and not found:
                if search_count > max_search:
                    break
                search_count += 1
                
                curr, path, used_edges, path_verts = stack.pop()
                
                if len(path) >= 3:
                    # Check if we can close the cycle
                    if (min(curr, start_vertex), max(curr, start_vertex)) in active_edges and \
                       (min(curr, start_vertex), max(curr, start_vertex)) not in used_edges:
                        cycle = list(path)
                        cycle_edges = set()
                        for i in range(len(cycle)):
                            a, b = cycle[i], cycle[(i+1) % len(cycle)]
                            cycle_edges.add((min(a,b), max(a,b)))
                        return cycle, cycle_edges
                
                # Explore neighbors
                nbrs = list(adj.get(curr, []))
                # Try neighbors in order of higher degree first (heuristic for longer cycles)
                nbrs.sort(key=lambda x: -len(adj.get(x, [])))
                
                for nb in nbrs:
                    edge_key = (min(curr, nb), max(curr, nb))
                    if edge_key not in active_edges or edge_key in used_edges:
                        continue
                    if nb == start_vertex and len(path) < 3:
                        continue
                    if nb in path_verts and nb != start_vertex:
                        continue
                    
                    new_path = path + [nb]
                    new_used = used_edges | {edge_key}
                    new_verts = path_verts | {nb}
                    
                    # Limit path length
                    if len(new_path) <= max_depth:
                        stack.append((nb, new_path, new_used, new_verts))
            
            return None, None
        
        # Try to find a cycle
        cycle, cycle_edges = find_cycle(start)
        
        if cycle and cycle_edges:
            # Use this cycle
            pieces.append(cycle)
            active_edges -= cycle_edges
            # Update adjacency
            for e in cycle_edges:
                u, v = e
                if u in adj and v in adj[u]:
                    adj[u].remove(v)
                if v in adj and u in adj[v]:
                    adj[v].remove(u)
        else:
            # No cycle found from this start; try other vertices
            found_cycle = False
            for e in list(active_edges):
                u, v = e
                if len(adj.get(u, [])) >= 2:
                    cycle, cycle_edges = find_cycle(u)
                    if cycle and cycle_edges:
                        pieces.append(cycle)
                        active_edges -= cycle_edges
                        for ce in cycle_edges:
                            a, b = ce
                            if a in adj and b in adj[a]:
                                adj[a].remove(b)
                            if b in adj and a in adj[b]:
                                adj[b].remove(a)
                        found_cycle = True
                        break
                if len(adj.get(v, [])) >= 2:
                    cycle, cycle_edges = find_cycle(v)
                    if cycle and cycle_edges:
                        pieces.append(cycle)
                        active_edges -= cycle_edges
                        for ce in cycle_edges:
                            a, b = ce
                            if a in adj and b in adj[a]:
                                adj[a].remove(b)
                            if b in adj and a in adj[b]:
                                adj[b].remove(a)
                        found_cycle = True
                        break
            
            if not found_cycle:
                # No cycles left; remaining edges as single pieces
                for e in active_edges:
                    pieces.append([e[0], e[1]])
                break
    
    return pieces

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    total_budget = 140  # Leave 10s margin for 150s limit
    
    results = []
    per_graph_budget = total_budget / len(specs)
    
    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 5:
            # Not enough time; just return edges
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, min(per_graph_budget, remaining - 2))
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
