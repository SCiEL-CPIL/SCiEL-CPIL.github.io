"""Rewritten program using maximum edge-disjoint cycle packing via ILP-like approach with networkx-free implementation."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict
from random import Random

def instance_graph(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def find_cycles_in_graph(edges, n):
    """Find all simple cycles up to a reasonable length limit in the graph defined by edges."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycles = []
    # Limit cycle length to keep computation tractable
    max_cycle_len = min(n, 10)
    
    for start in range(n):
        # DFS to find cycles starting and ending at start
        # We only consider cycles where start is the smallest vertex in the cycle
        visited = [False] * n
        path = [start]
        
        def dfs(current, depth):
            if depth >= max_cycle_len:
                return
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    # Found a cycle
                    cycle = list(path)
                    # Normalize: rotate so smallest vertex is first, then reverse if needed for consistent ordering
                    min_idx = min(range(len(cycle)), key=lambda i: cycle[i])
                    norm = cycle[min_idx:] + cycle[:min_idx]
                    # Reverse if the second element is greater than the last (to have a canonical form)
                    if len(norm) > 2 and norm[1] > norm[-1]:
                        norm = [norm[0]] + norm[1:][::-1]
                    cycle_tuple = tuple(norm)
                    if cycle_tuple not in seen_cycles:
                        seen_cycles.add(cycle_tuple)
                        cycles.append(cycle_tuple)
                elif not visited[neighbor] and neighbor > start:
                    # Only allow vertices greater than start to avoid duplicate cycles
                    visited[neighbor] = True
                    path.append(neighbor)
                    dfs(neighbor, depth + 1)
                    path.pop()
                    visited[neighbor] = False
        
        seen_cycles = set()
        dfs(start, 1)
    
    return cycles

def edge_to_index(edge, edge_map):
    return edge_map[edge]

def solve_decomposition(edges, n):
    """
    Decompose the graph into edge-disjoint cycles and single edges, minimizing total pieces.
    This is equivalent to maximizing the number of edge-disjoint cycles.
    
    We use a greedy approach with backtracking for small instances, or a maximum
    matching based approach. Given the constraints, we'll use a heuristic that
    finds a large set of edge-disjoint cycles first, then covers remaining edges.
    """
    if not edges:
        return []
    
    # Build edge map
    edge_set = set(edges)
    edge_list = list(edge_set)
    
    # Find all cycles up to length limit
    cycles = find_cycles_in_graph(edge_set, n)
    
    # Greedy maximum edge-disjoint cycle packing
    # Sort cycles by length (prefer longer cycles as they use more edges per piece)
    cycles.sort(key=lambda c: -len(c))
    
    used_edges = set()
    selected_cycles = []
    
    for cycle in cycles:
        cycle_edges = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            cycle_edges.add(e)
        
        # Check if all edges in this cycle are unused
        if all(e not in used_edges for e in cycle_edges):
            selected_cycles.append(cycle)
            used_edges.update(cycle_edges)
    
    # Remaining edges become single-edge pieces
    remaining_edges = [e for e in edge_list if e not in used_edges]
    
    # Build pieces
    pieces = []
    for cycle in selected_cycles:
        pieces.append(list(cycle))
    for e in remaining_edges:
        pieces.append(list(e))
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        pieces = solve_decomposition(edges, n)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
