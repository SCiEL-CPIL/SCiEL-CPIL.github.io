"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges):
    """Decompose edges into cycles and single edges."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    def build_adj(eset):
        adj = {}
        for u, v in eset:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj
    
    adj = build_adj(edges_set)
    
    def find_cycle(adj, eset):
        """Find a simple cycle using DFS."""
        n = max(adj.keys()) + 1 if adj else 0
        for start in range(n):
            if start not in adj or not adj[start]:
                continue
            stack = [(start, [start])]
            visited_in_path = {start}
            while stack:
                node, path = stack.pop()
                for neighbor in sorted(adj.get(node, [])):
                    edge = tuple(sorted((node, neighbor)))
                    if edge not in eset:
                        continue
                    if neighbor == start and len(path) >= 3:
                        return path + [start]
                    if neighbor not in visited_in_path:
                        visited_in_path.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))
            # Reset for next start
        return None
    
    max_iter = len(edges_set) * 2
    for _ in range(max_iter):
        if not edges_set:
            break
        cycle = find_cycle(adj, edges_set)
        if cycle is None:
            break
        # Remove cycle edges
        for i in range(len(cycle) - 1):
            edge = tuple(sorted((cycle[i], cycle[i+1])))
            edges_set.discard(edge)
            if edge[0] in adj and edge[1] in adj[edge[0]]:
                adj[edge[0]].discard(edge[1])
            if edge[1] in adj and edge[0] in adj[edge[1]]:
                adj[edge[1]].discard(edge[0])
            for v in (edge[0], edge[1]):
                if v in adj and not adj[v]:
                    del adj[v]
        pieces.append(cycle)
    
    # Remaining edges as single pieces
    for e in sorted(edges_set):
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
