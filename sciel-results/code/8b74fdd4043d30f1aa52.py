import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def build_adj(edges):
        adj = {}
        for e in edges:
            u, v = e
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj
    
    def find_longest_cycle(adj, n, rng, max_depth=200, max_restarts=50):
        """Find a long cycle using randomized DFS with restarts."""
        best_cycle = None
        best_len = 0
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            # Pick a random start vertex with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            start = rng.choice(candidates)
            
            # DFS to find a cycle back to start
            stack = [(start, [start])]
            visited_in_path = {start}
            found = None
            
            while stack:
                node, path = stack.pop()
                if len(path) > max_depth:
                    continue
                neighbors = sorted(adj[node], key=lambda x: (len(adj[x]), x))
                # Randomize neighbor order
                rng.shuffle(neighbors)
                
                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = path
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle[:]
                        found = True
                        break
                    elif nb not in visited_in_path:
                        visited_in_path.add(nb)
                        stack.append((nb, path + [nb]))
                        if found:
                            break
                if found:
                    break
            visited_in_path.clear()
            
            if found:
                # Continue searching for longer cycles
                continue
        
        return best_cycle
    
    def find_cycle_greedy(adj, n, rng):
        """Find any cycle using a simple approach - useful as fallback."""
        # Find a cycle by following edges
        for start in range(n):
            if start not in adj or len(adj[start]) < 2:
                continue
            # Try to find a cycle starting from start
            stack = [(start, [start])]
            visited = {start}
            while stack:
                node, path = stack.pop()
                for nb in sorted(adj[node]):
                    if nb == start and len(path) >= 3:
                        return path
                    elif nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
            visited.clear()
        return None
    
    def decompose_graph(edges, n, seed_offset):
        """Decompose a graph into cycles and edges, minimizing piece count."""
        rng = Random(42 + seed_offset)
        remaining = set(edges)
        pieces = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            adj = build_adj(remaining)
            
            # Try to find the longest cycle
            cycle = find_longest_cycle(adj, n, rng, max_depth=100, max_restarts=30)
            
            if cycle is None:
                # No cycle found, use greedy
                cycle = find_cycle_greedy(adj, n, rng)
            
            if cycle is not None and len(cycle) >= 3:
                # Extract this cycle as a piece
                pieces.append(cycle)
                # Remove cycle edges from remaining
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining.discard(edge)
            else:
                # No more cycles; remaining edges are single-edge pieces
                for e in sorted(remaining):
                    pieces.append([e[0], e[1]])
                remaining.clear()
        
        return pieces
    
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for i, (n, p, seed) in enumerate(specs):
        if time.time() > deadline:
            # Return what we have, pad with empty
            while len(results) < 3:
                results.append([])
            break
        
        edges = instance(n, p, seed)
        pieces = decompose_graph(edges, n, i)
        results.append(pieces)
    
    # Ensure we return exactly 3 lists
    while len(results) < 3:
        results.append([])
    
    return results
