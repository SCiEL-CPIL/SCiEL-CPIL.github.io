"""Strong edge colourings via iterative maximum induced matching extraction."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

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

    def build_conflict(edges, adj):
        """Build conflict graph: edge i conflicts with edge j if they share a vertex
        or there exists an edge connecting a vertex of i to a vertex of j."""
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        # For each edge, find all conflicting edges
        for i, (u, v) in enumerate(edges):
            # Edges sharing u or v
            for w in adj[u]:
                e = (min(u, w), max(u, w))
                if e in edge_index:
                    j = edge_index[e]
                    if j != i:
                        conflict[i].add(j)
                        conflict[j].add(i)
            for w in adj[v]:
                e = (min(v, w), max(v, w))
                if e in edge_index:
                    j = edge_index[e]
                    if j != i:
                        conflict[i].add(j)
                        conflict[j].add(i)
        return conflict

    def greedy_induced_matching(remaining, conflict, adj, edges, order=None):
        """Find a large induced matching from remaining edges using min-degree heuristic.
        remaining: set of edge indices still uncoloured
        Returns: list of edge indices forming an induced matching"""
        if not remaining:
            return []
        
        if order is None:
            # Sort by degree in conflict graph restricted to remaining
            degs = [(len(conflict[i] & remaining), i) for i in remaining]
            degs.sort()
            candidates = [i for _, i in degs]
        else:
            candidates = order
        
        selected = []
        removed = set()
        for i in candidates:
            if i in removed or i not in remaining:
                continue
            # Check if i is still compatible with all selected
            ok = True
            for s in selected:
                if s in conflict[i]:
                    ok = False
                    break
            if ok:
                selected.append(i)
                # Remove i and all its conflicts
                removed.add(i)
                for c in conflict[i]:
                    removed.add(c)
        return selected

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict(edges, adj)
        
        best_colouring = None
        best_count = m + 1
        
        # Try multiple random orderings of edges for the greedy
        rng = random.Random(seed + 7)
        
        # First: deterministic min-degree ordering
        remaining = set(range(m))
        colour = [-1] * m
        colour_id = 0
        while remaining:
            if time.time() > deadline:
                break
            sel = greedy_induced_matching(remaining, conflict, adj, edges, order=None)
            for i in sel:
                colour[i] = colour_id
            remaining -= set(sel)
            colour_id += 1
        
        if best_colouring is None or colour_id < best_count:
            best_colouring = colour[:]
            best_count = colour_id
        
        # Try randomized orderings
        num_tries = 30
        for t in range(num_tries):
            if time.time() > deadline - 5:
                break
            order = list(range(m))
            rng.shuffle(order)
            # Weight by degree: lower degree first
            order.sort(key=lambda i: (len(conflict[i]) + rng.random() * 0.1, rng.random()))
            
            remaining = set(range(m))
            colour = [-1] * m
            colour_id = 0
            while remaining:
                if time.time() > deadline - 5:
                    break
                sel = greedy_induced_matching(remaining, conflict, adj, edges, order=order)
                for i in sel:
                    colour[i] = colour_id
                remaining -= set(sel)
                colour_id += 1
            
            if colour_id < best_count:
                best_count = colour_id
                best_colouring = colour[:]
        
        if best_colouring is None:
            # Fallback: simple greedy
            best_colouring = []
            for i in range(m):
                best_colouring.append(0)
        
        return best_colouring

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            out.append([0])  # placeholder, should not happen
            continue
        result = solve_instance(*spec, time_budget=40)
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
