import time
from collections import defaultdict
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before the 150 s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def try_find_hamiltonian_cycle(adj, rng, visit_budget=2000):
        """
        Attempts to find a Hamiltonian cycle using randomized DFS.
        Returns the list of vertices in the cycle or None.
        """
        n = len(adj)
        if n < 3:
            return None
            
        # Start from a random vertex with high degree
        vertices = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
        if not vertices:
            return None
        
        # Try a few random starts from the top degree vertices
        starts = vertices[:min(5, n)]
        rng.shuffle(starts)
        
        for start in starts:
            if not adj[start]:
                continue
                
            visited = {start}
            path = [start]
            visits = [0]
            found = [None]
            
            sys.setrecursionlimit(100000)
            
            def dfs(u):
                if visits[0] >= visit_budget or found[0] is not None:
                    return
                
                if len(path) == n:
                    # Check if we can close the cycle
                    if start in adj[u]:
                        found[0] = list(path)
                        return
                    return
                
                neighbors = list(adj[u])
                rng.shuffle(neighbors)
                
                for w in neighbors:
                    if visits[0] >= visit_budget or found[0] is not None:
                        return
                    visits[0] += 1
                    
                    if w not in visited:
                        visited.add(w)
                        path.append(w)
                        dfs(w)
                        path.pop()
                        visited.discard(w)
                    
                    if found[0] is not None:
                        return
                        
            dfs(start)
            if found[0]:
                return found[0]
                
        return None

    def find_longest_cycle_in_tree(adj, start, rng, visit_budget=50000):
        """
        Performs a DFS from 'start' and returns the longest cycle found in the DFS tree.
        This biases the algorithm towards longer cycles, which reduces the total piece count.
        """
        if not adj[start]:
            return None
        
        color = {start: 1}
        parent = {start: -1}
        path = [start]
        best_cycle = None
        best_len = 0
        visits = [0]
        
        sys.setrecursionlimit(100000)

        def dfs(u):
            nonlocal best_cycle, best_len
            
            if visits[0] >= visit_budget:
                return
            
            # Get neighbors and shuffle them to randomize the DFS path
            neighbors = list(adj[u])
            rng.shuffle(neighbors)
            
            for w in neighbors:
                if visits[0] >= visit_budget:
                    return
                
                visits[0] += 1
                if w == parent[u]:
                    continue
                
                if color.get(w, 0) == 0:
                    color[w] = 1
                    parent[w] = u
                    path.append(w)
                    dfs(w)
                    path.pop()
                    color[w] = 2
                elif color.get(w, 0) == 1:
                    # Back edge found, extract cycle
                    cyc = []
                    cur = u
                    while cur != w:
                        cyc.append(cur)
                        cur = parent[cur]
                    cyc.append(w)
                    cyc.reverse()
                    if len(cyc) > best_len:
                        best_len = len(cyc)
                        best_cycle = list(cyc)
        
        dfs(start)
        return best_cycle

    def find_cycle_heuristic(adj, rng):
        """
        Tries to find a long cycle by starting from high-degree vertices
        and searching the entire DFS tree for the longest cycle.
        """
        # Sort vertices by degree descending to start from hubs
        vertices = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
        
        best_overall_cycle = None
        best_overall_len = 0
        
        # Try top K starts to balance quality and speed
        K = min(10, len(vertices)) # Increased from 6
        
        for v in vertices[:K]:
            if not adj[v]:
                continue
            
            # Find longest cycle in the DFS tree rooted at v
            cyc = find_longest_cycle_in_tree(adj, v, rng)
            if cyc is not None and len(cyc) > best_overall_len:
                best_overall_len = len(cyc)
                best_overall_cycle = cyc
                
        return best_overall_cycle

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_pieces = None
        best_count = float('inf')
        
        # Use multiple randomized restarts
        # The heuristic is stronger, so we might need fewer restarts, or we can afford more
        max_restarts = 2000 
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(i)
            current_adj = {u: set(v) for u, v in adj.items()}
            current_edges = set(edges)
            pieces = []
            
            # Greedy cycle packing with heuristic
            while True:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Check if any edges remain
                if not any(len(neigh) > 0 for neigh in current_adj.values()):
                    break
                
                # First try to find a Hamiltonian cycle
                cycle = try_find_hamiltonian_cycle(current_adj, rng)
                
                # If no Hamiltonian cycle, fall back to heuristic
                if cycle is None:
                    cycle = find_cycle_heuristic(current_adj, rng)
                    
                if cycle is None:
                    break
                
                # Remove cycle edges
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(a, b), max(a, b))
                    current_edges.discard(e)
                    if b in current_adj[a]:
                        current_adj[a].discard(b)
                    if a in current_adj[b]:
                        current_adj[b].discard(a)
                
                pieces.append(list(cycle))
            
            # Add remaining edges as single pieces
            for e in current_edges:
                pieces.append([e[0], e[1]])
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
                
        if best_pieces is None:
             # Fallback: just return edges
            return [[u, v] for u, v in edges]
             
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
