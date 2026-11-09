import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before the 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_long_cycle(adj, rng):
        """
        Find a long cycle in the graph defined by adj.
        Uses a randomized DFS/Backtracking approach similar to the inspiration.
        """
        if not adj:
            return None
        
        # Vertices with degree >= 2 are candidates for cycle start
        candidates = [u for u, neighbors in adj.items() if len(neighbors) >= 2]
        if not candidates:
            return None
        
        best_cycle = None
        best_len = 0
        
        # Try a few random starts, biased towards high degree
        # Sort by degree descending, take top few, add some random
        candidates_sorted = sorted(candidates, key=lambda x: len(adj[x]), reverse=True)
        starts = candidates_sorted[:5]
        if len(candidates) > 5:
            starts += rng.sample(candidates, min(5, len(candidates)))
        
        starts = list(set(starts))
        rng.shuffle(starts)
        
        for start in starts:
            # DFS to find a cycle back to start
            # We want the longest path that returns to start.
            # This is NP-hard generally, but for small graphs, a greedy/DFS heuristic works.
            
            # Simple DFS that tracks the path and tries to close it
            visited = {start: True}
            path = [start]
            
            # We'll do a recursive search with a depth limit or just standard DFS
            # Since we want a cycle, we look for a neighbor of start in the path.
            
            def dfs(u, current_path):
                nonlocal best_cycle, best_len
                
                # Try to close cycle
                for w in adj[u]:
                    if w == start and len(current_path) >= 3:
                        # Found a cycle
                        if len(current_path) > best_len:
                            best_len = len(current_path)
                            best_cycle = list(current_path)
                        return True # Stop this branch? Or continue? 
                        # Usually finding the longest is hard. Let's just find *a* long one.
                        # To find *the* longest, we'd need backtracking that explores all.
                        # Given time constraints, finding a reasonably long one is enough.
                        # The "greedy" part is that we remove edges.
                        # Let's just return the first valid cycle found of decent length?
                        # No, let's try to find a longer one by not returning immediately?
                        # Actually, standard DFS finds *a* cycle. 
                        # To maximize length, we should prefer paths that don't close yet.
                        # But closing early might block longer cycles later? No, edges are removed.
                        # So we want the longest cycle in the *current* graph.
                        
                        # Heuristic: DFS that prioritizes unvisited nodes?
                        # If we prioritize unvisited, we go deep.
                        # When we hit a dead end, we backtrack.
                        # If we hit a node connected to start, we found a cycle.
                        # Is it the longest? Not necessarily.
                        # But for these small graphs, it's likely good enough.
                        
                        # Let's try a different strategy: 
                        # 1. Pick a random edge (u,v).
                        # 2. Find a path from v to u in G - (u,v).
                        # 3. The path + (u,v) is a cycle.
                        # 4. To make it long, find the *longest* path from v to u.
                        # Longest path is NP-hard, but for small n, we can use DFS with pruning.
                        
                        pass
                return False

            # Let's implement the "Longest Path" heuristic properly.
            # It's expensive. 
            # Alternative: Just use the inspiration's approach which worked well.
            # The inspiration used a DFS that tracks color and finds back edges.
            
            # Let's restart with the Inspiration 1 logic inside find_long_cycle
            color = {start: 1}
            parent = {start: -1}
            path = [start]
            local_best_cycle = None
            local_best_len = 0
            
            def dfs(u):
                nonlocal local_best_cycle, local_best_len
                
                neighbors = list(adj[u])
                rng.shuffle(neighbors)
                
                for w in neighbors:
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
                        # Back edge found
                        # w is on the current path
                        try:
                            idx = path.index(w)
                            cyc = path[idx:]
                            if len(cyc) > local_best_len:
                                local_best_len = len(cyc)
                                local_best_cycle = list(cyc)
                        except ValueError:
                            pass
            
            dfs(start)
            
            if local_best_cycle and local_best_len > best_len:
                best_len = local_best_len
                best_cycle = local_best_cycle
                
            if best_len > 15: # Heuristic cutoff
                break

        return best_cycle

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_pieces = None
        best_count = float('inf')
        
        # Multiple restarts with different random seeds for the search
        # The inspiration did 500 restarts. We can do fewer if we're slower, or more if faster.
        # Let's try 100-200 restarts.
        
        max_restarts = 200
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(i)
            # Deep copy adj for this restart
            current_adj = {u: set(v) for u, v in adj.items()}
            current_edges = set(tuple(sorted(e)) for e in edges)
            pieces = []
            
            # Loop until no more cycles found
            while True:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Check if any edges remain with degree >= 2
                has_cycle_potential = False
                for u, neighbors in current_adj.items():
                    if len(neighbors) >= 2:
                        has_cycle_potential = True
                        break
                
                if not has_cycle_potential:
                    break
                
                cycle = find_long_cycle(current_adj, rng)
                if cycle is None:
                    break
                
                # Remove cycle edges
                for j in range(len(cycle)):
                    a, b = cycle[j], cycle[(j + 1) % len(cycle)]
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
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results

if __name__ == "__main__":
    construct()
