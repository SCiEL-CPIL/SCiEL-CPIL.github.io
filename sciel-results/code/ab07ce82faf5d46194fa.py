import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_2factor_decomposition(edges, rng):
        """
        Find a 2-factor (spanning subgraph with all vertices even degree)
        then decompose into cycles.
        """
        if not edges:
            return [], set()
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        
        # Step 1: Find a spanning subgraph where all vertices have even degree
        # Use a randomized approach: keep adding edges until we can find a 2-factor
        
        # First, try to find a 2-factor by repeatedly finding cycles
        # and removing them, but track which edges form the 2-factor
        
        # Alternative approach: Use the fact that any graph can be decomposed
        # into paths and cycles. We want to minimize pieces.
        
        # Better approach: Find maximum matching-like structure
        # Actually, let's use a different strategy:
        # 1. Find all cycles in the graph
        # 2. Select a set of edge-disjoint cycles that covers as many edges as possible
        # 3. Remaining edges are single-edge pieces
        
        # Let's try: repeatedly find longest cycles
        cycles = []
        
        # Work with a copy of the graph
        work_adj = {u: set(neigh) for u, neigh in adj.items()}
        work_edges = set(remaining_edges)
        
        # Try multiple times to find good cycles
        for attempt in range(50):
            if time.time() - start_time > DEADLINE:
                break
            
            if not work_edges:
                break
            
            # Find a cycle using DFS
            best_cycle = None
            vertices_with_edges = [u for u in work_adj if work_adj[u]]
            
            if not vertices_with_edges:
                break
            
            # Try different starting points
            for start in rng.sample(vertices_with_edges, min(10, len(vertices_with_edges))):
                path = [start]
                visited = {start}
                current = start
                
                # DFS to find a cycle back to start
                while len(path) < len(work_adj):
                    neighbors = [n for n in work_adj[current] if n not in visited]
                    if not neighbors:
                        break
                    
                    # Prefer neighbors with fewer unvisited options (Warnsdorff)
                    neighbors.sort(key=lambda n: len(work_adj[n] - visited))
                    next_node = neighbors[0]
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    
                    # Check if we can close the cycle
                    if path[0] in work_adj[current] and len(path) >= 3:
                        if best_cycle is None or len(path) > len(best_cycle):
                            best_cycle = list(path)
                        break
            
            if best_cycle:
                cycles.append(best_cycle)
                # Remove these edges from work graph
                for i in range(len(best_cycle)):
                    u, v = best_cycle[i], best_cycle[(i+1) % len(best_cycle)]
                    e = (min(u,v), max(u,v))
                    work_edges.discard(e)
                    work_adj[u].discard(v)
                    work_adj[v].discard(u)
            else:
                break
        
        # Remaining edges are single-edge pieces
        remaining = list(work_edges)
        
        return cycles, remaining

    def decompose(edges):
        if not edges:
            return []
            
        best_pieces = None
        best_score = float('inf')
        
        # Try multiple approaches
        # Approach 1: Greedy cycle extraction
        num_restarts = 30
        
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(2000 + r)
            cycles, remaining = find_2factor_decomposition(edges, rng)
            
            score = len(cycles) + len(remaining)
            
            if score < best_score:
                best_score = score
                best_pieces = []
                for c in cycles:
                    best_pieces.append(list(c))
                for e in remaining:
                    best_pieces.append([e[0], e[1]])
        
        if best_pieces is None:
             # Fallback: all edges
             return [[u, v] for u, v in edges]
             
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
