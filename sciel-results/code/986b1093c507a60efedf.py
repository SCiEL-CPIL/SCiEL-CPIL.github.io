"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
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

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        remaining = set(edges)
        pieces = []
        
        # Build dynamic adjacency for remaining edges
        def build_adj(rem):
            adj = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            return adj

        adj = build_adj(remaining)

        def find_longest_cycle(start, max_len=50):
            """Find a long cycle starting and ending at start using randomized greedy DFS."""
            if len(remaining) < 3:
                return None
            
            best_cycle = None
            
            # Calculate degrees for greedy choice
            degs = {}
            for u, v in remaining:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + 1

            # Try multiple randomized greedy walks
            for _ in range(200):
                path = [start]
                visited = {start}
                current = start
                success = False
                
                for step in range(max_len):
                    neighbors = [nb for nb in adj.get(current, []) 
                                if (min(current, nb), max(current, nb)) in remaining]
                    
                    if not neighbors:
                        break
                    
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    
                    if start in neighbors and len(path) >= 3:
                        # Close the cycle if no unvisited or if path is reasonably long
                        if not unvisited or len(path) >= 10:
                            path.append(start)
                            success = True
                            break
                    
                    if unvisited:
                        # Pick neighbor with highest degree (top 3 for randomness)
                        top_neighbors = sorted(unvisited, key=lambda x: -degs.get(x, 0))
                        if len(top_neighbors) > 3:
                            top_neighbors = top_neighbors[:3]
                        next_node = random.choice(top_neighbors)
                    else:
                        # Only visited neighbors left, try to close cycle
                        if start in neighbors and len(path) >= 3:
                            path.append(start)
                            success = True
                            break
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if success and (best_cycle is None or len(path) > len(best_cycle)):
                    best_cycle = path[:]
            
            return best_cycle

        def find_cycle_greedy():
            """Find a cycle using a greedy approach that prefers longer cycles."""
            if len(remaining) < 3:
                return None
            
            # Find vertices with degree >= 2
            vertex_degrees = {}
            for u, v in remaining:
                vertex_degrees[u] = vertex_degrees.get(u, 0) + 1
                vertex_degrees[v] = vertex_degrees.get(v, 0) + 1
            
            candidates = [v for v in vertex_degrees if vertex_degrees[v] >= 2]
            if not candidates:
                return None
            
            # Sort candidates by degree descending to start from high-degree nodes
            candidates.sort(key=lambda x: -vertex_degrees[x])
            
            best_cycle = None
            
            # Try top 5 candidate vertices to balance speed and quality
            for start in candidates[:5]:
                cycle = find_longest_cycle(start)
                if cycle and (best_cycle is None or len(cycle) > len(best_cycle)):
                    best_cycle = cycle
            
            return best_cycle

        # Greedily extract cycles, preferring longer ones
        while remaining:
            cycle = find_cycle_greedy()
            
            if cycle:
                # Extract this cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    edge = (min(u, v), max(u, v))
                    if edge in remaining:
                        remaining.discard(edge)
                pieces.append(cycle[:-1][:])
                # Rebuild adjacency
                adj = build_adj(remaining)
            else:
                # No more cycles; take remaining edges as singles
                for e in sorted(remaining):
                    pieces.append([e[0], e[1]])
                remaining = set()
                break

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
