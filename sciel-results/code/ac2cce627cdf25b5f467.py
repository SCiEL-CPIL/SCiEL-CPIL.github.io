import time
import sys
from random import Random
import networkx as nx
import heapq

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposes three fixed graphs into few cycles and edges.
    Strategy: Iteratively extract simple cycles using a greedy heuristic.
    1. Generate the instance graph.
    2. While edges remain:
       a. Find a simple cycle in the remaining graph.
       b. Add the cycle to the decomposition.
       c. Remove the edges of the cycle from the graph.
    3. Any remaining edges become individual pieces.
    
    This minimizes pieces because a cycle of length k replaces k single-edge pieces 
    with 1 cycle piece.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
        
        pieces = []
        remaining_edges = set(edges)
        
        # Build adjacency structure for fast lookup
        # We will update this as we remove edges.
        adj = {}
        for u, v in remaining_edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Helper to find a simple cycle in the current graph
        def find_cycle():
            # Try to find a cycle. We use a DFS approach.
            # To make it "greedy" for efficiency, we can just pick any cycle found.
            # A simple cycle in a graph with edges exists if the graph is not a forest.
            
            # We'll use a standard cycle finding algorithm.
            # Since N is small (20-30), this is fast.
            visited = set()
            parent = {}
            
            def dfs(u):
                visited.add(u)
                for v in adj.get(u, []):
                    if v not in visited:
                        parent[v] = u
                        if dfs(v):
                            return True
                    elif parent.get(u) != v:
                        # Found a back edge
                        # Reconstruct cycle
                        cycle = []
                        curr = u
                        while curr != v:
                            cycle.append(curr)
                            curr = parent[curr]
                        cycle.append(v)
                        # The cycle is v -> ... -> u -> v
                        # We need to return the list of vertices in order
                        return cycle + [u]
                return False

            for node in list(adj.keys()):
                if node not in visited:
                    res = dfs(node)
                    if res:
                        return res
            
            return None

        while remaining_edges:
            if time.time() > deadline:
                # Time limit reached, break and return what we have
                # We must ensure we don't return partial invalid state if we can't finish,
                # but the problem says "return best result found". 
                # However, a partial decomposition is invalid if it doesn't cover all edges.
                # So we must finish or fall back to single edges for the rest.
                # Given N<=30, this loop should be very fast. 
                # If it times out, we just append remaining as edges.
                break
                
            cycle = find_cycle()
            if not cycle:
                break
            
            # Convert cycle vertices to edge list
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                cycle_edges.add(e)
            
            # Verify all edges exist (should be true)
            if all(e in remaining_edges for e in cycle_edges):
                pieces.append(cycle)
                # Remove edges
                for e in cycle_edges:
                    remaining_edges.discard(e)
                    u, v = e
                    if u in adj and v in adj[u]:
                        adj[u].discard(v)
                    if v in adj and u in adj[v]:
                        adj[v].discard(u)
                    if not adj.get(u):
                        del adj[u]
                    if not adj.get(v):
                        del adj[v]
            else:
                # Inconsistency, break to avoid infinite loop
                break
        
        # Any remaining edges are single pieces
        for e in remaining_edges:
            pieces.append(list(e))
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # If we run out of time, we can't generate the next graph properly?
            # We should have generated them in advance or be fast enough.
            # Let's assume the loop above is fast enough for 3 graphs.
            # If we are here, we just return what we have, but we need 3 lists.
            # This shouldn't happen.
            pass
        
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
