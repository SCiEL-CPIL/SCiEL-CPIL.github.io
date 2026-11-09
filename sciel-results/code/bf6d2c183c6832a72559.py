import random
import itertools
import time
import sys

def construct():
    # New approach: Local search to find a graph with no regular induced subgraphs.
    # We aim for n=21, then 20, etc.
    
    def count_regular_subgraphs(n, adj, size=6):
        """
        Count the number of induced subgraphs of the given size that are regular.
        This is used as a cost function for local search.
        """
        count = 0
        vertices = range(n)
        for subset in itertools.combinations(vertices, size):
            # Check if induced subgraph is regular
            # Optimize: pre-compute degrees within subset?
            # For size 6, this is small.
            degs = []
            for i in subset:
                d = 0
                for j in subset:
                    if i != j and j in adj[i]:
                        d += 1
                degs.append(d)
            if len(set(degs)) == 1:
                count += 1
        return count

    def is_valid_graph(n, adj):
        """
        Check if the graph is valid: no induced subgraph of size >= 6 is regular.
        """
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                degs = []
                for i in subset:
                    d = 0
                    for j in subset:
                        if i != j and j in adj[i]:
                            d += 1
                degs.append(d)
                if len(set(degs)) == 1:
                    return False
        return True

    def local_search(n, max_iterations=5000):
        """
        Try to find a graph on n vertices with 0 regular induced subgraphs of size 6.
        Then verify for all sizes >= 6.
        """
        # Initialize with random graph
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u].add(v)
                    adj[v].add(u)
        
        current_cost = count_regular_subgraphs(n, adj, 6)
        
        # If already 0, check full validity
        if current_cost == 0:
            if is_valid_graph(n, adj):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges

        # Local search
        best_cost = current_cost
        best_adj = [s.copy() for s in adj]
        
        for i in range(max_iterations):
            if time.time() > 110: # Time limit safety
                break
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
            
            new_cost = count_regular_subgraphs(n, adj, 6)
            
            if new_cost <= best_cost:
                best_cost = new_cost
                best_adj = [s.copy() for s in adj]
                if best_cost == 0:
                    # Check full validity
                    if is_valid_graph(n, best_adj):
                        edges = []
                        for u in range(n):
                            for v in best_adj[u]:
                                if u < v:
                                    edges.append([u, v])
                        return edges
            else:
                # Revert
                if v in adj[u]:
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Occasionally restart if stuck?
            if i % 1000 == 0 and best_cost > 0:
                # Try a different random start if stuck too long
                if i > 2000:
                    adj = [set() for _ in range(n)]
                    for u in range(n):
                        for v in range(u + 1, n):
                            if random.random() < 0.5:
                                adj[u].add(v)
                                adj[v].add(u)
                    current_cost = count_regular_subgraphs(n, adj, 6)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = [s.copy() for s in adj]
        
        # If we didn't find one with 0 cost in search, try the best found
        if best_cost == 0:
            if is_valid_graph(n, best_adj):
                edges = []
                for u in range(n):
                    for v in best_adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        
        return None

    # Try n=21 first
    start_time = time.time()
    edges = local_search(21, max_iterations=2000)
    if edges is not None:
        return {'n': 21, 'edges': edges}
        
    if time.time() - start_time > 30:
        # Try n=20
        edges = local_search(20, max_iterations=2000)
        if edges is not None:
            return {'n': 20, 'edges': edges}
            
    if time.time() - start_time > 60:
        # Try n=19
        edges = local_search(19, max_iterations=2000)
        if edges is not None:
            return {'n': 19, 'edges': edges}
            
    # Fallback to a known small valid graph if all else fails
    # n=5 is trivially valid (no subset of size 6)
    return {'n': 5, 'edges': []}
