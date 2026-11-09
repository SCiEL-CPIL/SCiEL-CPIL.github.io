import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 16
    best_edges = None
    best_count = 0
    
    def check_diameter_two(adj):
        """Check if graph has diameter exactly 2."""
        # Check all pairs have distance <= 2
        for i in range(n):
            if i not in adj:
                continue
            visited = {i}
            for j in adj[i]:
                visited.add(j)
            for j in list(visited):
                if j == i:
                    continue
                for k in adj[j]:
                    if k not in visited:
                        visited.add(k)
            if len(visited) != n:
                return False
        # Check not complete (diameter exactly 2, not 1)
        edge_count = sum(len(adj[i]) for i in adj) // 2
        if edge_count == n*(n-1)//2:
            return False
        return True
    
    def check_all_edges_essential(adj):
        """Check if every edge is essential for diameter-2."""
        for i in range(n):
            for j in adj[i]:
                if i >= j:
                    continue
                # Remove edge (i,j)
                adj2 = {k: list(v) for k, v in adj.items()}
                adj2[i].remove(j)
                adj2[j].remove(i)
                if not check_diameter_two(adj2):
                    # Edge is essential
                    continue
                else:
                    return False
        return True
    
    # Try several random dense graphs
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        # Random graph with high edge probability
        p = random.uniform(0.4, 0.7)
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i].append(j)
                    adj[j].append(i)
        
        if not check_diameter_two(adj):
            continue
        
        if check_all_edges_essential(adj):
            edge_count = sum(len(adj[i]) for i in adj) // 2
            if edge_count > best_count:
                best_count = edge_count
                best_edges = [[i, j] for i in range(n) for j in adj[i] if i < j]
        
        # Also try: take current best, try adding edges
        if best_edges is not None and time.time() < deadline - 10:
            # Try to add one edge to best and check
            adj_best = {i: set() for i in range(n)}
            for e in best_edges:
                adj_best[e[0]].add(e[1])
                adj_best[e[1]].add(e[0])
            
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in [(min(e), max(e)) for e in best_edges]:
                        if i not in adj_best[j]:
                            # Try adding
                            adj_try = {k: set(v) for k, v in adj_best.items()}
                            adj_try[i].add(j)
                            adj_try[j].add(i)
                            if check_diameter_two(adj_try):
                                if check_all_edges_essential(adj_try):
                                    edge_count = sum(len(v) for v in adj_try.values()) // 2
                                    if edge_count > best_count:
                                        best_count = edge_count
                                        best_edges = [[i, j] for i in range(n) for j in adj_try[i] if i < j]
    
    if best_edges is None:
        # Fallback: star
        return [[0, v] for v in range(1, n)]
    
    return best_edges
