import time
import random
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    # Precompute edge list
    def get_edges(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges

    # Check if adding edge (u, v) with color c creates a C6
    # This is true if there is a path of length 5 between u and v in the graph of color c
    def has_path_5(u, v, adj, c, n):
        # DFS from u to v, depth 5
        # adj[c] is a list of lists of neighbors
        # We want to know if there is a simple path of length 5
        visited = [False] * n
        visited[u] = True
        
        def dfs(node, depth):
            if depth == 5:
                return node == v
            for neighbor in adj[c][node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if dfs(neighbor, depth + 1):
                        return True
                    visited[neighbor] = False
            return False
            
        return dfs(u, 0)

    def search_n(n, time_budget):
        edges = get_edges(n)
        m = len(edges)
        colors = [-1] * m
        # adj[c][u] = list of neighbors
        adj = [[[] for _ in range(n)] for _ in range(4)]
        
        def is_valid_move(idx, c):
            u, v = edges[idx]
            # Check if path of length 5 exists in color c between u and v
            if has_path_5(u, v, adj, c, n):
                return False
            return True

        def add_edge(idx, c):
            u, v = edges[idx]
            adj[c][u].append(v)
            adj[c][v].append(u)
            colors[idx] = c

        def remove_edge(idx, c):
            u, v = edges[idx]
            adj[c][u].remove(v)
            adj[c][v].remove(u)
            colors[idx] = -1

        def backtrack(idx):
            if time.time() > deadline - 5:
                return False, "timeout"
            
            if idx == m:
                return True, "success"
            
            u, v = edges[idx]
            # Try colors 0, 1, 2, 3 in random order
            color_order = [0, 1, 2, 3]
            random.shuffle(color_order)
            
            for c in color_order:
                if is_valid_move(idx, c):
                    add_edge(idx, c)
                    success, reason = backtrack(idx + 1)
                    if success:
                        return True, reason
                    remove_edge(idx, c)
                    if reason == "timeout":
                        return False, reason
            
            return False, "backtrack"

        # Run search with restarts
        while time.time() < start_time + time_budget:
            # Reset
            for i in range(m):
                colors[i] = -1
            for c in range(4):
                for u in range(n):
                    adj[c][u] = []
            
            # We can also shuffle the edge order to diversify search
            # But standard order is often better for constraint propagation if we had it.
            # Here, we just rely on random color order.
            
            success, reason = backtrack(0)
            if success:
                return {'n': n, 'colors': colors}
            if reason == "timeout":
                break
            # If we backtracked all the way, restart with new randomness

        return None

    # Targets
    # We want to maximize n.
    # Try 17, 16, 15...
    
    targets = [17, 16, 15, 14, 13, 12, 11, 10]
    
    # Time allocation
    # We have ~140s.
    # Give 17 a large chunk.
    
    time_budgets = {
        17: 40.0,
        16: 40.0,
        15: 30.0,
        14: 30.0,
        13: 30.0,
        12: 30.0,
        11: 30.0,
        10: 30.0
    }
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        budget = time_budgets.get(n, 10.0)
        # Ensure we don't exceed deadline
        if time.time() + budget > deadline - 5:
            budget = deadline - 5 - time.time()
            if budget < 1.0:
                break
                
        result = search_n(n, budget)
        
        if result:
            best_result = result
            # If we found a valid coloring for n, and n is high, we can return.
            # Since we iterate from high to low, the first success is the best found.
            # We can return immediately.
            return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
