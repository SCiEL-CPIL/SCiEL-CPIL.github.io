import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    n = 40
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best_edges = []
    best_count = 0

    def has_c6_path(u, v, adj):
        """
        Returns True if there is a simple path of length 5 between u and v in graph adj.
        adj is a list of lists of neighbors.
        """
        # Iterative DFS with explicit stack to avoid recursion overhead/limits
        # State: (node, depth, mask)
        # We need to backtrack, so we store the stack of (node, next_neighbor_index, mask)
        # Or just use recursion since depth is 5. Recursion is simpler and likely fast enough for depth 5.
        
        def dfs(node, depth, mask):
            if depth == 5:
                return node == v
            for nxt in adj[node]:
                bit = 1 << nxt
                if not (mask & bit):
                    if dfs(nxt, depth + 1, mask | bit):
                        return True
            return False

        return dfs(u, 0, 1 << u)

    def greedy_build(seed):
        random.seed(seed)
        adj = [[] for _ in range(n)]
        
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        count = 0
        for u, v in edges_to_try:
            # Check if adding (u,v) creates a C6
            # This is equivalent to checking if there is a simple path of length 5 between u and v in the current graph.
            if not has_c6_path(u, v, adj):
                # Add edge
                adj[u].append(v)
                adj[v].append(u)
                count += 1
                
        # Return list of edges
        res_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    res_edges.append([u, v])
        return res_edges, count

    # Run multiple restarts
    seed = 0
    # We can also vary the random seed strategy
    while time.time() < deadline:
        edges, count = greedy_build(seed)
        if count > best_count:
            best_count = count
            best_edges = edges
            
        seed += 1
        # Every 100 seeds, we might try a different initialization? 
        # For now, keep it simple.
        
    return best_edges
# EVOLVE-BLOCK-END
