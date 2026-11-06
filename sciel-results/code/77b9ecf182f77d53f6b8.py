import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

# Precompute arrival index (0-based)
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    arrival[(min(u, v), max(u, v))] = idx

def get_weight(u, v):
    if u > v:
        u, v = v, u
    return arrival[(u, v)] + 1 # 1-based score

def has_hamiltonian_cycle(adj, n, deadline):
    """
    Checks if the graph defined by adjacency list 'adj' has a Hamiltonian cycle.
    adj[u] is a set of neighbors v such that edge (u,v) exists in G_K.
    Returns the cycle as a list of nodes if found, else None.
    """
    # Fix node 0 as start.
    # To break symmetry, we can fix node 1 as the second node if 1 is a neighbor of 0.
    # But we must be careful: what if the cycle is 0-2-...-1-0?
    # We can just iterate neighbors of 0.
    
    # Optimization: If any node has degree < 2, return None.
    for u in range(n):
        if len(adj[u]) < 2:
            return None

    # Use DFS with backtracking
    # State: current node, visited mask, path
    # We use recursion with a deadline check.
    
    sys.setrecursionlimit(10000)
    
    found_cycle = [None]
    
    def dfs(u, visited_mask, path):
        if time.time() > deadline:
            return False
        
        if len(path) == n:
            # Check if we can close the cycle to 0
            if 0 in adj[u]:
                found_cycle[0] = path[:]
                return True
            return False
        
        # Pruning: Check connectivity of unvisited nodes + u + 0
        # This is expensive. Let's try a simpler pruning first.
        
        # Pruning: If u cannot go to any unvisited node and cannot go to 0 (if all visited), fail.
        # But we only check closing to 0 when len(path)==n.
        
        # Degree pruning for unvisited nodes:
        # Every unvisited node must have at least 2 neighbors in (unvisited U {u, 0})
        # This is O(N^2) per step, which is 24^2 = 576. Acceptable?
        # Let's try it.
        
        unvisited = [v for v in range(n) if not (visited_mask & (1 << v))]
        if unvisited:
            # The set of available nodes for the sub-path is unvisited + {u} + {0} (for the end connection)
            # Actually, the sub-path must start at u and end at 0 (conceptually, or rather, connect to 0 at the end).
            # The edges used will be within {u} U unvisited, plus one edge from the last node to 0.
            # So, for each v in unvisited, it needs degree >= 2 in the graph induced by {u} U unvisited, 
            # OR degree >= 1 in induced graph + 1 edge to 0.
            
            # Simpler check:
            # Build a small graph for {u} U unvisited.
            # Check if it's connected.
            # Check if every node in unvisited has degree >= 2 in {u} U unvisited, 
            # or degree >= 1 in {u} U unvisited and connected to 0.
            
            # Let's just check connectivity of {u} U unvisited.
            # If not connected, fail.
            
            # Find component of u in {u} U unvisited
            comp = set()
            stack = [u]
            comp.add(u)
            while stack:
                node = stack.pop()
                for neighbor in adj[node]:
                    if neighbor in unvisited or neighbor == u: # neighbor is in {u} U unvisited
                        if neighbor not in comp:
                            comp.add(neighbor)
                            stack.append(neighbor)
            
            # All unvisited nodes must be in comp
            for v in unvisited:
                if v not in comp:
                    return False
            
            # Degree check for unvisited nodes
            for v in unvisited:
                deg = 0
                for neighbor in adj[v]:
                    if neighbor in comp: # neighbor is in {u} U unvisited
                        deg += 1
                # If v is the last node before 0, it needs 1 edge to comp and 1 to 0.
                # So min degree in comp is 1.
                # If v is not last, it needs 2 edges in comp.
                # We don't know which is last.
                # But if deg == 0, fail.
                # If deg == 1, v MUST be the last node (connected to 0).
                # So there can be at most ONE node with deg==1 in comp that is connected to 0.
                # And all others must have deg >= 2.
                
                if deg == 0:
                    return False
                # We'll handle the "last node" logic more carefully later or just rely on connectivity + basic degree.
        
        # Try next nodes
        # Order neighbors by weight? No, we are checking for existence in G_K.
        # Just iterate.
        for v in adj[u]:
            if not (visited_mask & (1 << v)):
                if dfs(v, visited_mask | (1 << v), path + [v]):
                    return True
        return False

    # Start DFS
    # Fix 0 as start.
    # To reduce symmetry, we can try neighbors of 0 in a specific order.
    # Also, we can fix the second node if possible?
    # Let's just iterate all neighbors of 0.
    
    for v in adj[0]:
        if dfs(v, (1 << 0) | (1 << v), [0, v]):
            return found_cycle[0]
            
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    # Binary search for the minimum K
    # Lower bound: 24 (min possible score for 24 edges)
    # Upper bound: m (max possible score)
    
    low = 24
    high = m
    best_cycle = None
    
    # We can also try a greedy construction + local search as a fallback or to get a good upper bound.
    # But let's stick to the exact search first.
    
    while low <= high:
        if time.time() > deadline - 5.0:
            break
            
        mid = (low + high) // 2
        
        # Build adjacency list for G_mid
        adj = [set() for _ in range(n)]
        for idx, (u, v) in enumerate(edge_order):
            if idx + 1 <= mid:
                adj[u].add(v)
                adj[v].add(u)
        
        # Check for Hamiltonian cycle
        cycle = has_hamiltonian_cycle(adj, n, deadline)
        
        if cycle:
            best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1
            
    if best_cycle is None:
        # Fallback: return a valid cycle (e.g., 0-1-2...-23-0)
        # This might have a high score, but it's valid.
        return list(range(n))
        
    return best_cycle

# EVOLVE-BLOCK-END
