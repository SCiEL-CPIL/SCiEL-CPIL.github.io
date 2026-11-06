"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _get_arrival():
    """Build dict mapping (u,v) sorted tuple to one-based arrival index."""
    n = P['n']
    arrival = {}
    for i, (u, v) in enumerate(P['edge_order'], 1):
        key = (min(u, v), max(u, v))
        arrival[key] = i
    return arrival

def _find_hamilton_cycle_in_subgraph(adj, n, start, deadline, best_score_ref):
    """
    DFS to find a Hamilton cycle in the subgraph defined by adj (list of sets).
    adj[v] contains neighbors connected via edges in the current prefix.
    Returns a list of vertices forming the cycle, or None.
    best_score_ref[0] is the current best score; we only accept cycles with max arrival < best.
    """
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    def dfs(u):
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if edge from last to start exists
            last = path[-1]
            if start in adj[last]:
                return path[:]
            return None
        
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                path.append(v)
                result = dfs(v)
                if result is not None:
                    return result
                path.pop()
                visited[v] = False
        return None
    
    return dfs(start)

def construct():
    n = P['n']
    arrival = _get_arrival()
    deadline = time.time() + 80  # Leave 10s safety margin for 90s limit
    
    best_cycle = list(range(n))  # fallback: identity permutation
    best_score = max(arrival[(min(u,v), max(u,v))] for u, v in 
                     [(best_cycle[i], best_cycle[(i+1) % n]) for i in range(n)])
    
    # Try increasing prefix sizes
    # Minimum possible is n (24 edges for a cycle on 24 vertices)
    for prefix_size in range(n, n + 100):
        if time.time() > deadline:
            break
        
        # Build adjacency list using only edges arriving in first prefix_size positions
        adj = [set() for _ in range(n)]
        max_arr_in_prefix = 0
        for i, (u, v) in enumerate(P['edge_order'][:prefix_size], 1):
            adj[u].add(v)
            adj[v].add(u)
            max_arr_in_prefix = i
        
        # Check if all vertices have degree >= 2 (necessary for Hamilton cycle)
        if any(len(adj[v]) < 2 for v in range(n)):
            continue
        
        # Try DFS from vertex 0
        result = _find_hamilton_cycle_in_subgraph(adj, n, 0, deadline, best_score)
        if result is not None:
            # Verify and compute score
            score = 0
            valid = True
            for i in range(n):
                u = result[i]
                v = result[(i + 1) % n]
                key = (min(u, v), max(u, v))
                if key not in arrival:
                    valid = False
                    break
                score = max(score, arrival[key])
            if valid and score < best_score:
                best_score = score
                best_cycle = result
                # If we found a cycle with score == prefix_size, that's optimal for this prefix
                # But we can still try to find better with smaller prefix (already done) or same prefix with lower max
                # Actually since we're increasing prefix_size, first found is best for that prefix
                # But different cycles in same prefix might have different max arrival
                # For now, accept and move on to next prefix (which won't be better)
                # Actually we should break because larger prefix can't give lower score
                break
    
    return best_cycle
# EVOLVE-BLOCK-END
