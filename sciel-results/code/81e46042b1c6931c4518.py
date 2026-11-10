import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Precompute adjacency list
adj = {i: [] for i in range(P['n'])}
edge_set = set()
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)
    edge_set.add(frozenset((u, v)))

# Sort neighbors to help heuristic search (e.g., lower degree first might prune faster, or higher degree to find cycles quicker? 
# Usually, in Hamiltonian path search, connecting to nodes with fewer remaining options is good. 
# Let's just sort by vertex ID for determinism, or try a simple heuristic.)
for i in adj:
    adj[i].sort()

def find_hamiltonian_cycle(used_edges, start=0, time_limit=None, start_time=None):
    """
    Finds a Hamiltonian cycle using backtracking.
    Returns a list of vertices representing the cycle, or None if not found.
    """
    if time_limit is None:
        time_limit = 5.0
    if start_time is None:
        start_time = time.time()
        
    n = P['n']
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    def dfs(current, depth):
        if time.time() - start_time > time_limit:
            return None
        
        if depth == n:
            # Check if closing edge is valid and not used
            last = current
            first = start
            edge = frozenset((last, first))
            if edge in edge_set and edge not in used_edges:
                return list(path)
            return None
        
        for neighbor in adj[current]:
            if not visited[neighbor]:
                edge = frozenset((current, neighbor))
                if edge not in used_edges:
                    visited[neighbor] = True
                    path.append(neighbor)
                    
                    result = dfs(neighbor, depth + 1)
                    if result is not None:
                        return result
                    
                    path.pop()
                    visited[neighbor] = False
                    
        return None

    return dfs(start, 1)

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s kill
    
    cycles = []
    used_edges = set()
    
    # Try to find up to 3 cycles
    # We can try different starting vertices or just rely on the search.
    # Since the graph is symmetric enough, starting at 0 is fine, but if the first cycle
    # blocks the existence of a second, we might need to backtrack the choice of the first cycle.
    # However, for small N, we can just try to find 3. If we fail to find the 3rd, we return 2.
    # If we fail to find the 2nd, we return 1.
    
    # To be more robust, we could implement a backtracking search for k cycles,
    # but given the time limit, a greedy approach with a few retries might suffice.
    # Let's try to find 3 cycles greedily first.
    
    for i in range(3):
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 1.0:
            break
            
        # We can vary the start vertex to increase diversity, but the cycle must include all vertices.
        # The start vertex doesn't change the edge set of the cycle, just the representation.
        # So finding a cycle starting at 0 is equivalent to finding any cycle.
        # The challenge is edge disjointness.
        
        cycle = find_hamiltonian_cycle(used_edges, start=0, time_limit=remaining_time - 0.5, start_time=time.time())
        
        if cycle is None:
            break
        
        # Add edges of this cycle to used_edges
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            used_edges.add(frozenset((u, v)))
            
        cycles.append(cycle)
        
    # Ensure we return a valid list of lists of ints
    # If no cycles found, we should return a valid single cycle to score 1.0 instead of 0.
    # But the graph is dense, so we should find at least one.
    if not cycles:
        # Fallback: Find any Hamiltonian cycle without edge constraints
        # This is basically the same as find_hamiltonian_cycle with empty used_edges
        cycle = find_hamiltonian_cycle(set(), start=0, time_limit=10.0, start_time=time.time())
        if cycle:
            cycles.append(cycle)
        else:
            # Extreme fallback: return a dummy valid cycle? No, must be valid.
            # If we can't find any, we fail. But this graph definitely has one.
            pass

    return cycles
# EVOLVE-BLOCK-END
