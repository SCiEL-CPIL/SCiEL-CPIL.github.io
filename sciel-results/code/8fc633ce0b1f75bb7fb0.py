"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Pre-compute adjacency list for speed
adj = {i: set() for i in range(P['n'])}
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

def find_hamiltonian_cycle(available_edges_set, start=0):
    """
    Finds a Hamiltonian cycle using only edges in available_edges_set.
    Returns a list of vertices representing the cycle, or None if not found.
    """
    n = P['n']
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    # Heuristic: Sort neighbors by the number of available edges to unvisited nodes
    def get_neighbors_sorted(u, current_path_len):
        neighbors = []
        for v in adj[u]:
            if not visited[v]:
                # Count available edges from v to other unvisited nodes
                # This helps avoid dead ends
                avail_count = 0
                for w in adj[v]:
                    if not visited[w] and (min(v,w), max(v,w)) in available_edges_set:
                        avail_count += 1
                # If we are at the last step, we must connect to start
                if current_path_len == n - 1:
                    if (min(u, start), max(u, start)) in available_edges_set:
                         # Only valid if v is start? No, v is unvisited. 
                         # If current_path_len is n-1, only 1 vertex is unvisited.
                         # That vertex must be start? No, start is visited.
                         # So if current_path_len == n-1, the only unvisited vertex must be connected to u.
                         # And then u must connect to start.
                         pass
                neighbors.append((avail_count, v))
        neighbors.sort()
        return [v for _, v in neighbors]

    def dfs(u, depth):
        if depth == n:
            # Check if we can close the cycle
            if (min(u, start), max(u, start)) in available_edges_set:
                return path[:]
            return None
        
        for v in get_neighbors_sorted(u, depth):
            if (min(u, v), max(u, v)) in available_edges_set:
                visited[v] = True
                path.append(v)
                
                result = dfs(v, depth + 1)
                if result:
                    return result
                
                path.pop()
                visited[v] = False
                
    return dfs(start, 1)

def construct():
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    available_edges = set()
    for u, v in P['edges']:
        available_edges.add((min(u, v), max(u, v)))
        
    cycles = []
    
    for i in range(3): # Max possible is 3
        if time.time() > deadline:
            break
            
        # Try to find a cycle
        # We might need to try different start vertices if 0 fails, but 0 is usually fine
        # To be robust, we can try a few starts if the first fails
        found = None
        # Try start 0 first
        found = find_hamiltonian_cycle(available_edges, start=0)
        
        if not found:
             # If start 0 fails, try another start vertex?
             # Or just give up for this iteration?
             # Given the graph symmetry, if a Hamiltonian cycle exists, it likely exists starting from any vertex.
             # But edge-disjointness might block specific paths.
             # Let's try a couple of other starts just in case.
             for s in [1, 2, 3, 4]:
                 if time.time() > deadline:
                     break
                 found = find_hamiltonian_cycle(available_edges, start=s)
                 if found:
                     break
                     
        if found:
            cycles.append(found)
            # Remove edges from available set
            for j in range(len(found)):
                u = found[j]
                v = found[(j + 1) % len(found)]
                edge = (min(u, v), max(u, v))
                if edge in available_edges:
                    available_edges.remove(edge)
        else:
            break
            
    # Ensure types are correct
    # Cycles are lists of ints
    # The function returns a list of lists
    return cycles

# EVOLVE-BLOCK-END
