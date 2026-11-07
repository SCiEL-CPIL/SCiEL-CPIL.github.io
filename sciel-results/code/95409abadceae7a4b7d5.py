import random
import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    original_edges = set(tuple(sorted(e)) for e in P['edges'])
    adj = {i: set() for i in range(n)}
    for u, v in original_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Function to find a Hamiltonian cycle in a given adjacency list
    # Returns the list of vertices in the cycle or None
    def find_hamiltonian_cycle(adj, n, start_time, deadline):
        # We fix the start vertex to 0 to break rotational symmetry
        # And we fix the second vertex to be the smallest neighbor to break reflection symmetry (optional, but helps)
        # Actually, just fixing start to 0 is enough for uniqueness of the cycle sequence up to rotation.
        # We need to ensure we don't visit vertices out of order in a way that causes redundant searches.
        
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def dfs(u, depth):
            if time.time() > deadline:
                return None
            
            if depth == n:
                if v in adj[u]: # Check if last vertex connects back to start
                    return path[:]
                return None
            
            # Get neighbors
            neighbors = list(adj[u])
            # Heuristic: sort by degree in remaining graph (fewest options first)
            # But calculating degree dynamically is expensive. 
            # Simple heuristic: randomize or sort by index.
            # Let's try sorting by degree in the *original* graph first, or just randomize.
            # Given the graph is regular, degree is constant. 
            # Let's use a randomized order to avoid getting stuck in bad local minima.
            random.shuffle(neighbors)
            
            for v in neighbors:
                if not visited[v]:
                    # Pruning: if this is the last step (depth == n-1), check if v connects to 0
                    if depth == n - 1:
                        if 0 in adj[v]:
                            visited[v] = True
                            path.append(v)
                            result = path[:]
                            path.pop()
                            visited[v] = False
                            return result
                    else:
                        visited[v] = True
                        path.append(v)
                        res = dfs(v, depth + 1)
                        if res:
                            return res
                        path.pop()
                        visited[v] = False
            
            return None
        
        return dfs(0, 1)

    cycles = []
    remaining_edges = set(original_edges)
    
    # We want to find 3 cycles
    for i in range(3):
        cycle_found = None
        # Try multiple random restarts if the first attempt fails
        # Since the graph is small, one attempt might be enough, but let's be safe.
        # Actually, with a good heuristic, it should be fast.
        
        # Build adjacency list from remaining_edges
        adj = {i: set() for i in range(n)}
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check if any vertex has degree < 2, if so, no Hamiltonian cycle exists
        valid = True
        for i in range(n):
            if len(adj[i]) < 2:
                valid = False
                break
        if not valid:
            break
            
        # Attempt to find cycle
        # We might need to try a few times with different random seeds/orders if it fails
        # But for this specific graph, it's likely easy.
        
        cycle = find_hamiltonian_cycle(adj, n, start_time, deadline)
        
        if cycle is None:
            # If failed, we can't proceed
            break
        
        cycles.append(cycle)
        
        # Remove edges from remaining_edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            if edge in remaining_edges:
                remaining_edges.remove(edge)
            else:
                # This shouldn't happen if the cycle is valid
                pass
                
        if time.time() > deadline:
            break

    # Convert cycles to lists of ints (already are)
    # Ensure types are correct
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
        
    return result
# EVOLVE-BLOCK-END
