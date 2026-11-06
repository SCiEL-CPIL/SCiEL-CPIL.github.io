import itertools
import time

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    vertices = list(range(16))
    
    # Candidate edges: all pairs with Hamming distance 2
    # This graph has 48 edges.
    # We will greedily remove edges to destroy all Q3 subgraphs.
    
    def hamming_dist(u, v):
        return bin(u ^ v).count('1')
        
    # Initial edge set
    initial_edges = set()
    for u in range(16):
        for v in range(u + 1, 16):
            if hamming_dist(u, v) == 2:
                initial_edges.add((u, v))
                
    edges = list(initial_edges)
    edge_set = set(edges)
    
    def get_induced_edges(subset, edge_set):
        """Get edges in edge_set induced by subset"""
        s = set(subset)
        induced = []
        for e in edge_set:
            if e[0] in s and e[1] in s:
                induced.append(e)
        return induced
        
    def is_q3(subset, edge_set):
        """Check if induced subgraph on subset is isomorphic to Q3"""
        induced_edges = get_induced_edges(subset, edge_set)
        if len(induced_edges) != 12:
            return False
            
        # Check degrees
        degrees = {v: 0 for v in subset}
        for u, v in induced_edges:
            degrees[u] += 1
            degrees[v] += 1
            
        # Q3 is 3-regular
        for v in subset:
            if degrees[v] != 3:
                return False
                
        # Check bipartite
        # Build adjacency list
        adj = {v: [] for v in subset}
        for u, v in induced_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Try to color
        color = {}
        for start in subset:
            if start in color:
                continue
            color[start] = 0
            queue = [start]
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor in color:
                        if color[neighbor] == color[curr]:
                            return False
                    else:
                        color[neighbor] = 1 - color[curr]
                        queue.append(neighbor)
                        
        # If we reach here, it is a 3-regular bipartite graph on 8 vertices
        # Which implies it is Q3
        return True

    # List of all 8-vertex subsets
    subsets = list(itertools.combinations(vertices, 8))
    
    # Greedy removal
    # We need to be careful not to loop forever.
    # Max edges 48. We will remove at most 48 edges.
    # Each check is O(12870 * 12). ~150k ops. Very fast.
    
    while time.time() - start_time < timeout:
        found = False
        for subset in subsets:
            if is_q3(subset, edge_set):
                # Find an edge to remove
                # Remove the first edge in the induced subgraph
                induced_edges = get_induced_edges(subset, edge_set)
                if induced_edges:
                    e_to_remove = induced_edges[0]
                    edge_set.remove(e_to_remove)
                    found = True
                    break
        if not found:
            break
            
    return [list(e) for e in edge_set]
