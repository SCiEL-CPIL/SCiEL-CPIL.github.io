import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit for safety
    
    n = 12
    target_degree = 3
    
    # Generate all 3-regular graphs on n vertices
    # We use a backtracking approach.
    # To reduce symmetry, we can fix the edges of vertex 0.
    # Vertex 0 connects to 3 vertices. WLOG, let them be 1, 2, 3.
    # Then vertex 1 needs 2 more edges.
    
    # This might be complex to implement robustly from scratch in short time.
    # Alternative: Use a known list or a simpler search.
    # Given the small size, we can also just try to find a specific structure.
    
    # Let's try to find a graph with 1 distinct cycle length first.
    # If not found, accept 2.
    
    # Hardcoded check for specific candidate structures?
    # Or just run a limited search.
    
    # Let's implement a simple search for 3-regular graphs.
    
    def generate_3_regular():
        # Returns a generator of edge lists
        # We represent edges as a set of frozensets or sorted tuples.
        # Degrees array.
        
        # Symmetry breaking:
        # Vertex 0 connects to 1, 2, 3.
        edges = [(0,1), (0,2), (0,3)]
        deg = [0]*n
        for u,v in edges:
            deg[u]+=1
            deg[v]+=1
            
        # Now vertex 1 has deg 1 (needs 2).
        # Vertex 2 has deg 1 (needs 2).
        # Vertex 3 has deg 1 (needs 2).
        # Vertex 4..11 have deg 0 (need 3).
        
        # This backtracking is tricky to get right quickly.
        # Let's try a different tactic:
        # We know the current best is 2 (from the hexagon matching likely).
        # Can we get 1?
        # If we can't prove 1 is impossible, we should search for it.
        
        # Let's try to construct a specific graph known to have small cycle variety.
        # The "Petersen" is girth 5.
        # What about the "Möbius-Kantor" graph? (8 vertices, 3-reg, girth 4).
        # Cycles in Möbius-Kantor: 4, 6, 8?
        
        # Let's just return the current best if search is too complex, 
        # but try a quick random search for 3-regular graphs.
        
        import random
        best_val = float('inf')
        best_edges = None
        
        # Random search for 3-regular graphs
        # Use configuration model + fix duplicates/self-loops?
        # Or just sample from known small graphs?
        
        # Let's try to verify if 1 is possible.
        # If we find a graph with distinct lengths 1, we return it.
        
        # Actually, the problem allows disconnected graphs.
        # Can we have a component that is a single 4-cycle?
        # No, min degree 3.
        
        # Let's assume the minimum is 2 and try to find a graph with 2 distinct lengths 
        # that is robust, or try to find 1.
        
        # Given the constraints, I will provide a program that searches for the optimal graph 
        # using a limited backtracking or random restarts.
        
        pass

    # Since I cannot run code here, I will provide a robust heuristic search.
    # The current program score is 1.0 (value 2).
    # If I find a graph with value 1, score becomes 2.0.
    
    # Let's try to construct a graph with only 4-cycles.
    # We established it's hard.
    
    # Let's try to construct a graph with only 4-cycles and 6-cycles? No, we want min distinct.
    
    # I will write a program that tries to find a 3-regular graph on 12 vertices 
    # with the minimum number of distinct cycle lengths.
    
    # For the sake of the output, I will provide a solution that 
    # returns a known good graph (value 2) if search fails, 
    # but includes a search for value 1.
    
    # Actually, looking at the "INSPIRATIONS" and "LESSONS", there are none.
    # The current program is a specific construction.
    
    # Let's try the **Petersen graph** plus 2 vertices? No, 12 vertices.
    
    # I will implement a random search for 3-regular graphs.
    
    def random_3_regular_graph(n):
        # Configuration model
        # 3 stubs per vertex
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        while True:
            random.shuffle(stubs)
            edges = []
            valid = True
            for i in range(0, n*3, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.append(tuple(sorted((u, v))))
            
            if valid:
                return [list(e) for e in edges]
                
    def get_cycle_lengths(edges):
        # Build adjacency list
        adj = {i: [] for i in range(12)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        lengths = set()
        
        # DFS to find cycles
        # Since N=12, we can do a simple DFS from each node.
        # To avoid duplicates, we only consider cycles where the start node is the smallest in the cycle.
        
        def dfs(start, curr, path, visited):
            if len(path) > 1 and curr == start:
                lengths.add(len(path))
                return
            for neighbor in adj[curr]:
                if neighbor == start and len(path) > 1:
                    lengths.add(len(path) + 1)
                elif neighbor not in visited and neighbor > start:
                    visited.add(neighbor)
                    dfs(start, neighbor, path + [neighbor], visited)
                    visited.remove(neighbor)
                    
        for i in range(12):
            dfs(i, i, [i], {i})
            
        return lengths

    best_val = float('inf')
    best_edges = None
    
    # Try random search
    for _ in range(5000):
        if time.time() > deadline:
            break
        edges = random_3_regular_graph(12)
        if not edges:
            continue
        # Check min degree (should be 3 by construction)
        # Check triangle-free
        # Check cycle lengths
        lengths = get_cycle_lengths(edges)
        if not lengths:
            continue
        val = len(lengths)
        if val < best_val:
            best_val = val
            best_edges = edges
            if best_val == 1:
                break
                
    if best_edges:
        return best_edges
    else:
        # Fallback to current program's graph
        es=set()
        for i in range(6):
            for u,v in [(i,(i+1)%6),(i+6,(i+1)%6+6),(i,i+6)]:es.add(tuple(sorted((u,v))))
        return [list(e) for e in sorted(es)]

# EVOLVE-BLOCK-END
