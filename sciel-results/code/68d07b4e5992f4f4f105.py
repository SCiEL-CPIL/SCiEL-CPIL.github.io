"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 15}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    
    def is_valid(adj):
        """Check if graph satisfies the constraint."""
        deg = [sum(adj[i]) for i in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                # For edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
                for s in range(n):
                    if s == v or not adj[u][s]:
                        continue
                    for t in range(n):
                        if t == u or not adj[v][t]:
                            continue
                        if s == t:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def violation_count(adj):
        """Count number of violating 3-edge paths."""
        deg = [sum(adj[i]) for i in range(n)]
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == v or not adj[u][s]:
                        continue
                    for t in range(n):
                        if t == u or not adj[v][t]:
                            continue
                        if s == t:
                            continue
                        if deg[s] == deg[t]:
                            count += 1
        return count
    
    def edge_count(adj):
        return sum(sum(adj[i]) for i in range(n)) // 2
    
    # Start with complete graph
    adj = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
    
    # Greedy: try to keep edges, remove only when necessary
    # Actually, let's try a different approach: start sparse and add edges greedily
    
    # Better: start with complete graph, remove edges to fix violations
    # But that might remove too many. Let's try greedy addition.
    
    # Approach: try all subsets of edges in a smart order
    # Start with empty graph, add edges one by one if it doesn't create violations
    # Order edges by some heuristic (e.g., prefer edges that connect vertices with very different degrees)
    
    # Actually, let's try: start with complete graph and iteratively remove edges
    # that are "most responsible" for violations
    
    # Let's try a local search: start with complete graph, remove edges to minimize violations
    # then try to add back edges
    
    # First, let's see how many violations the complete graph has
    # In complete graph, all degrees are n-1=14, so every 3-edge path has equal-degree endpoints
    # So we need to remove many edges.
    
    # Let's try a construction: partition vertices into groups with distinct degrees
    # If we can make degrees as distinct as possible, we minimize violations
    
    # Try: make a graph where degrees are as spread out as possible
    # Use a greedy construction
    
    # Start with empty graph
    adj = [[0]*n for _ in range(n)]
    
    # Try adding edges in order of "benefit"
    # An edge (u,v) is beneficial if it doesn't create new violations
    # But adding an edge changes degrees, so it's complex
    
    # Let's try: enumerate possible degree sequences and find a valid one
    # Or use a simple heuristic: build a graph with a specific structure
    
    # Idea: use a "blow-up" of a small valid graph
    # Or: make a graph where most vertices have unique degrees
    
    # Let's try: create a graph where degree of vertex i is approximately i
    # This is like a "threshold graph" or similar
    
    # Threshold graphs: vertices ordered, each new vertex is either isolated or connected to all previous
    # Let's try a threshold graph construction
    
    # In a threshold graph, if we add vertices in order and each is either universal or isolated,
    # the degrees will be very structured
    
    # Let's try: vertex 0 is isolated, vertex 1 connected to 0, vertex 2 connected to 0,1, etc.
    # No, that gives degrees 0,1,2,...,14 which is impossible as noted.
    
    # Let's try: make a complete bipartite graph K_{a,b} plus some extra edges
    # In K_{a,b}, vertices in part A have degree b, vertices in part B have degree a
    # If a ≠ b, then within a part, all have same degree, but paths of length 3 
    # within a part don't exist (bipartite). Paths of length 3 go A-B-A-B or B-A-B-A
    # So endpoints are in the same part, hence same degree. Violation!
    
    # So pure bipartite doesn't work unless one part is empty.
    
    # What if we add edges within parts? Then degrees change.
    
    # Let's try a different approach: use a random search / local search
    
    # Start with a random dense graph, then fix violations by removing edges
    import random
    random.seed(42)
    
    best_adj = None
    best_count = -1
    
    # Try multiple random restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Start with a random graph with edge probability p
        p = 0.5
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = adj[j][i] = 1
        
        # Local search: try removing edges to fix violations
        for it in range(500):
            if time.time() > deadline - 5:
                break
            if is_valid(adj):
                break
            # Find an edge whose removal reduces violations the most
            vc = violation_count(adj)
            if vc == 0:
                break
            best_removal = None
            best_vc = vc
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        adj[i][j] = adj[j][i] = 0
                        new_vc = violation_count(adj)
                        if new_vc < best_vc:
                            best_vc = new_vc
                            best_removal = (i, j)
                        adj[i][j] = adj[j][i] = 1
            if best_removal is None:
                break
            i, j = best_removal
            adj[i][j] = adj[j][i] = 0
        
        # Try to add edges back
        for it in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        adj[i][j] = adj[j][i] = 1
                        if is_valid(adj):
                            improved = True
                        else:
                            adj[i][j] = adj[j][i] = 0
            if not improved:
                break
        
        ec = edge_count(adj)
        if ec > best_count and is_valid(adj):
            best_count = ec
            best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        # Fallback: star graph (always valid)
        adj = [[0]*n for _ in range(n)]
        for v in range(1, n):
            adj[0][v] = adj[v][0] = 1
        best_adj = adj
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([int(i), int(j)])
    return edges
# EVOLVE-BLOCK-END
