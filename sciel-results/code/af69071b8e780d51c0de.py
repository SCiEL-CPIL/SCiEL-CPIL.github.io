"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 18
    all_vertices = list(range(n))
    
    def is_k4_free(adj):
        """Check if graph with adjacency matrix adj is K4-free."""
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b]:
                    for c in range(b+1, n):
                        if adj[a][c] and adj[b][c]:
                            for d in range(c+1, n):
                                if adj[a][d] and adj[b][d] and adj[c][d]:
                                    return False
        return True
    
    def max_triangle_free_induced_size(adj):
        """Find the maximum number of vertices inducing a triangle-free subgraph."""
        # For each subset, check if it's triangle-free and count size
        # This is exponential, so we use a smarter approach
        # A set of vertices is triangle-free if no three form a triangle
        # We can check: for each vertex, its neighbors within the set must form an independent set
        
        best = 0
        # Try all subsets of size 10-18 (since we want to minimize, we expect small values)
        # Actually, let's use a different approach: 
        # A set S is triangle-free iff for every vertex v in S, the neighbors of v in S form an independent set
        # Equivalently, S contains no triangle
        
        # Use branch and bound: try to find a triangle-free set of size k
        # Start from large k and decrease
        for k in range(n, 0, -1):
            found = False
            # Check if there exists a triangle-free subset of size k
            # We can use: a set is triangle-free iff it's a 2-colorable set (no triangle)
            # Actually, let's just check all subsets of size k using a smarter method
            
            # For each subset of size k, check if it contains a triangle
            # This is C(18,k) which for k=9 is 48620, manageable
            if k <= 10:
                for subset in combinations(range(n), k):
                    subset_set = set(subset)
                    # Check if subset is triangle-free
                    triangle_free = True
                    for u, v, w in combinations(subset, 3):
                        if adj[u][v] and adj[v][w] and adj[u][w]:
                            triangle_free = False
                            break
                    if triangle_free:
                        return k
            else:
                # For k > 10, use complement: if no triangle-free set of size k,
                # then every set of size k contains a triangle
                # This is hard to check directly, so we use a heuristic
                # Actually, for k >= 11, by Ramsey R(3,3)=6, any 6 vertices contain a triangle or independent set
                # But we need triangle-free, not independent
                # Let's just check a sample of subsets
                import random as _rnd
                samples = 1000
                for _ in range(samples):
                    subset = set(_rnd.sample(range(n), k))
                    triangle_free = True
                    for u, v, w in combinations(subset, 3):
                        if adj[u][v] and adj[v][w] and adj[u][w]:
                            triangle_free = False
                            break
                    if triangle_free:
                        return k
                # If no triangle-free set of size k found in samples, try k-1
                continue
        
        return 0  # Should not reach here
    
    def max_triangle_free_induced_size_v2(adj):
        """More efficient: find max triangle-free induced subgraph size."""
        # A set S is triangle-free iff for each v in S, N_S(v) is an independent set
        # Use: S is triangle-free iff S contains no K3
        
        # We'll use a simple approach: for each candidate size from n down to 1,
        # check if there's a triangle-free subset of that size
        # For efficiency, use the fact that a triangle-free graph has at most floor(n^2/4) edges (Mantel's theorem)
        
        best = 1
        # Check sizes from 9 down (we expect the answer to be around 9-11 for n=18)
        for k in range(min(n, 12), 0, -1):
            # Use random sampling to check if a triangle-free subset of size k exists
            if k <= 8:
                # Exhaustive check for small k
                for subset in combinations(range(n), k):
                    s = set(subset)
                    tf = True
                    for u, v, w in combinations(subset, 3):
                        if adj[u][v] and adj[v][w] and adj[u][w]:
                            tf = False
                            break
                    if tf:
                        return k
            else:
                # Random sampling
                for _ in range(2000):
                    subset = set(random.sample(range(n), k))
                    tf = True
                    for u, v, w in combinations(subset, 3):
                        if adj[u][v] and adj[v][w] and adj[u][w]:
                            tf = False
                            break
                    if tf:
                        return k
                # If not found in samples, assume no triangle-free set of this size
                # and continue to smaller k
        return 1
    
    def score_graph(adj):
        """Compute the score (largest triangle-free induced subgraph size)."""
        return max_triangle_free_induced_size_v2(adj)
    
    def random_k4_free_graph():
        """Generate a random K4-free graph."""
        # Start with empty graph and add edges, ensuring no K4 is formed
        adj = [[0]*n for _ in range(n)]
        edges_to_add = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_add)
        
        for u, v in edges_to_add:
            # Check if adding edge (u,v) creates a K4
            # A K4 containing (u,v) needs two other vertices a,b such that
            # all edges among {u,v,a,b} exist
            can_add = True
            for a in range(n):
                if a == u or a == v:
                    continue
                if adj[u][a] and adj[v][a]:
                    for b in range(a+1, n):
                        if b == u or b == v:
                            continue
                        if adj[u][b] and adj[v][b] and adj[a][b]:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                adj[u][v] = 1
                adj[v][u] = 1
        
        return adj
    
    def hill_climbing(adj, max_iter=5000):
        """Improve graph via local search."""
        current_score = score_graph(adj)
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Try flipping a random edge
            u, v = random.sample(range(n), 2)
            if u > v:
                u, v = v, u
            
            new_adj = [row[:] for row in adj]
            new_adj[u][v] = 1 - new_adj[u][v]
            new_adj[v][u] = new_adj[u][v]
            
            if not is_k4_free(new_adj):
                continue
            
            new_score = score_graph(new_adj)
            if new_score < current_score:
                adj = new_adj
                current_score = new_score
            elif new_score == current_score and random.random() < 0.3:
                # Accept equal score moves to escape local optima
                adj = new_adj
        
        return adj, current_score
    
    # Start with complete tripartite K6,6,6
    adj = [[0]*n for _ in range(n)]
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = 1
                    adj[b][a] = 1
    
    best_adj = [row[:] for row in adj]
    best_score = score_graph(best_adj)
    
    # Try multiple random restarts and hill climbing
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate random K4-free graph
        adj = random_k4_free_graph()
        adj, score = hill_climbing(adj, max_iter=2000)
        
        if score < best_score:
            best_adj = [row[:] for row in adj]
            best_score = score
    
    # Also try improving the initial tripartite graph
    adj = [[0]*n for _ in range(n)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = 1
                    adj[b][a] = 1
    adj, score = hill_climbing(adj, max_iter=5000)
    if score < best_score:
        best_adj = [row[:] for row in adj]
        best_score = score
    
    # Convert adjacency matrix to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
