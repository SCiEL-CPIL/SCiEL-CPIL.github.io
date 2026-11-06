"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _has_triangle(adj, n):
    """Check if graph with adjacency matrix adj has a triangle.
    adj is a set of frozensets or a list of sets for speed."""
    # adj is a list of sets
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            # check common neighbors
            for k in adj[i]:
                if k > j and k in adj[j]:
                    return True
    return False

def _count_triangles(adj, n):
    """Count triangles in graph."""
    count = 0
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            common = adj[i] & adj[j]
            count += len(common)
    return count // 3

def _max_independent_set_size(adj, n, limit=10):
    """Branch and bound to check if independent set of size 'limit' exists.
    Returns True if independent set of size >= limit exists."""
    # Use a greedy + backtracking approach
    # We want to know if alpha >= limit
    # This is equivalent to: can we find 'limit' vertices no two adjacent?
    
    # Use bitmasks for n <= 40
    # Represent independent set candidate as bitmask
    # adj_mask[i] is bitmask of neighbors of i
    
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    # We need to find if there's an independent set of size 'limit'
    # This is the complement of: does the complement graph have a clique of size 'limit'?
    # Or directly: branch and bound on independent set
    
    # Simple backtracking with pruning
    # Order vertices by degree (in original graph) descending to prune faster?
    # Actually for independent set, we want to pick vertices.
    # If we pick vertex v, we cannot pick any neighbor of v.
    
    # Let's use a recursive approach with memoization on bitmask of available vertices
    # But 2^40 is too large.
    
    # Better: use the fact that we only care about size 10.
    # We can use a simple backtracking with ordering.
    
    # Heuristic: pick vertex with highest degree first to reduce options quickly?
    # Or lowest degree?
    # Let's try a simple DFS.
    
    # To make it fast, we'll use a greedy upper bound.
    # If the remaining vertices can't form an independent set of size 'need', prune.
    
    # A simple upper bound on independent set size in a set of vertices S:
    # |S| / (avg_degree_in_S + 1) is not tight.
    # Turan's theorem?
    # Let's just use |S| as a weak bound, and rely on the small limit (10).
    
    # Actually, for n=40 and limit=10, a well-ordered backtracking should be fast enough
    # if we prune well.
    
    # Let's use a different approach: 
    # The graph is triangle-free. By Ramsey theory, R(3,10) <= 41 means any graph on 41 vertices
    # has a triangle or independent set of 10.
    # We are checking if our graph has an independent set of 10.
    
    # Let's use a simple recursive search.
    # State: (current_set_size, available_mask)
    # But available_mask is 40 bits, so we can't memoize easily.
    
    # Let's use a simple backtracking with vertex ordering.
    # Order vertices by degree in the original graph.
    
    # For speed, let's use a greedy coloring of the complement graph?
    # No, let's stick to simple backtracking.
    
    # We'll use a list of available vertices.
    # At each step, pick a vertex, add to independent set, remove it and its neighbors.
    
    # To prune: if current_size + len(available) < limit, return False.
    # Also, if current_size >= limit, return True.
    
    # This might be slow if the graph is dense. But triangle-free graphs are relatively sparse.
    
    # Let's implement it.
    
    # Precompute neighbor masks
    # adj_mask is already computed
    
    def dfs(available_mask, current_size):
        if current_size >= limit:
            return True
        if available_mask == 0:
            return False
        # Upper bound: current_size + popcount(available_mask) < limit
        if current_size + bin(available_mask).count('1') < limit:
            return False
        
        # Pick a vertex from available_mask
        # Choose vertex with highest degree in the subgraph induced by available_mask?
        # Or just pick the first one for simplicity.
        # Let's pick the vertex with the highest degree in the original graph to prune faster.
        
        # Find the vertex with the highest degree among available vertices
        # This might be expensive. Let's just pick the lowest index for now.
        # Or pick the one with the most neighbors in available_mask.
        
        # For efficiency, let's pick the vertex with the most neighbors in available_mask.
        # This is the "most constrained" vertex.
        
        best_v = -1
        best_deg = -1
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # Count neighbors in available_mask
            deg = bin(adj_mask[v] & available_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb
        
        if best_v == -1:
            return False
        
        # Option 1: Include best_v in independent set
        # Remove best_v and all its neighbors from available_mask
        new_mask = available_mask & ~(adj_mask[best_v] | (1 << best_v))
        if dfs(new_mask, current_size + 1):
            return True
        
        # Option 2: Exclude best_v
        # Just remove best_v from available_mask
        new_mask = available_mask & ~(1 << best_v)
        if dfs(new_mask, current_size):
            return True
        
        return False
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask, 0)

def construct():
    # Start with the Andrasfai graph on 26 vertices (score 1.0)
    # Then try to extend it to 40 vertices using simulated annealing
    
    # Step 1: Build initial graph
    # We'll start with a random triangle-free graph on 40 vertices
    # Or start with a known construction and add vertices
    
    # Let's try to build a graph on 40 vertices
    n_target = 40
    
    # Initialize with a random graph, then remove triangles
    # Or start with a complete bipartite graph K_20,20 (triangle-free, alpha=20, too large)
    # We need alpha < 10
    
    # Let's use the Andrasfai graph as a seed and try to add vertices
    
    # Actually, let's start from scratch with a random graph and use simulated annealing
    
    random.seed(42)
    
    # Start with a random graph on 40 vertices
    n = n_target
    adj = [set() for _ in range(n)]
    
    # Add random edges
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Remove triangles
    # This is a constraint satisfaction problem
    # Let's use a simple iterative removal
    
    # First, let's try to build a triangle-free graph with small independence number
    
    # Use a known construction: the Clebsch graph (16 vertices, alpha=5)
    # Or the Hoffman-Singleton graph (50 vertices, but that's too many)
    
    # Let's try the Andrasfai graph And(9) on 26 vertices
    # And(9): Z_26 with i ~ j iff (j-i) mod 26 is 1 mod 3
    # This has alpha=9
    
    # Let's start with this and try to add 14 more vertices
    
    # Actually, let's just use simulated annealing on a random graph
    
    # Objective: minimize (number_of_triangles * 100 + max_independent_set_size)
    # But checking max_independent_set_size is expensive
    
    # Let's use a simpler objective: minimize number of triangles, then check alpha
    
    # Simulated annealing
    temperature = 1.0
    cooling_rate = 0.999
    iterations = 5000
    
    def count_triangles_fast(adj, n):
        count = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    count += len(common)
        return count // 3
    
    # Initial triangle count
    tri_count = count_triangles_fast(adj, n)
    
    # We want triangle-free, so tri_count should be 0
    # And alpha < 10
    
    # Let's try a different approach: start with a known triangle-free graph
    # and use edge swaps to reduce alpha
    
    # The Andrasfai graph And(9) has 26 vertices and alpha=9
    # Let's use it as a base
    
    n_base = 26
    adj_base = [set() for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % n_base % 3 == 1:
                adj_base[i].add(j)
                adj_base[j].add(i)
    
    # Now try to add 14 more vertices
    # For each new vertex, connect it to some existing vertices
    # such that no triangle is formed and alpha doesn't increase too much
    
    # This is complex. Let's just use simulated annealing on the full 40-vertex graph
    
    # Restart with a better initialization
    adj = [set() for _ in range(n)]
    
    # Start with a random bipartite graph (triangle-free)
    # Partition vertices into two sets of 20
    left = list(range(20))
    right = list(range(20, 40))
    
    for i in left:
        for j in right:
            if random.random() < 0.5:
                adj[i].add(j)
                adj[j].add(i)
    
    # This is triangle-free but alpha is likely large
    # We need to add edges within parts to reduce alpha, but that creates triangles
    
    # This is tricky. Let's use a different strategy
    
    # Use the fact that R(3,10) <= 41, so a graph on 40 vertices without triangle
    # must have alpha >= 10? No, R(3,10) <= 41 means any graph on 41 vertices has
    # a triangle or independent set of 10. So a graph on 40 vertices might not have
    # either.
    
    # We need to find a triangle-free graph on 40 vertices with alpha < 10
    
    # Let's try the Clebsch graph (16 vertices, alpha=5) and extend it
    
    # Actually, let's just use a simple greedy approach:
    # Start with no edges, add edges one by one, avoiding triangles
    # and trying to keep alpha small
    
    # This is essentially building a triangle-free graph with small independence number
    
    # Let's use a known result: the largest known triangle-free graph with alpha < 10
    # is on 39 vertices (from the survey)
    
    # So let's try to construct a graph on 39 or 40 vertices
    
    # I'll use a simple simulated annealing approach
    
    # Initialize with a random graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Remove triangles by removing edges
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in list(adj[i]):
                if j <= i:
                    continue
                if adj[i] & adj[j]:
                    # Remove edge (i,j)
                    adj[i].remove(j)
                    adj[j].remove(i)
                    changed = True
    
    # Now check alpha
    if _max_independent_set_size(adj, n, 10):
        # Alpha >= 10, need to reduce it
        # Add edges to reduce alpha, but avoid creating triangles
        # This is hard. Let's try edge swaps
        
        # Simulated annealing
        temperature = 1.0
        for it in range(10000):
            # Pick a random non-edge and add it if no triangle is formed
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or j in adj[i]:
                continue
            
            # Check if adding (i,j) creates a triangle
            if adj[i] & adj[j]:
                continue
            
            # Add the edge
            adj[i].add(j)
            adj[j].add(i)
            
            # Check if alpha is still < 10
            if not _max_independent_set_size(adj, n, 10):
                # Good, keep the edge
                temperature *= 0.999
            else:
                # Bad, remove the edge
                adj[i].remove(j)
                adj[j].remove(i)
                temperature *= 0.999
        
        # Final check
        if not _max_independent_set_size(adj, n, 10):
            # Success!
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if j > i:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # If we couldn't get a valid graph on 40 vertices, try 39
    n = 39
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.3:
                adj[i].add(j)
                adj[j].add(i)
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in list(adj[i]):
                if j <= i:
                    continue
                if adj[i] & adj[j]:
                    adj[i].remove(j)
                    adj[j].remove(i)
                    changed = True
    
    if not _max_independent_set_size(adj, n, 10):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Fallback: return the Andrasfai graph on 26 vertices
    n = 26
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
