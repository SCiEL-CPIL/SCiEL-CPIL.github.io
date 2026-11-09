import time
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    """
    Constructs a triangle-free graph on n vertices with no independent set of 10.
    Strategy:
    1. Use a fast bitwise independent set checker.
    2. Start with the Andrasfai graph for n=29 (known to be valid).
    3. For n > 29, use a local search (Simulated Annealing style) to add/remove edges to break triangles and independent sets.
    4. Maximize n up to 40.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s safety margin for validation/overhead in evaluator? 
    # Actually, the limit is 150s. Let's use 140s for search to be safe.
    # The evaluator might add overhead. 140s is safe.
    
    def time_up():
        return time.time() > deadline

    def build_adj_from_edges(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def to_edges(adj, n):
        edges = []
        for u in range(n):
            mask = adj[u] & (1 << (u + 1)) # Clear bits <= u
            # Actually, let's just iterate v > u
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def is_triangle_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Check if any w > v is connected to both
                    # Or any w
                    # w must be connected to u and v
                    # adj[u] & adj[v] gives common neighbors
                    # We need to ensure no common neighbor exists
                    if adj[u] & adj[v]:
                        return False
        return True

    def has_independent_set(adj, n, target_size=10):
        """
        Returns True if there is an independent set of size >= target_size.
        Uses branch and bound with bitmasks.
        """
        # Candidate set: all vertices
        # We want to find an independent set.
        # Independent set in G is a clique in complement.
        # But checking independent set directly:
        # A set S is independent if no edges between vertices in S.
        
        def dfs(candidates, size):
            if size >= target_size:
                return True
            if candidates == 0:
                return False
            # Pruning: even if we pick all candidates, can we reach target?
            # Simple bound: size + popcount(candidates) < target_size
            # Python 3.10+ has int.bit_count()
            if size + candidates.bit_count() < target_size:
                return False
            
            # Heuristic: pick a vertex from candidates
            # Choose vertex with highest degree in the induced subgraph of candidates?
            # Or just pick the lowest set bit for speed?
            # Let's pick the vertex with the highest degree in the *original* graph 
            # restricted to candidates? 
            # Actually, for independent set, we want to exclude neighbors.
            # Picking a vertex with high degree in the complement (low degree in G) might be better?
            
            # Let's just pick the lowest set bit to keep it fast, but this can be slow.
            # Better: Pick a vertex v.
            # If we include v, we remove v and all its neighbors from candidates.
            # If we exclude v, we just remove v.
            
            # To prune effectively, we should order vertices.
            # Let's pick v as the lowest set bit.
            v_bit = candidates & (-candidates)
            v = v_bit.bit_length() - 1
            rest = candidates ^ v_bit
            
            # Option 1: Include v
            # New candidates: rest & ~adj[v]
            if dfs(rest & ~adj[v], size + 1):
                return True
            
            # Option 2: Exclude v
            if dfs(rest, size):
                return True
            
            return False
        
        return dfs((1 << n) - 1, 0)

    def andrasfai_adj(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                # Standard Andrasfai: (j-i) % n in {1, 3, ..., n-2} ?
                # The seed used (j-i) % n % 3 == 1.
                # Let's stick to the seed's proven construction for n=29.
                if (j - i) % n % 3 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # 1. Baseline: n=29
    best_n = 0
    best_adj = None
    best_edges = []
    
    adj29 = andrasfai_adj(29)
    if is_triangle_free(adj29, 29) and not has_independent_set(adj29, 29):
        best_n = 29
        best_adj = adj29
        best_edges = to_edges(adj29, 29)
    else:
        # Fallback n=26
        adj26 = andrasfai_adj(26)
        if is_triangle_free(adj26, 26) and not has_independent_set(adj26, 26):
            best_n = 26
            best_adj = adj26
            best_edges = to_edges(adj26, 26)
        else:
            # Should not happen, but just in case
            best_n = 1
            best_adj = [0] * 1
            best_edges = []

    # 2. Search for larger n (30 to 40)
    # We use a local search approach.
    # Start with a random triangle-free graph.
    # Then try to add edges to reduce alpha, while maintaining triangle-free.
    # This is hard because adding edges creates triangles.
    # Alternative: Start with a graph with alpha <= 9 (which is hard to find randomly)
    # or Start with a triangle-free graph and try to "repair" alpha.
    
    # Let's try a different heuristic for local search:
    # Start with empty graph (alpha = n).
    # Add edges randomly, ensuring no triangle.
    # Check alpha. If alpha > 9, try to remove edges? No, removing edges increases alpha.
    # We need to ADD edges to decrease alpha, but adding edges risks triangles.
    
    # Actually, a better strategy:
    # Start with a random regular-ish graph.
    # If it has a triangle, remove an edge from the triangle.
    # If it has an independent set, add an edge within it (if it doesn't create a triangle).
    # This is a local search on the space of triangle-free graphs.
    
    def local_search(n, max_iters=5000):
        # Initialize with a random triangle-free graph
        # Greedy: iterate edges, add if no triangle
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if not (adj[u] & adj[v]): # No common neighbor, so adding (u,v) won't create triangle with any w
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Now adj is triangle-free.
        # Check alpha
        if not has_independent_set(adj, n):
            return adj
        
        # Local search
        current_adj = adj
        for i in range(max_iters):
            if time_up():
                break
            
            # Find an independent set of size 10
            # We need a function to FIND it, not just check existence.
            # Let's implement find_independent_set
            iso_set = find_independent_set(current_adj, n, 10)
            if iso_set is None:
                return current_adj
            
            # Try to add an edge within iso_set
            # An edge (u, v) in iso_set can be added if it doesn't create a triangle.
            # u and v are not adjacent (since they are in independent set).
            # Adding (u, v) creates a triangle if there is w such that (u,w) and (v,w) are edges.
            # i.e., adj[u] & adj[v] != 0.
            
            improved = False
            # Shuffle iso_set to randomize choice
            random.shuffle(iso_set)
            for u in iso_set:
                for v in iso_set:
                    if u >= v:
                        continue
                    if not (current_adj[u] & (1 << v)): # They are not adjacent
                        # Check if adding edge creates triangle
                        if not (current_adj[u] & current_adj[v]):
                            # Safe to add
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try to remove an edge that is part of many triangles? 
                # Or just random edge removal to escape local minimum?
                # If we can't add an edge to break the independent set, 
                # maybe we need to restructure.
                # For now, break if stuck? 
                # Or try a different independent set?
                # Let's just break to save time, as adding edges is the main goal.
                break
        
        return None # Failed to make valid

    def find_independent_set(adj, n, target_size=10):
        """Returns a list of vertices forming an independent set of size target_size, or None."""
        found_set = []
        
        def dfs(candidates, size):
            if size >= target_size:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target_size:
                return False
            
            v_bit = candidates & (-candidates)
            v = v_bit.bit_length() - 1
            rest = candidates ^ v_bit
            
            # Include v
            found_set.append(v)
            if dfs(rest & ~adj[v], size + 1):
                return True
            found_set.pop()
            
            # Exclude v
            if dfs(rest, size):
                return True
            
            return False
        
        if dfs((1 << n) - 1, 0):
            return found_set
        return None

    # Search for n = 30 to 40
    for n in range(30, 41):
        if time_up():
            break
        
        # Try multiple restarts
        success = False
        attempts = 0
        while not success and not time_up() and attempts < 100:
            attempts += 1
            adj_candidate = local_search(n, max_iters=200)
            if adj_candidate is not None:
                # Final validation
                if is_triangle_free(adj_candidate, n) and not has_independent_set(adj_candidate, n):
                    best_n = n
                    best_adj = adj_candidate
                    best_edges = to_edges(adj_candidate, n)
                    success = True
                    break
            
    # Ensure we return the best found
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    result = construct()
    print(f"Best n: {result['n']}")
