import time
import random
import numpy as np

def construct():
    """
    Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph.
    Strategy: Start from a base construction, then use local search (edge swaps) to minimize
    the maximum triangle-free induced subgraph size.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 18
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    def is_k4_free(adj):
        # Check if graph is K4-free
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def max_triangle_free_subgraph_size(adj):
        """
        Find the largest set of vertices inducing a triangle-free subgraph.
        Use branch and bound.
        """
        # Precompute: for each vertex, which vertices form triangles with it
        # A set S is triangle-free if no three vertices in S form a triangle
        
        # Use a heuristic: try to find large triangle-free sets
        # Branch and bound: maintain a candidate set and try to add vertices
        
        best = [0]
        
        def is_triangle_free_set(vertices):
            """Check if the given set of vertices induces a triangle-free graph."""
            vlist = list(vertices)
            k = len(vlist)
            for i in range(k):
                for j in range(i+1, k):
                    u, v = vlist[i], vlist[j]
                    if not adj[u][v]:
                        continue
                    for w in vlist:
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            return False
            return True
        
        # Branch and bound search
        # State: current set of vertices (as frozenset for hashing), current size
        # We want to maximize size subject to triangle-free constraint
        
        # Use a greedy + local search approach for speed, then verify
        # Actually, for n=18, we can do a more systematic search
        
        # Key insight: A set is triangle-free iff it's an independent set in the "triangle graph"
        # where two vertices are connected if they participate in a common triangle... 
        # No, that's not quite right. Let me think again.
        
        # A set S is triangle-free if for any u,v in S with edge (u,v), there is no w in S with edges (u,w) and (v,w).
        
        # Branch and bound:
        # - Maintain a partial solution (set of included vertices)
        # - Maintain a set of candidate vertices that can still be added
        # - Upper bound: current_size + len(candidates)
        
        candidates_all = set(range(n))
        
        def bbsolve(current_set, candidates):
            """Branch and bound to find max triangle-free set."""
            if time.time() > deadline - 5.0:
                return 0  # Timeout safeguard
            
            current_size = len(current_set)
            if current_size + len(candidates) <= best[0]:
                return
            
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Try to add a vertex from candidates
            # Heuristic: pick vertex with most edges to current_set (most likely to cause conflict)
            # or pick vertex that's most connected within candidates
            
            # For efficiency, try adding vertices in order of how "safe" they are
            # A vertex v is safe to add if it doesn't create a triangle with any pair in current_set
            
            # Actually, let's just iterate through candidates
            for v in list(candidates):
                # Check if v can be added to current_set without creating a triangle
                can_add = True
                for u in current_set:
                    if adj[u][v]:
                        for w in current_set:
                            if w != u and adj[u][w] and adj[v][w]:
                                can_add = False
                                break
                    if not can_add:
                        break
                
                if can_add:
                    new_set = current_set | {v}
                    new_candidates = candidates - {v}
                    # Filter candidates: remove any that would form a triangle with v and some other in new_set
                    # Actually, we need to be more careful. Let's just recurse.
                    bbsolve(new_set, new_candidates)
                
                # Also try NOT adding v (but this is implicit in the loop)
            
            # Also, we should consider that not all candidates can be added simultaneously
            # The above approach is too slow. Let me use a different strategy.
        
        # Better approach: Use the fact that n=18 is small.
        # Enumerate all subsets? 2^18 = 262144. That's feasible!
        
        best_size = 0
        # Use bit masks for efficiency
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_mask[i] |= (1 << j)
        
        # For each subset, check if it's triangle-free
        # A subset S is triangle-free if for all u,v in S with edge(u,v), no w in S with edges(u,w) and (v,w)
        
        # Precompute for each pair (u,v) that is an edge, the set of vertices w that form a triangle with u and v
        triangle_w = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    ws = []
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            ws.append(w)
                    triangle_w[(u,v)] = ws
                    triangle_w[(v,u)] = ws
        
        # Now, a subset S (as bitmask) is triangle-free iff:
        # For all edges (u,v) in S, no w in S is in triangle_w[(u,v)]
        
        # Iterate through all 2^18 subsets
        full_mask = (1 << n) - 1
        
        # Use a smarter enumeration: only check subsets that could potentially beat best_size
        # We can prune: if the subset size is <= best_size, skip
        
        # Actually, let's just enumerate all subsets and check
        # For speed, use numpy or just loop
        
        # More efficient: for each subset, check triangle-freeness incrementally
        # But 2^18 = 262144, and for each we might check O(n^2) things... that's ~85 million ops, might be too slow in Python
        
        # Let's use a recursive approach with pruning
        # Or, let's use the following: 
        # A set is triangle-free iff it's an independent set in the graph H where
        # H has vertices 0..17 and edge (u,v) in H iff u and v are in a common triangle in G.
        # Wait, that's not right either.
        
        # Let me think differently. 
        # S is triangle-free iff there's no triangle entirely within S.
        # So we need to check: for all triangles (u,v,w) in G, not all of u,v,w are in S.
        
        # Precompute all triangles in G
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            triangles.append((u, v, w))
        
        # Now S is triangle-free iff for all (u,v,w) in triangles, S doesn't contain all three.
        # Equivalent: S is a hitting set complement... no, S must avoid containing any triangle entirely.
        
        # So we want the largest S such that S doesn't contain any triangle as a subset.
        # This is equivalent to: S is an independent set in the 3-uniform hypergraph where hyperedges are triangles.
        
        # For n=18, we can do branch and bound on this.
        
        # Let's do a simple branch and bound:
        # Order vertices, decide include/exclude
        # Prune if current_size + remaining <= best
        
        # Heuristic ordering: vertices in more triangles first (more constrained)
        tri_count = [0] * n
        for (u, v, w) in triangles:
            tri_count[u] += 1
            tri_count[v] += 1
            tri_count[w] += 1
        
        order = sorted(range(n), key=lambda x: -tri_count[x])
        
        # Map from original vertex to position in order
        pos = {v: i for i, v in enumerate(order)}
        
        # For branch and bound, we need to check if adding vertex v creates a triangle
        # with already included vertices
        
        best[0] = 0
        
        def bbsolve2(idx, current_mask, current_size):
            if time.time() > deadline - 3.0:
                return
            
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Prune: even if we add all remaining, can't beat best
            if current_size + (n - idx) <= best[0]:
                return
            
            v = order[idx]
            
            # Option 1: Include v
            # Check if including v creates a triangle with any two already included vertices
            can_include = True
            # For each pair of already included vertices u, w that are both adjacent to v,
            # if u and w are also adjacent, then (u,v,w) is a triangle
            # More efficiently: for each u already included with adj[u][v],
            # check if there's w already included with adj[u][w] and adj[v][w]
            
            for u in order[:idx]:
                if (current_mask >> pos[u]) & 1:
                    if adj[u][v]:
                        for w in order[:idx]:
                            if w != u and (current_mask >> pos[w]) & 1:
                                if adj[u][w] and adj[v][w]:
                                    can_include = False
                                    break
                if not can_include:
                    break
            
            if can_include:
                bbsolve2(idx + 1, current_mask | (1 << pos[v]), current_size + 1)
            
            # Option 2: Exclude v
            bbsolve2(idx + 1, current_mask, current_size)
        
        bbsolve2(0, 0, 0)
        return best[0]
    
    def local_search(adj, max_time):
        """Improve the graph via edge swaps to minimize max triangle-free subgraph."""
        start = time.time()
        current_adj = [row[:] for row in adj]
        current_score = max_triangle_free_subgraph_size(current_adj)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try all edge flips
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > max_time:
                        return current_adj, current_score
                    
                    # Flip edge (i,j)
                    old_val = current_adj[i][j]
                    current_adj[i][j] = not old_val
                    current_adj[j][i] = not old_val
                    
                    # Check K4-free
                    if not is_k4_free(current_adj):
                        current_adj[i][j] = old_val
                        current_adj[j][i] = old_val
                        continue
                    
                    new_score = max_triangle_free_subgraph_size(current_adj)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        current_adj[i][j] = old_val
                        current_adj[j][i] = old_val
        
        return current_adj, current_score
    
    # Start with a random K4-free graph
    # Strategy: generate random graph, remove edges to make it K4-free
    
    def generate_random_k4_free():
        """Generate a random K4-free graph."""
        adj = [[False] * n for _ in range(n)]
        
        # Start with a random graph
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Remove edges to make it K4-free
        # Find all K4s and remove one edge from each
        changed = True
        while changed:
            changed = False
            for a in range(n):
                for b in range(a+1, n):
                    if not adj[a][b]:
                        continue
                    for c in range(b+1, n):
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                # Found K4: a,b,c,d. Remove a random edge.
                                edges_to_remove = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                u, v = random.choice(edges_to_remove)
                                adj[u][v] = False
                                adj[v][u] = False
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        return adj
    
    # Try multiple random starts and local search
    best_adj = None
    best_score = n + 1
    
    num_attempts = 50
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        
        adj = generate_random_k4_free()
        
        # Quick evaluation
        score = max_triangle_free_subgraph_size(adj)
        
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
        
        # Local search with time limit
        remaining = deadline - 20.0 - time.time()
        if remaining > 5.0:
            adj_ls, score_ls = local_search(adj, min(remaining, 10.0))
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls
    
    # Also try the Turan graph as a baseline
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    turan_adj = [[False] * n for _ in range(n)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    turan_adj[a][b] = True
                    turan_adj[b][a] = True
    
    turan_score = max_triangle_free_subgraph_size(turan_adj)
    if turan_score < best_score:
        best_score = turan_score
        best_adj = turan_adj
    
    if best_adj is None:
        best_adj = turan_adj
    
    return get_edges_from_adj(best_adj)

# Test
if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
