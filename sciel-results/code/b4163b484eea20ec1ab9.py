"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Precompute edge indices for quick lookup if needed, though we use list of tuples
    # We will represent the graph as a set of edges or a boolean mask.
    # Boolean mask is faster for iteration.
    
    def count_violations(edge_mask):
        """
        Counts the number of paths s-u-v-t such that deg(s) == deg(t).
        """
        deg = [0] * n
        adj = [set() for _ in range(n)]
        
        for i in range(m):
            if edge_mask[i]:
                u, v = all_edges[i]
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
        
        violations = 0
        # Iterate over all edges (u, v) as the middle edge
        for i in range(m):
            if not edge_mask[i]:
                continue
            u, v = all_edges[i]
            
            # Get degrees of neighbors of u (excluding v)
            # We need to check if any neighbor s of u (s!=v) has a neighbor t of v (t!=u) 
            # such that deg[s] == deg[t] AND s != t.
            # Actually, the path is s-u-v-t. s is neighbor of u, t is neighbor of v.
            # Condition: deg[s] == deg[t].
            # Note: s and t must be distinct? The problem says "simple three-edge path".
            # Simple path implies all vertices distinct. So s != t, s != v, t != u, s != u, t != v.
            # Since s is neighbor of u, s != u. t is neighbor of v, t != v.
            # s != v is enforced by "neighbors of u excluding v".
            # t != u is enforced by "neighbors of v excluding u".
            # We must ensure s != t.
            
            # Optimization: Group neighbors of u by degree.
            # deg_count_u[d] = number of neighbors s of u (s!=v) with deg[s]=d
            deg_count_u = {}
            for s in adj[u]:
                if s != v:
                    ds = deg[s]
                    deg_count_u[ds] = deg_count_u.get(ds, 0) + 1
            
            if not deg_count_u:
                continue
                
            # Now for each neighbor t of v (t!=u), check if deg[t] is in deg_count_u
            # If so, we have a potential match.
            # But we must ensure s != t.
            # If deg[t] is in deg_count_u, let k = deg_count_u[deg[t]].
            # If t itself is one of the neighbors s counted in k, we must subtract 1 for that specific pair (s=t).
            # However, the path s-u-v-t requires s != t.
            # If s == t, the path is t-u-v-t, which is not a simple path (it's a cycle of length 3? No, u-v is edge, v-t is edge, t-u is edge. That's a triangle. But the path is length 3 edges: t-u, u-v, v-t. Vertices t, u, v, t. Not simple because start=end? No, simple path means distinct vertices. t appears twice. So s!=t is required).
            
            for t in adj[v]:
                if t == u:
                    continue
                dt = deg[t]
                if dt in deg_count_u:
                    # There are deg_count_u[dt] candidates for s.
                    # If t is one of those candidates (i.e., t is a neighbor of u, t != v, and deg[t] == dt), 
                    # then the pair (s=t, t) is invalid.
                    # Is t a neighbor of u?
                    if t in adj[u] and t != v:
                        # t is a valid candidate for s.
                        # So one of the deg_count_u[dt] matches is s=t, which is invalid.
                        violations += deg_count_u[dt] - 1
                    else:
                        # t is not a neighbor of u (or is v, but t!=u and t in adj[v], if t==v it's a loop which doesn't exist).
                        # So all deg_count_u[dt] matches are valid s != t.
                        violations += deg_count_u[dt]
                        
        return violations

    def get_edge_count(edge_mask):
        return sum(edge_mask)

    def local_search_repair(edge_mask):
        """
        Greedily remove edges to minimize violations to 0.
        """
        current_mask = edge_mask[:]
        current_violations = count_violations(current_mask)
        
        # If already valid, return
        if current_violations == 0:
            return current_mask
        
        max_steps = 200 # Safety limit for repair steps
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            if current_violations == 0:
                break
                
            best_delta = 0
            best_edge_idx = -1
            
            # Try removing each edge
            for i in range(m):
                if current_mask[i]:
                    # Simulate removal
                    old_state = current_mask[i]
                    current_mask[i] = False
                    new_violations = count_violations(current_mask)
                    delta = new_violations - current_violations
                    current_mask[i] = old_state
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_edge_idx = i
            
            if best_edge_idx == -1:
                # No single removal reduces violations. 
                # This can happen if we are in a local minimum where removing any edge increases or keeps violations same.
                # In this case, we might need to add edges? 
                # Or break and let ILS perturb.
                break
            
            # Apply the best removal
            current_mask[best_edge_idx] = False
            current_violations += best_delta
            
        return current_mask

    def local_search_maximize(edge_mask):
        """
        Greedily add edges to maximize edge count while keeping violations 0.
        """
        current_mask = edge_mask[:]
        current_violations = count_violations(current_mask)
        
        if current_violations != 0:
            return current_mask # Only maximize if valid
            
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding each missing edge
            # To make it deterministic and fast, iterate in fixed order or random?
            # Random order helps escape local optima in maximization.
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                if not current_mask[i]:
                    # Simulate addition
                    current_mask[i] = True
                    new_violations = count_violations(current_mask)
                    current_mask[i] = False
                    
                    if new_violations == 0:
                        # Safe to add
                        current_mask[i] = True
                        improved = True
                        # We could break and restart loop to re-shuffle, but 
                        # adding one edge might open up others or close others.
                        # Continuing in same pass is faster.
                        
        return current_mask

    def iterated_local_search():
        # Start with a dense random graph
        # p=0.8 is dense
        current_mask = [random.random() < 0.8 for _ in range(m)]
        
        best_mask = current_mask[:]
        best_score = -1
        
        num_restarts = 1000 # High count, but each restart is fast
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # 1. Repair: Remove edges to make valid
            repaired_mask = local_search_repair(current_mask)
            
            # 2. Maximize: Add edges back
            maximized_mask = local_search_maximize(repaired_mask)
            
            # 3. Evaluate
            score = get_edge_count(maximized_mask)
            
            if score > best_score:
                best_score = score
                best_mask = maximized_mask[:]
                
            # 4. Perturb for next iteration
            # Remove a random subset of edges to escape local optimum
            # Probability of removing an existing edge: 0.1
            # Probability of adding a missing edge: 0.05 (to explore different structures)
            next_mask = best_mask[:]
            for i in range(m):
                if next_mask[i]:
                    if random.random() < 0.1:
                        next_mask[i] = False
                else:
                    if random.random() < 0.05:
                        next_mask[i] = True
                        
            current_mask = next_mask
            
        return best_mask, best_score

    # Run ILS
    best_overall_mask, best_overall_score = iterated_local_search()
    
    # Final check and polish
    # Sometimes ILS might end with an invalid graph if deadline hit during repair?
    # local_search_repair returns the best it could do. 
    # We must ensure the returned graph is VALID.
    final_violations = count_violations(best_overall_mask)
    if final_violations != 0:
        # Fallback: Force validity by aggressive removal
        # Or use a known valid sparse graph
        # A star graph is valid.
        # Let's try to repair the best mask found, even if it takes time.
        repaired = local_search_repair(best_overall_mask)
        if count_violations(repaired) == 0:
            best_overall_mask = repaired
            best_overall_score = get_edge_count(best_overall_mask)
        else:
            # Emergency fallback: Star
            best_overall_mask = [False] * m
            for i in range(1, n):
                # Find edge index for (0, i)
                # all_edges is sorted (i,j) with i<j
                # (0, i) is at index?
                # It's easier to just build the mask from scratch for star
                pass
            
            # Reconstruct star mask
            best_overall_mask = [False] * m
            for idx, (u, v) in enumerate(all_edges):
                if u == 0:
                    best_overall_mask[idx] = True
            
    # Convert mask to list of edges
    best_edges = []
    for i in range(m):
        if best_overall_mask[i]:
            u, v = all_edges[i]
            best_edges.append([u, v])
            
    # Ensure types are correct
    return best_edges

# EVOLVE-BLOCK-END
