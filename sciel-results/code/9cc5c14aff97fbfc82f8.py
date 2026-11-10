import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n = 16
    # Pairs of vertices (u, v) with u < v
    pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    num_pairs = len(pairs)
    
    # Precompute adjacency masks for pairs
    # adj_mask[i] is a bitmask of vertices connected to i
    # We will maintain a list of adj masks
    
    best_edges_count = 0
    best_edge_list = []
    
    # We will use a backtracking search.
    # State:
    # adj: list of n integers, adj[i] is bitmask of neighbors of i
    # edges_count: current number of edges
    # pair_index: current pair being considered
    
    # To make it fast, we use a greedy heuristic ordering of pairs.
    # However, for exactness and maximizing edges, we need to search.
    # A pure exhaustive search over 120 pairs is 2^120, which is impossible.
    # We must use strong pruning.
    
    # Pruning 1: Triangle-free. If we add (u, v), we must ensure no common neighbor.
    # Pruning 2: Not 3-colorable. If the current graph is 3-colorable, we might still add edges to make it 4-chromatic.
    # But if we are near the end and it's 3-colorable, we can prune if we know we can't add enough edges?
    # Actually, we want to maximize edges.
    
    # Alternative Strategy: 
    # Generate random dense triangle-free graphs and check chromatic number.
    # If chromatic number >= 4, record it.
    # Use local search to improve edge count.
    
    # Let's try a hybrid: 
    # 1. Start with a random maximal triangle-free graph (Maximal means no edge can be added without creating a triangle).
    # 2. Check if it's 3-colorable.
    # 3. If not, it's a candidate.
    # 4. Try to add edges by removing some existing edges and adding different ones (swap moves).
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    # Check if u and v share a common neighbor
                    common = adj[u] & adj[v]
                    if common != 0:
                        return False
        return True

    def is_3_colorable(adj):
        # DSATUR heuristic for 3-colorability
        # Returns True if 3-colorable, False otherwise
        
        # If n is small, exact backtracking is fine.
        # Let's use a simple backtracking with bitsets.
        
        # colors: list of -1 or 0,1,2
        # We can optimize by ordering vertices by degree descending.
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Determine available colors
            # If neighbor has color c, c is not available
            used_colors = 0
            for u in range(n):
                if (adj[v] >> u) & 1:
                    if colors[u] != -1:
                        used_colors |= (1 << colors[u])
            
            # Try colors 0, 1, 2
            for c in range(3):
                if not (used_colors & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        # Time check inside backtrack? No, keep it fast.
        # If it takes too long, assume 3-colorable (safe to prune? No, if it's 4-chromatic, we want to keep it. 
        # If we assume 3-colorable and it's actually 4, we might discard a good graph.
        # So we need to be accurate.
        # For n=16, backtracking is usually very fast unless it's hard.
        
        return backtrack(0)

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def make_maximal_triangle_free(adj):
        # Add any edge that doesn't create a triangle
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1):
                        # Check if adding (u,v) creates a triangle
                        common = adj[u] & adj[v]
                        if common == 0:
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    def random_maximal_tf():
        # Start with empty graph
        adj = [0] * n
        # Randomly add edges, ensuring triangle-free
        # To make it dense, we should add edges greedily?
        # Random order:
        pairs_shuffled = pairs[:]
        random.shuffle(pairs_shuffled)
        for u, v in pairs_shuffled:
            if not ((adj[u] >> v) & 1):
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Now make it maximal
        adj = make_maximal_triangle_free(adj)
        return adj

    # Local Search
    def local_search(adj):
        # Try to increase edge count by removing 1 edge and adding 2? Or swapping?
        # A common move: Remove edge (u,v), add edges (u,w) and (v,x) if valid.
        # Or simpler: Try to add an edge by removing a conflicting edge.
        
        current_edges = sum(bin(a).count('1') for a in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        # Iterate for a fixed number of steps or until no improvement
        steps = 0
        while steps < 500:
            if time.time() > deadline:
                break
            
            improved = False
            # Pick a random edge to remove
            if current_edges == 0: break
            edge_to_remove = None
            # Find a random edge
            # To speed up, keep a list of edges? Or just pick random u,v and check.
            # Let's pick random u, v.
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or not ((adj[u] >> v) & 1):
                steps += 1
                continue
            
            # Remove (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges -= 1
            
            # Try to add new edges
            # Candidates for u: non-neighbors w such that u-w is triangle-free
            # Candidates for v: non-neighbors x such that v-x is triangle-free
            # We want to add at least 2 edges to improve, or 1 edge if we removed 1 (net 0, but maybe opens up more?)
            # Let's try to add 2 edges.
            
            # Get potential neighbors for u
            pot_u = []
            for w in range(n):
                if w != u and w != v and not ((adj[u] >> w) & 1):
                    # Check triangle-free with current adj
                    if (adj[u] & adj[w]) == 0:
                        pot_u.append(w)
                        
            pot_v = []
            for x in range(n):
                if x != v and x != u and not ((adj[v] >> x) & 1):
                    if (adj[v] & adj[x]) == 0:
                        pot_v.append(x)
                        
            if not pot_u or not pot_v:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                steps += 1
                continue
                
            # Try combinations of one from pot_u and one from pot_v
            # To limit search, pick a few random candidates
            random.shuffle(pot_u)
            random.shuffle(pot_v)
            
            found_improvement = False
            for w in pot_u[:5]:
                for x in pot_v[:5]:
                    if w == x: continue
                    # Check if adding (u,w) and (v,x) is valid
                    # 1. (u,w) triangle-free
                    # 2. (v,x) triangle-free
                    # 3. (u,w) and (v,x) don't create triangles with each other?
                    #    Actually, we just need the final graph to be triangle-free.
                    
                    # Apply
                    adj[u] |= (1 << w)
                    adj[w] |= (1 << u)
                    adj[v] |= (1 << x)
                    adj[x] |= (1 << v)
                    
                    new_edges = current_edges + 2
                    # Check triangle-free globally? 
                    # We checked locally for u and v. 
                    # But what if w and x are connected? Or share neighbors?
                    # The local check `(adj[u] & adj[w]) == 0` was done BEFORE adding (u,w).
                    # After adding (u,w), adj[u] and adj[w] change.
                    # We must verify the whole graph or at least the affected parts.
                    
                    # For safety and speed, let's just check if the whole graph is triangle-free.
                    # n=16 is small.
                    if is_triangle_free(adj):
                        # Check 3-colorability
                        if not is_3_colorable(adj):
                            if new_edges > best_edges:
                                best_edges = new_edges
                                best_adj = adj[:]
                                improved = True
                                found_improvement = True
                                break
                    # Rollback
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    adj[v] &= ~(1 << x)
                    adj[x] &= ~(1 << v)
                    
                if found_improvement:
                    break
            
            if not improved:
                # Rollback the initial removal
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
            else:
                # Keep the improvement
                current_edges = new_edges
                
            steps += 1
            
        return best_adj, best_edges

    # Main Loop
    num_restarts = 10000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        adj = random_maximal_tf()
        
        # Check if it's 4-chromatic
        if not is_3_colorable(adj):
            # It's a valid candidate
            edge_count = sum(bin(a).count('1') for a in adj) // 2
            if edge_count > best_edges_count:
                best_edges_count = edge_count
                best_edge_list = get_edge_list(adj)
                
            # Try to improve
            adj, improved_count = local_search(adj)
            if improved_count > best_edges_count:
                best_edges_count = improved_count
                best_edge_list = get_edge_list(adj)
                
        # If we have a very high edge count, we might stop early?
        # Max possible edges in triangle-free graph on 16 vertices is floor(16^2/4) = 64.
        # But we need 4-chromatic.
        # If we reach 50+, it's good.
        
    if not best_edge_list:
        # Fallback: Grötzsch graph (11 vertices, 20 edges) + 5 isolated vertices?
        # No, isolated vertices don't help chromatic number.
        # We need a 4-chromatic graph.
        # Grötzsch graph is 4-chromatic.
        # Let's return Grötzsch graph on 0..10, and leave 11..15 isolated.
        # This is valid (simple graph on 16 vertices).
        # But edge count is low (20).
        # Better fallback: Try to find a small 4-chromatic graph quickly.
        # For now, return Grötzsch.
        
        groetzsch_edges = []
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        for u, w in cycle:
            groetzsch_edges.append([u, w])
            groetzsch_edges.append([u, 5 + w])
            groetzsch_edges.append([w, 5 + u])
        for u in range(5):
            groetzsch_edges.append([5 + u, 10])
        return groetzsch_edges

    return best_edge_list

# EVOLVE-BLOCK-END
