import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def get_idx(i, j, n):
        # Return the index of edge (i, j) in lexicographic order for K_n
        # Edges are (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Index for (i, j) where i < j:
        # Number of edges starting with 0..i-1: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        # Offset within row i: j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adj_masks(colors, n):
        # Build adjacency masks for each color
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def count_violations(adj, n):
        # Count the number of C4s (violations) in the coloring
        # A C4 exists if two vertices u, w have >1 common neighbor in color c
        v = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                for w in range(u + 1, n):
                    common = mask_u & adj[c][w]
                    cnt = common.bit_count()
                    if cnt > 1:
                        v += cnt - 1
        return v

    def find_violating_edges(adj, n):
        """
        Returns a set of edges (i, j) that are part of at least one C4.
        Specifically, an edge (i, j) with color c is 'conflict' if there exists 
        a color c' (could be c) and vertices u, v such that u-v is an edge of color c' 
        and u, v share 2 neighbors in color c'? No, that's not right.
        
        Let's refine: A C4 is a set of 4 vertices {a,b,c,d} and 4 edges.
        If we have a violation in color c, it means there are vertices u, w 
        with common neighbors v, x in color c. The C4 is u-v-w-x-u.
        The edges involved are (u,v), (v,w), (w,x), (x,u).
        We want to recolor edges to break these cycles.
        
        For efficiency, we can just return a list of candidate edges to try.
        Any edge in a C4 is a candidate.
        """
        candidates = set()
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                for w in range(u + 1, n):
                    common = mask_u & adj[c][w]
                    if common.bit_count() > 1:
                        # There is a C4 involving u and w in color c
                        # The neighbors in common are the other two vertices of the C4
                        # Let's just add all edges connected to u and w in color c?
                        # Actually, the edges (u,v) and (w,v) for any v in common are part of the C4.
                        # So we can add (u, v) and (w, v) for all v in common.
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            v = lsb.bit_length() - 1
                            mask ^= lsb
                            # Add edge (u, v)
                            candidates.add((min(u, v), max(u, v)))
                            # Add edge (w, v)
                            candidates.add((min(w, v), max(w, v)))
        return candidates

    def solve_sa_conflict_driven(n, max_iters, start_temp, cooling, initial_colors=None):
        """
        Simulated Annealing with conflict-driven move selection.
        """
        num_edges = n * (n - 1) // 2
        
        if initial_colors is not None:
            colors = initial_colors[:]
        else:
            colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        adj = build_adj_masks(colors, n)
        current_violations = count_violations(adj, n)
        
        if current_violations == 0:
            return True, colors

        temp = start_temp
        best_violations = current_violations
        best_colors = colors[:]
        
        # Precompute edge endpoints for faster lookup
        edge_endpoints = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_endpoints.append((i, j))
                
        # Map edge tuple to index
        edge_to_idx = {}
        for idx, (i, j) in enumerate(edge_endpoints):
            edge_to_idx[(i, j)] = idx

        stagnant_counter = 0
        max_stagnant = 500
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            if current_violations == 0:
                return True, colors

            # Select an edge to recolor
            # 1. Find conflicting edges
            conflict_edges = find_violating_edges(adj, n)
            
            if conflict_edges:
                # Pick a random edge from the conflict set
                # Convert set to list for random selection
                # Optimization: If the set is large, sampling is O(1) if we use a list
                # But creating the set is O(V^2). For n=26, this is small.
                edge_tuple = random.choice(list(conflict_edges))
            else:
                # No conflicts found? Should be impossible if violations > 0
                # Fallback to random edge
                edge_tuple = random.choice(edge_endpoints)
                
            idx = edge_to_idx[edge_tuple]
            i, j = edge_tuple
            
            old_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue

            # Calculate delta using incremental updates
            delta = 0
            
            # Remove from old_color
            c = old_color
            for w in range(n):
                if w == i or w == j:
                    continue
                
                # Pair (i, w)
                old_common = (adj[c][i] & adj[c][w]).bit_count()
                if (adj[c][w] >> j) & 1:
                    new_common = old_common - 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
                
                # Pair (j, w)
                old_common = (adj[c][j] & adj[c][w]).bit_count()
                if (adj[c][w] >> i) & 1:
                    new_common = old_common - 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
            
            # Update masks for removal
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            
            # Add to new_color
            c = new_color
            for w in range(n):
                if w == i or w == j:
                    continue
                
                # Pair (i, w)
                old_common = (adj[c][i] & adj[c][w]).bit_count()
                if (adj[c][w] >> j) & 1:
                    new_common = old_common + 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
                
                # Pair (j, w)
                old_common = (adj[c][j] & adj[c][w]).bit_count()
                if (adj[c][w] >> i) & 1:
                    new_common = old_common + 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
            
            # Update masks for addition
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
            new_violations = current_violations + delta
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                colors[idx] = new_color
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    stagnant_counter = 0
                else:
                    stagnant_counter += 1
            else:
                # Revert
                adj[new_color][i] &= ~(1 << j)
                adj[new_color][j] &= ~(1 << i)
                adj[old_color][i] |= (1 << j)
                adj[old_color][j] |= (1 << i)
                stagnant_counter += 1
                
            temp *= cooling
            
            # If stuck, restart with best colors found so far
            if stagnant_counter > max_stagnant:
                # Reset to best
                colors = best_colors[:]
                adj = build_adj_masks(colors, n)
                current_violations = best_violations
                temp = start_temp
                stagnant_counter = 0

        if current_violations == 0:
            return True, colors
        return False, best_colors

    # Strategy:
    # 1. Try to find a valid coloring for n=26.
    # 2. If found, try to extend to n=27.
    # 3. If n=27 fails, return n=26.
    
    best_n = 0
    best_colors = []
    
    # Try n=26 first, as it's the known bound and most likely target for max score improvement
    # We will use multiple restarts with conflict-driven SA
    
    target_ns = [26]
    
    for n in target_ns:
        if time.time() > deadline - 10.0:
            break
            
        success = False
        # Try multiple restarts
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
                
            max_iters = 50000
            start_temp = 2.0
            cooling = 0.9999
            
            res_success, res_colors = solve_sa_conflict_driven(n, max_iters, start_temp, cooling)
            
            if res_success:
                best_n = n
                best_colors = res_colors
                success = True
                break
        
        if success:
            # If we found n=26, try to extend to 27
            if n == 26 and time.time() < deadline - 30.0:
                # Extension strategy:
                # Add vertex 26.
                # Initialize new edges randomly or based on existing structure.
                # Then run SA on n=27.
                n_ext = 27
                num_edges_ext = n_ext * (n_ext - 1) // 2
                colors_ext = [0] * num_edges_ext
                
                # Copy existing colors
                for i in range(26):
                    for j in range(i + 1, 26):
                        idx_old = get_idx(i, j, 26)
                        idx_new = get_idx(i, j, n_ext)
                        colors_ext[idx_new] = best_colors[idx_old]
                
                # Initialize new edges (26, j) randomly
                for j in range(26):
                    i = 26
                    idx_new = get_idx(j, i, n_ext) # get_idx expects i<j
                    colors_ext[idx_new] = random.randint(0, 4)
                    
                # Run SA on n=27
                res_success_ext, res_colors_ext = solve_sa_conflict_driven(n_ext, 50000, 2.0, 0.9999, initial_colors=colors_ext)
                if res_success_ext:
                    best_n = n_ext
                    best_colors = res_colors_ext
                    break # Stop, we found a larger n
            
            # If we didn't extend, we keep n=26
            break # We found n=26, no need to try other base n's if we are targeting 26
    
    # Fallback if n=26 wasn't found (should not happen with enough time, but just in case)
    if best_n == 0:
        # Try smaller n to ensure we return something valid
        for n in [25, 24, 20, 15]:
            if time.time() > deadline - 5.0:
                break
            res_success, res_colors = solve_sa_conflict_driven(n, 20000, 2.0, 0.9999)
            if res_success:
                best_n = n
                best_colors = res_colors
                break
        if best_n == 0:
            # Absolute fallback: n=1 is always valid
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Result: n={res['n']}, edges={len(res['colors'])}")
