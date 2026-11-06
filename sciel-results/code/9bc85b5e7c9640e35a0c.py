import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 11
    best_colors = None
    
    # Fallback solution: n=11 (known valid)
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Try to find larger n
    # We know R_5(C4) > 11. Lazebnik/Woldar gives 26. 
    # Local search might find 12, 13, ... 
    # Let's try to improve upon 11.
    
    max_n_to_try = 20 # We won't brute force up to 26 with simple local search in 110s likely, but let's try up to a reasonable limit
    # Actually, let's be ambitious but safe. 
    # A valid coloring for n=12 is a good target.
    
    for n in range(12, 16): # Try 12, 13, 14, 15
        if time.time() > deadline - 10.0:
            break
            
        # Initialize random coloring
        # Number of edges is n*(n-1)/2
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Helper to get color index for (i, j)
        def get_idx(i, j):
            if i > j: i, j = j, i
            # Lexicographic order: 
            # (0,1), (0,2)...(0,n-1), (1,2)...
            # Index = i*(n-1) - i*(i-1)/2 + (j-i-1) ? 
            # Let's use a precomputed map or formula
            # Formula for (i, j) with 0 <= i < j < n:
            # idx = i*(n-1) - i*(i+1)//2 + j - i - 1
            # Check: n=3
            # (0,1): 0*2 - 0 + 1-0-1 = 0. Correct.
            # (0,2): 0*2 - 0 + 2-0-1 = 1. Correct.
            # (1,2): 1*2 - 1*2//2 + 2-1-1 = 2 - 1 + 0 = 1? No.
            # Let's just build a lookup table for speed and correctness.
            pass
        
        # Precompute edge indices for speed
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def get_color(i, j):
            if i > j: i, j = j, i
            return colors[edge_map[(i, j)]]
            
        def set_color(i, j, c):
            if i > j: i, j = j, i
            colors[edge_map[(i, j)]] = c

        # Function to count C4s
        # A C4 is a set of 4 distinct vertices {a,b,c,d} such that edges (a,b), (b,c), (c,d), (d,a) are all same color.
        # Or more simply: two vertices u, v have two common neighbors x, y such that (u,x), (x,v), (u,y), (y,v) are all same color.
        # Actually, the condition "No colour class may contain a four-cycle" means:
        # For any color c, the subgraph induced by edges of color c must be C4-free.
        # A C4 in color c exists if there are distinct vertices a,b,c,d such that 
        # (a,b), (b,c), (c,d), (d,a) are all color c.
        # This is equivalent to: for any color c, and any pair of vertices (u, v), 
        # the number of common neighbors w such that (u,w) and (w,v) are color c must not form a cycle?
        # No, just checking for any 4-cycle is O(n^4) which is too slow for n=15 in Python inside a loop.
        
        # Better constraint check:
        # For a fixed color c, let G_c be the graph.
        # G_c has no C4 if and only if for any pair of vertices u, v, 
        # they do not share two common neighbors in G_c? 
        # Wait, if u and v share two common neighbors x and y in G_c, then (u,x,v,y,u) is a C4?
        # Yes, u-x-v-y-u is a 4-cycle.
        # So the condition is: For every color c, and every pair of vertices (u, v), 
        # the number of common neighbors in color c is at most 1.
        # Wait, if u and v are adjacent in color c, say (u,v) is color c, and they share a common neighbor w, 
        # that's a triangle, not a C4.
        # The condition "No C4" is equivalent to: 
        # For any color c, the number of paths of length 2 between any two vertices u, v in G_c is at most 1?
        # If there are two distinct paths of length 2 between u and v, say u-a-v and u-b-v, 
        # then u-a-v-b-u is a 4-cycle (if a!=b and u!=v and a!=u etc).
        # Yes. So for each color c, for each pair (u, v), count the number of w such that (u,w) is c and (w,v) is c.
        # This count must be <= 1.
        
        # This is much faster to check incrementally or globally.
        # Global check cost: 
        # For each color (5), for each pair (u,v) (n^2/2), count common neighbors.
        # Naive count is O(n). Total O(5 * n^3). For n=15, 5 * 3375 = ~17k ops. Very fast.
        
        def count_violations(colors, n, edge_map):
            violations = 0
            # Precompute adjacency lists for each color?
            # adj[c][u] = set of neighbors
            adj = [ [set() for _ in range(n)] for _ in range(5) ]
            
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i].add(j)
                    adj[c][j].add(i)
                    idx += 1
            
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        # Count common neighbors of u and v in color c
                        common = len(adj[c][u].intersection(adj[c][v]))
                        if common > 1:
                            violations += common - 1
            return violations

        # Local Search
        # Objective: Minimize violations
        # Move: Change color of one edge
        
        current_violations = count_violations(colors, n, edge_map)
        best_local_violations = current_violations
        best_local_colors = colors[:]
        
        # Simulated Annealing / Hill Climbing
        # We want to reach 0 violations.
        
        max_iters = 50000
        temp = 1.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            if current_violations == 0:
                # Found valid coloring for this n
                best_n = n
                best_colors = colors[:]
                break
                
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            # Find i, j for e_idx
            # Inverse of edge_map
            # Since we built edge_map sequentially, we can just iterate or precompute inverse
            # Precomputing inverse for 105 edges (n=15) is trivial
            inv_map = {v: k for k, v in edge_map.items()}
            i, j = inv_map[e_idx]
            
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            # Apply change
            colors[e_idx] = new_color
            
            # Recalculate violations?
            # Recalculating from scratch is O(n^3) ~ 17k ops. 
            # 50000 iters * 17k ops = 850M ops. Might be too slow in Python.
            # We need incremental update or faster check.
            
            # Let's optimize count_violations.
            # Instead of sets, we can use bitmasks if n <= 64.
            # n=15 fits in 16-bit mask.
            
            # Let's rewrite count_violations using bitmasks for speed.
            
            pass # Placeholder, will implement below

        # Implementing bitmask version
        # adj_mask[c][u] is an integer where bit v is set if (u,v) is color c.
        
        def count_violations_fast(colors, n, edge_map):
            violations = 0
            # adj_mask[c][u]
            adj_mask = [ [0] * n for _ in range(5) ]
            
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj_mask[c][i] |= (1 << j)
                    adj_mask[c][j] |= (1 << i)
                    idx += 1
            
            for c in range(5):
                for u in range(n):
                    mask_u = adj_mask[c][u]
                    if mask_u == 0: continue
                    for v in range(u + 1, n):
                        mask_v = adj_mask[c][v]
                        if mask_v == 0: continue
                        common = mask_u & mask_v
                        # Count bits in common
                        # If count > 1, violation += count - 1
                        if common:
                            # Popcount
                            # Python 3.10+ has int.bit_count()
                            cnt = common.bit_count()
                            if cnt > 1:
                                violations += cnt - 1
            return violations

        # Restart loop with fast checker
        current_violations = count_violations_fast(colors, n, edge_map)
        best_local_violations = current_violations
        best_local_colors = colors[:]
        
        max_iters = 20000 # Reduce iters since fast checker is still not instant
        temp = 1.0
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            if current_violations == 0:
                best_n = n
                best_colors = colors[:]
                break
                
            e_idx = random.randint(0, num_edges - 1)
            i, j = inv_map[e_idx]
            
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            colors[e_idx] = new_color
            new_violations = count_violations_fast(colors, n, edge_map)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                if current_violations < best_local_violations:
                    best_local_violations = current_violations
                    best_local_colors = colors[:]
            else:
                colors[e_idx] = old_color
                
            temp *= cooling
            
        # If we didn't find 0, we might have improved it, but we need 0 for validity.
        # If we didn't find 0, this n failed. Try next n? 
        # Or try again with different random seed?
        # Given time constraints, if n=12 fails, n=13 is harder.
        # So if n=12 fails, we probably should stop trying larger n and return best found.
        
        if current_violations != 0:
            # If we couldn't solve n, likely can't solve n+1 easily
            # But let's try one more restart for the same n if time permits?
            # No, let's just break and return best_n.
            # Actually, if n=12 is hard, n=13 is harder.
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
