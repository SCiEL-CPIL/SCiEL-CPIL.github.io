import time
import itertools
import sys

def construct():
    start_time = time.time()
    timeout = 140  # seconds, leaving 10s margin
    
    def solve_n(n):
        # Returns list of colors for n vertices, or None
        # Edges in lexicographic order
        num_edges = n * (n - 1) // 2
        if num_edges == 0:
            return []
        
        # Precompute C4s
        # A C4 is a set of 4 edges
        # To speed up, we check conflicts incrementally.
        
        # colors[i] is the color of edge i
        colors = [None] * num_edges
        
        # Map (i, j) to edge index
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * n - (i * (i + 1)) // 2 + (j - i - 1)
            # Wait, standard lexicographic index:
            # 0,1; 0,2; ... 0,n-1; 1,2; ...
            # index = i*(n-1) - i*(i+1)/2 + (j-i-1) ?
            # Let's use a simpler mapping or precompute.
            
        # Precompute edge indices
        edge_to_pair = {}
        pair_to_edge = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_to_pair[idx] = (i, j)
                pair_to_edge[(i, j)] = idx
                idx += 1
        
        # Precompute for each edge, which other edges form a C4 with it?
        # Actually, we check if adding edge (u,v) with color c creates a C4.
        # This happens if there exists w such that (u,w) and (v,w) are both color c.
        # So we maintain for each color c, and each vertex pair (u,v), the count of common neighbors w with color c on (u,w) and (v,w).
        # If count >= 1, adding (u,v) with color c creates a C4.
        
        # common_neighbors[c][u][v] = count of w such that color(u,w)==c and color(v,w)==c
        # We need to update this efficiently.
        
        # This state is O(5 * n^2) = 5 * 25^2 = 3125 ints. Very small.
        
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        
        # List of edges to assign
        edges = list(range(num_edges))
        
        # Heuristic: order edges by degree or something?
        # For now, just lexicographic.
        
        def backtrack(k):
            if time.time() - start_time > timeout:
                return False
            
            if k == num_edges:
                return True
            
            u, v = edge_to_pair[edges[k]]
            
            # Try colors 0..4
            # Symmetry breaking: if u==0, try only color 0?
            # If v==1 and u==0, try only 0,1?
            
            for c in range(5):
                # Check if color c is allowed
                if common[c][u][v] > 0:
                    continue
                
                # Assign color c to edge (u,v)
                colors[edges[k]] = c
                
                # Update common neighbors
                # For all w != u, v:
                # If color(u,w) is c, then common[c][v][w] increases?
                # Wait.
                # common[c][u][v] counts w such that (u,w) and (v,w) are c.
                # We just set (u,v) to c.
                # This affects common[c][u][x] for all x?
                # No.
                # common[c][a][b] is the number of w such that (a,w) and (b,w) are c.
                # We changed the color of (u,v).
                # This edge (u,v) acts as (u,w) for w=v, and (v,w) for w=u.
                # So it contributes to common[c][u][v] and common[c][v][u].
                # But we already checked common[c][u][v] == 0.
                # So adding it makes common[c][u][v] = 1.
                
                # Does it affect other common[c][a][b]?
                # Yes, if there is an edge (u, x) with color c, then (u,v) and (u,x) share u.
                # This contributes to common[c][v][x].
                # Specifically, if color(u,x) == c, then v and x have a common neighbor u with color c.
                # So we need to increment common[c][v][x] and common[c][x][v] for all x such that color(u,x) == c.
                # Similarly for all x such that color(v,x) == c, increment common[c][u][x].
                
                # This update is O(n) per edge.
                # Total O(n^3) per step? No, O(n) per step.
                # Backtracking depth n^2. Total O(n^4).
                # For n=15, 15^4 = 50k. Fast.
                # For n=25, 25^4 = 390k. Still fast.
                
                # Update
                changed = []
                for x in range(n):
                    if x == u or x == v: continue
                    if colors[pair_to_edge.get((u,x), -1)] == c:
                        common[c][v][x] += 1
                        common[c][x][v] += 1
                        changed.append((v, x, 1))
                    if colors[pair_to_edge.get((v,x), -1)] == c:
                        common[c][u][x] += 1
                        common[c][x][u] += 1
                        changed.append((u, x, 1))
                
                if backtrack(k+1):
                    return True
                
                # Undo
                for a, b, delta in changed:
                    common[c][a][b] -= delta
                    common[c][b][a] -= delta
                colors[edges[k]] = None
                
            return False
        
        return backtrack(0)

    # Try increasing n
    best_n = 11
    best_colors = None
    
    # Seed n=11 colors for fallback
    n_seed = 11
    best_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    for n in range(12, 27):
        if time.time() - start_time > timeout - 10:
            break
        
        # Reset state for new n
        # We call solve_n which handles its own state
        
        # To avoid restarting from scratch every time, we could use the previous solution,
        # but simple restart is safer for correctness.
        
        # Optimization: Only try if n is not too big for the remaining time.
        # n=12: 66 edges.
        # n=13: 78 edges.
        
        result = solve_n(n)
        if result:
            best_n = n
            best_colors = result
        else:
            # If it failed, it might be due to timeout or no solution.
            # If timeout, we stop.
            if time.time() - start_time > timeout - 5:
                break
            # If no solution found (and not timeout), we might continue to next n?
            # Usually if n fails, n+1 is harder.
            # But maybe n=12 fails and n=13 works? Unlikely.
            # Break to save time for verification or larger n?
            # Let's break to be safe.
            break

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
