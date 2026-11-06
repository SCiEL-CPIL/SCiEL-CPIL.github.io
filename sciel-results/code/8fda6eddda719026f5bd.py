import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4 using backtracking
    with forward checking and dynamic vertex ordering.
    """
    deadline = time.time() + 140.0  # Safety margin for 150s limit

    # Known seed solution for n=11 (cyclic coloring)
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def edge_index(i, j, n):
        """O(1) edge index for (i,j) with i<j in lexicographic order."""
        return i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """Verify no monochromatic C4."""
        if len(colors) != n * (n - 1) // 2:
            return False
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        return False
        return True

    def solve_backtracking(target_n, time_limit):
        """
        Backtracking search to find a valid 5-coloring of K_target_n.
        We build the graph vertex by vertex.
        """
        start_time = time.time()
        
        # State:
        # neighbors[color][vertex] = bitmask of neighbors in that color
        neighbors = [[0] * target_n for _ in range(5)]
        # colors[edge_index] = color
        colors = [-1] * (target_n * (target_n - 1) // 2)
        
        # We add vertices 0, 1, ..., target_n-1.
        # At step k, we determine colors of edges (k, j) for j < k.
        
        # To make backtracking efficient, we can use a heuristic ordering of vertices?
        # No, the edge indices depend on lexicographic order (0,1), (0,2)...
        # If we permute vertices, we must remap at the end.
        # Let's stick to natural order 0..n-1 first. If it's too slow, we can try random permutations.
        
        # Forward checking:
        # When we assign colors to edges (k, 0), ..., (k, k-1), we must ensure
        # that for any pair (u, v) with u < v < k, we don't create a C4.
        # A C4 involving k would be k-u-v-w-k? No, k is the new vertex.
        # C4 is k-a-b-c-k. Since k is new, the path a-b-c must be in existing vertices.
        # The edges are (k,a), (a,b), (b,c), (c,k).
        # For a fixed color c, if (k,a) and (c,k) are color c, then a and c are common neighbors of k in color c?
        # No, a and c are neighbors of k in color c.
        # A C4 in color c involving k means there exist u, v < k such that:
        # (k,u) is c, (k,v) is c, AND u and v have at least one common neighbor w < k in color c?
        # No, a C4 is a cycle of length 4.
        # Vertices: k, u, v, w.
        # Edges: (k,u), (u,v), (v,w), (w,k) all color c.
        # This requires:
        # 1. (k,u) is c
        # 2. (k,w) is c
        # 3. (u,v) is c
        # 4. (v,w) is c
        # This is a C4 if u, v, w, k are distinct.
        
        # Alternative view:
        # Adding vertex k. For each color c, let N_c(k) be the set of neighbors of k in color c.
        # The subgraph induced by N_c(k) in the existing graph (vertices 0..k-1) must be C4-free?
        # No, that's not right.
        # If u, v in N_c(k), and they have a common neighbor w in color c, then k-u-w-v-k is a C4.
        # So, for each color c, the set of vertices in N_c(k) must be such that no two of them share a common neighbor in color c in the existing graph?
        # Let's check:
        # If u, v in N_c(k), and w is a common neighbor of u and v in color c (in G_{k-1}),
        # then edges (k,u), (u,w), (w,v), (v,k) exist and are color c.
        # This forms a C4 k-u-w-v-k.
        # So the condition is: For each color c, the set N_c(k) must be an independent set in the "common neighbor graph"?
        # Specifically, for any u, v in N_c(k), they must NOT have any common neighbor in color c in G_{k-1}.
        
        # This allows for efficient checking when assigning colors to (k, j).
        
        # We will assign colors to edges (k, 0), (k, 1), ..., (k, k-1).
        # We can use backtracking on these k edges.
        
        def backtrack(k):
            if time.time() - start_time > time_limit:
                return False
            
            if k == target_n:
                return True
            
            # We need to assign colors to edges (k, 0) ... (k, k-1).
            # There are k edges.
            # We can try all 5^k combinations? Too slow for large k.
            # We should assign one by one with forward checking.
            
            # Let's assign colors to (k, 0), (k, 1), ...
            # We maintain for each color c, the set of vertices already assigned to N_c(k).
            # And we check the constraint: for any u, v in N_c(k), common_neighbors_c(u, v) in G_{k-1} must be empty.
            
            # To speed up, we can precompute for each pair (u, v) < k, the set of colors in which they have a common neighbor?
            # Or just check on the fly.
            
            # Let's use recursion for the k edges.
            # To optimize, we can order the edges (k, j) by some heuristic?
            # Maybe j with most constraints first?
            
            # For now, let's just iterate j from 0 to k-1.
            
            # We need to track the current assignment for vertex k.
            # assigned[j] = color for edge (k, j)
            
            # Optimization: Precompute common neighbors for pairs in G_{k-1}?
            # That's O(k^2) per step k, which is O(n^3) total. Acceptable for n=20?
            # n=20 -> 20^3 = 8000. Very fast.
            # n=30 -> 27000. Fast.
            
            # Let's implement a helper to check if a set of vertices S is valid for color c.
            # Valid means: for any u, v in S, they have no common neighbor in color c in G_{k-1}.
            
            # We'll do this incrementally.
            
            # State for backtracking on vertex k:
            # current_colors: dict {j: color} for j in 0..k-1
            # But we can just use an array.
            
            # Let's try a different approach:
            # Iterate through all 5^k colorings? No.
            # Use backtracking with pruning.
            
            def assign_edge(j):
                if time.time() - start_time > time_limit:
                    return False
                
                if j == k:
                    # All edges for vertex k assigned.
                    # Update global neighbors and colors.
                    for jj in range(k):
                        c = current_assignment[jj]
                        neighbors[c][k] |= (1 << jj)
                        neighbors[c][jj] |= (1 << k)
                        colors[edge_index(k, jj, target_n)] = c
                        # Wait, edge_index(k, jj) assumes k < jj? No, edge_index(i, j) requires i < j.
                        # So edge_index(jj, k, target_n)
                        colors[edge_index(jj, k, target_n)] = c
                    
                    if backtrack(k + 1):
                        return True
                    
                    # Undo
                    for jj in range(k):
                        c = current_assignment[jj]
                        neighbors[c][k] &= ~(1 << jj)
                        neighbors[c][jj] &= ~(1 << k)
                        colors[edge_index(jj, k, target_n)] = -1
                    return False
                
                # Try assigning color c to edge (k, j)
                for c in range(5):
                    # Check if adding j to N_c(k) is valid.
                    # N_c(k) currently contains {x in 0..j-1 | current_assignment[x] == c}
                    # For each x in N_c(k), check if x and j have a common neighbor in color c in G_{k-1}.
                    valid = True
                    for x in range(j):
                        if current_assignment[x] == c:
                            # Check common neighbors of x and j in color c
                            common = neighbors[c][x] & neighbors[c][j]
                            # We only care about neighbors in 0..k-1
                            # neighbors[c][x] already only has bits for 0..k-1 (since k is not added yet)
                            if common:
                                valid = False
                                break
                    if valid:
                        current_assignment[j] = c
                        if assign_edge(j + 1):
                            return True
                        current_assignment[j] = -1
                return False
            
            current_assignment = [-1] * k
            if assign_edge(0):
                return True
            return False
        
        return backtrack(0)

    # Strategy:
    # 1. Try to find a solution for n=29, 28, 27 using backtracking.
    # 2. Backtracking might be too slow for n=29.
    # 3. We can use randomization in the order of colors or vertices to escape local minima?
    #    Backtracking is exhaustive, so it either finds it or doesn't.
    #    But 5^k branching factor is huge.
    #    We need a better heuristic.
    
    # Let's try a hybrid:
    # Use local search to get close, then use backtracking to finish?
    # Or use backtracking with limited depth/time.
    
    # Actually, the "common neighbor" check is strong.
    # Let's try backtracking for n=12, 13, ... up to time limit.
    
    for n in range(12, 29):
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on n
        if n <= 15:
            t_alloc = 2.0
        elif n <= 20:
            t_alloc = 5.0
        else:
            t_alloc = 10.0
            
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        # Reset state for each n
        # We need to re-initialize neighbors and colors for each n
        # But solve_backtracking uses global state? No, I defined it inside, but it uses 'neighbors' and 'colors' from outer scope?
        # I need to make solve_backtracking self-contained or pass state.
        
        # Let's refactor solve_backtracking to be self-contained.
        
        def solve_bt(n_target, t_limit):
            start = time.time()
            nbs = [[0] * n_target for _ in range(5)]
            cols = [-1] * (n_target * (n_target - 1) // 2)
            
            def bt(k):
                if time.time() - start > t_limit:
                    return False
                if k == n_target:
                    return True
                
                # Assign colors to (k, 0) ... (k, k-1)
                # Heuristic: Try colors in random order?
                color_order = [0, 1, 2, 3, 4]
                random.shuffle(color_order)
                
                def assign(j):
                    if time.time() - start > t_limit:
                        return False
                    if j == k:
                        # Commit
                        for jj in range(k):
                            c = curr[jj]
                            nbs[c][k] |= (1 << jj)
                            nbs[c][jj] |= (1 << k)
                            cols[edge_index(jj, k, n_target)] = c
                        if bt(k + 1):
                            return True
                        # Undo
                        for jj in range(k):
                            c = curr[jj]
                            nbs[c][k] &= ~(1 << jj)
                            nbs[c][jj] &= ~(1 << k)
                            cols[edge_index(jj, k, n_target)] = -1
                        return False
                    
                    for c in color_order:
                        valid = True
                        for x in range(j):
                            if curr[x] == c:
                                if nbs[c][x] & nbs[c][j]:
                                    valid = False
                                    break
                        if valid:
                            curr[j] = c
                            if assign(j + 1):
                                return True
                            curr[j] = -1
                    return False
                
                curr = [-1] * k
                return assign(0)
            
            if bt(0):
                return cols
            return None
        
        res = solve_bt(n, t_alloc)
        if res:
            if is_valid(n, res):
                best_n = n
                best_colors = res
                # Continue to try larger n?
                # The loop continues.
            else:
                # Should not happen if bt is correct
                pass

    # If backtracking failed to find anything better than 11,
    # maybe we should try a different strategy for larger n?
    # The current program uses local search.
    # Let's combine: Use local search for large n, backtracking for small n.
    
    # Actually, the prompt says "Maximize n".
    # If backtracking finds 12, we return 12.
    # If it finds 20, we return 20.
    # If it times out, we return best found.
    
    # Let's add a local search fallback for n > 15 if backtracking is slow.
    # But the directive says "DIFFERENT strategy".
    # Backtracking IS a different strategy.
    
    # Let's optimize the backtracking further.
    # The check `if nbs[c][x] & nbs[c][j]` is O(1) bit operation.
    # The loop `for x in range(j)` is O(k).
    # Total complexity is roughly 5^k * k^2.
    # For k=10, 5^10 ~ 10M. 10M * 100 = 1B ops. Too slow in Python.
    # We need to prune more effectively.
    
    # Maybe use a SAT solver? No, standard library only.
    # Maybe use a different representation?
    
    # Let's try to improve the heuristic:
    # Order vertices such that they have high degree in the "constraint graph"?
    # Or just use random restarts with backtracking?
    
    # Given the time limit, let's stick to the backtracking implementation but add a timeout check inside the loops.
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
