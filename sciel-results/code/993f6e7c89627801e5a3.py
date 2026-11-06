import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try to find a solution for n starting from high to low
    # 27 is the target for improvement, 26 is the known best
    for n in [27, 26, 25, 24]:
        if time.time() > deadline - 15.0:
            break
            
        # Allocate time for each n
        # Give 27 a chance, but don't spend too much if it's hard
        time_budget = 30.0 if n == 27 else 40.0
        
        result = solve_backtrack(n, time_budget)
        if result is not None:
            return result
            
    # Fallback to a small valid graph if all else fails
    # n=11 is the seed baseline
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def solve_backtrack(n, time_limit):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge to index
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    # colors[i] is the color of edges[i], -1 if unassigned
    colors = [-1] * num_edges
    
    # common_count[c][u][v] = number of common neighbors of u and v in color c
    # We only need to store for u < v to save memory, but for speed, full matrix is fine for n<=28
    # 5 colors * 28 * 28 ints is small
    common_count = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    # List of vertices for each color connected to u
    # neighbors[c][u] = list of v such that (u, v) is color c
    neighbors = [[[] for _ in range(n)] for _ in range(5)]
    
    def is_valid_color(u, v, c):
        # Check if assigning color c to (u, v) is valid
        # 1. Check pairs (v, x) where (u, x) is c
        for x in neighbors[c][u]:
            if common_count[c][v][x] >= 1:
                return False
        # 2. Check pairs (u, y) where (v, y) is c
        for y in neighbors[c][v]:
            if common_count[c][u][y] >= 1:
                return False
        # 3. Check pair (u, v) if they share a neighbor x in color c
        # i.e., if there exists x such that (u, x) is c and (v, x) is c
        # This means x is in neighbors[c][u] AND x is in neighbors[c][v]
        # We can check this by iterating the smaller list
        if len(neighbors[c][u]) < len(neighbors[c][v]):
            for x in neighbors[c][u]:
                if x in neighbors[c][v]:
                    if common_count[c][u][v] >= 1:
                        return False
        else:
            for x in neighbors[c][v]:
                if x in neighbors[c][u]:
                    if common_count[c][u][v] >= 1:
                        return False
        return True

    def assign_color(u, v, c, idx):
        colors[idx] = c
        neighbors[c][u].append(v)
        neighbors[c][v].append(u)
        
        # Update common_count
        # For x in neighbors[c][u] (excluding v? No, v is just added, so it's not in neighbors[c][u] yet? 
        # Wait, we appended v to neighbors[c][u]. 
        # So neighbors[c][u] now contains v.
        # When we iterate neighbors[c][u], we will see v.
        # common_count[c][v][v] is not used.
        # But we need to be careful: 
        # The new edge is (u, v).
        # For any x in OLD neighbors[c][u], the pair (v, x) gains common neighbor u.
        # For any y in OLD neighbors[c][v], the pair (u, y) gains common neighbor v.
        # If x == y (i.e., x in intersection), the pair (u, v) gains common neighbor x.
        
        # Since we already appended v, we should iterate over the list BEFORE appending?
        # Or just handle it carefully.
        # Let's revert the append logic for the check, or do it differently.
        
        # Better approach for assign:
        # 1. Save old lists
        # 2. Update counts based on old lists
        # 3. Append new edge
        
        pass # Placeholder

    def unassign_color(u, v, c, idx):
        colors[idx] = -1
        # Remove v from neighbors[c][u]
        neighbors[c][u].remove(v)
        neighbors[c][v].remove(u)
        
        # Revert common_count updates
        pass # Placeholder

    # Refined Assign/Unassign
    
    def do_assign(u, v, c, idx):
        colors[idx] = c
        
        # We need to update common_count based on the neighbors BEFORE adding (u, v)
        # Let's get the current neighbors
        nu = neighbors[c][u][:]
        nv = neighbors[c][v][:]
        
        # Update common_count for pairs involving v and x in nu
        for x in nu:
            common_count[c][v][x] += 1
            common_count[c][x][v] += 1
            
        # Update common_count for pairs involving u and y in nv
        for y in nv:
            common_count[c][u][y] += 1
            common_count[c][y][u] += 1
            
        # Update common_count for pair (u, v) if there is common neighbor
        # Intersection of nu and nv
        # Since n is small, simple check is fine
        # But we already checked validity. 
        # If there is an x in nu and nv, then common_count[c][u][v] increases by 1
        # Note: If x is in nu and nv, it was counted in both loops above?
        # Loop 1: pair (v, x). 
        # Loop 2: pair (u, x).
        # Pair (u, v) is separate.
        
        # Check intersection
        # To avoid O(N^2) or set operations, we can just check the smaller list
        # But we need to know HOW MANY common neighbors were added.
        # Actually, we just need to increment common_count[c][u][v] by the size of intersection.
        # Let's compute intersection size.
        # For n=28, lists are small.
        set_nv = set(nv)
        count_common = 0
        for x in nu:
            if x in set_nv:
                count_common += 1
                
        if count_common > 0:
            common_count[c][u][v] += count_common
            common_count[c][v][u] += count_common
            
        # Now add the edge to neighbor lists
        neighbors[c][u].append(v)
        neighbors[c][v].append(u)

    def do_unassign(u, v, c, idx):
        colors[idx] = -1
        
        # Remove from neighbor lists first?
        # We need the lists to revert counts.
        # The lists currently contain v and u.
        # The "old" lists for counting purposes are the ones without v/u.
        # So we can just use the current lists, but we must be careful not to count the edge itself.
        # Actually, the logic is symmetric.
        
        # Revert common_count
        nu = neighbors[c][u][:]
        nv = neighbors[c][v][:]
        
        # Remove v from nu and u from nv for the calculation of "old" neighbors
        # Wait, nu contains v. nv contains u.
        # The "old" neighbors of u were nu \ {v}.
        # The "old" neighbors of v were nv \ {u}.
        
        old_nu = [x for x in nu if x != v]
        old_nv = [y for y in nv if y != u]
        
        for x in old_nu:
            common_count[c][v][x] -= 1
            common_count[c][x][v] -= 1
            
        for y in old_nv:
            common_count[c][u][y] -= 1
            common_count[c][y][u] -= 1
            
        set_old_nv = set(old_nv)
        count_common = 0
        for x in old_nu:
            if x in set_old_nv:
                count_common += 1
                
        if count_common > 0:
            common_count[c][u][v] -= count_common
            common_count[c][v][u] -= count_common
            
        # Now remove from lists
        neighbors[c][u].remove(v)
        neighbors[c][v].remove(u)

    def backtrack(edge_idx):
        if time.time() > time_limit + start_time:
            return False
            
        if edge_idx == num_edges:
            return True
            
        u, v = edges[edge_idx]
        
        # Heuristic: Try colors in random order
        colors_to_try = [0, 1, 2, 3, 4]
        random.shuffle(colors_to_try)
        
        for c in colors_to_try:
            if is_valid_color(u, v, c):
                do_assign(u, v, c, edge_idx)
                if backtrack(edge_idx + 1):
                    return True
                do_unassign(u, v, c, edge_idx)
                
        return False

    # Run multiple random restarts if the first one fails or is slow
    # But backtracking is deterministic if order is fixed.
    # Randomizing the edge order or color order helps.
    
    # To maximize chance, we can randomize the edge order for each attempt
    # But we have a strict time limit.
    # Let's try one run with randomized colors.
    
    random.seed(42) # Deterministic for reproducibility in evaluation? Or random?
    # The task doesn't specify determinism. Random is fine.
    
    if backtrack(0):
        return {'n': n, 'colors': colors}
        
    return None
