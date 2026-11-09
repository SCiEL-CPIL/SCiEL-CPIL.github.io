import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # We want to maximize n.
    # Try n=26 first. If found, return.
    # If not, try n=25, etc.
    # But 26 is the target.
    
    def solve(n):
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Neighborhoods for each color: list of int bitmasks
        # neigh[c][v] is a bitmask of neighbors of v in color c
        neigh = [[0] * n for _ in range(5)]
        
        # colors[k] for edge k
        colors = [-1] * m
        
        def check_valid(u, v, c):
            # Check if adding (u,v) with color c creates a C4
            # C4 exists if u and v have >= 2 common neighbors in color c
            common = neigh[c][u] & neigh[c][v]
            # Count bits
            if common & (common - 1):
                # If there are at least 2 bits set, it's a C4
                # common & (common - 1) removes the lowest set bit.
                # If the result is non-zero, there was more than one bit.
                return False
            return True

        def backtrack(idx):
            if idx == m:
                return True
            
            if time.time() > deadline:
                return False
            
            u, v = edges[idx]
            
            # Heuristic: Try colors in random order or specific order
            # For symmetry breaking, we can fix the first edge to color 0
            if idx == 0:
                candidates = [0]
            else:
                candidates = [0, 1, 2, 3, 4]
                # Randomize to avoid getting stuck in local minima?
                # random.shuffle(candidates)
                
            for c in candidates:
                if check_valid(u, v, c):
                    # Assign
                    colors[idx] = c
                    neigh[c][u] |= (1 << v)
                    neigh[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Unassign
                    neigh[c][u] &= ~(1 << v)
                    neigh[c][v] &= ~(1 << u)
                    colors[idx] = -1
                    
            return False

        # Try multiple random restarts with different orderings?
        # Or just one run?
        # With n=26, backtracking might be slow if it fails early.
        # But the density is low? No, it's a complete graph.
        # The constraint is tight.
        
        # Let's try a few restarts with random edge orderings if the first fails?
        # Or random color orderings.
        
        # For now, try standard order.
        if backtrack(0):
            return colors
        
        return None

    # Try to find a solution for n=26
    # Since it might be hard, we might need to be clever.
    # But let's try.
    
    best_n = 11
    best_colors = [0] * (11 * 10 // 2) # Fallback to seed
    
    # Try n=26
    res = solve(26)
    if res is not None:
        return {'n': 26, 'colors': res}
        
    # If 26 fails, try 25?
    # The prompt says 26 is known.
    # If my search is too slow, I might time out.
    # I should add a time check inside backtrack.
    
    # If 26 is too hard for simple DFS, I might need to fall back to a greedy construction
    # or a smaller n.
    # But the score for 26 is much better.
    
    # Let's add a timeout guard to the outer loop too.
    
    # If solve(26) times out, it returns None.
    # Then try 25?
    # 25 is also good.
    
    for n in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline:
            break
        res = solve(n)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback to seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
