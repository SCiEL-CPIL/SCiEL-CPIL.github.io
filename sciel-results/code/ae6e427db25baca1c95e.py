import time
import random
import math

def count_c4s(colors, n):
    """
    Count the total number of monochromatic C4s in the given coloring.
    colors: list of integers (0-4) for each edge (i,j) with i<j in lexicographic order.
    n: number of vertices.
    """
    # Map edge index to (i, j)
    # It's faster to build adjacency lists for each color
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            # Actually, we need adjacency lists for vertices
            # Let's use a list of sets for each color and vertex
            idx += 1
    
    # Re-do: Build adjacency lists per color
    adj = [ [set() for _ in range(n)] for _ in range(5) ]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    total_c4s = 0
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                # Count common neighbors of i and j in color c
                common = adj[c][i] & adj[c][j]
                k = len(common)
                if k >= 2:
                    total_c4s += k * (k - 1) // 2
    return total_c4s

def solve_n(n, time_limit):
    """
    Attempt to find a valid 5-coloring of K_n with no monochromatic C4s.
    Uses Simulated Annealing.
    Returns the coloring list if found, else None.
    """
    start_time = time.time()
    num_edges = n * (n - 1) // 2
    
    # Initialize with random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    current_cost = count_c4s(colors, n)
    best_colors = colors[:]
    best_cost = current_cost
    
    if current_cost == 0:
        return colors
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random edge
        idx = random.randint(0, num_edges - 1)
        old_c = colors[idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
            
        # Apply move
        colors[idx] = new_c
        new_cost = count_c4s(colors, n)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors
        else:
            # Reject move
            colors[idx] = old_c
            
        T *= cooling_rate
        
    return None

def construct():
    deadline = time.time() + 110  # Safety margin
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to increase n
    for n in range(12, 27): # Try up to 26
        if time.time() > deadline - 10: # Leave some time for final check/return
            break
            
        # Allocate time for this n. More time for larger n? Or fixed?
        # Let's give 5 seconds per attempt initially, increasing if we fail?
        # Actually, if we fail for n, we probably can't solve n+1 easily without a different start.
        # But sometimes a good random start works.
        # Let's try 2 seconds per n first.
        time_limit = 2.0
        result = solve_n(n, time_limit)
        
        if result is not None:
            best_n = n
            best_colors = result
            # If we found a solution, we can try to extend or just move to next n
            # We don't need to keep searching for this n, we move to n+1
        else:
            # If we fail, we might try again with more time?
            # Or break?
            # Since the problem is hard, failing at n might mean n is too hard.
            # But we know 26 is possible.
            # Let's try a few more times with different seeds?
            # Or just break and return the best found so far.
            # Given the time constraint, breaking is safer.
            # However, we can try a bit harder for the next step.
            # Let's just break for now to ensure we return a valid result quickly.
            # Actually, we can try to increase time limit for subsequent failures?
            # No, let's stick to a simple loop. If we fail at 12, we return 11.
            # This is suboptimal if 12 is easy but we had bad luck.
            # Let's do a few restarts for each n.
            pass

    # If we broke early, we might have time left.
    # We can try to improve the best_n found? No, we just need to maximize n.
    # So we just return the best valid n found.
    
    return {'n': best_n, 'colors': best_colors}
