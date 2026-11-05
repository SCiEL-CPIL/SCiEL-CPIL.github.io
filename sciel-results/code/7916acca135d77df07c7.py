import time
import random
import sys

def count_c4s(colors, n):
    """
    Count the number of monochromatic C4s.
    colors: list of length n*(n-1)//2
    """
    if n < 5:
        return 0
        
    # Map (i, j) to index
    # idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    # Or precompute indices for speed?
    # For n <= 15, simple loop is fine.
    
    # Build adjacency sets for each color
    adj = [ [set() for _ in range(n)] for _ in range(5) ]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    total_c4s = 0
    for c in range(5):
        for a in range(n):
            for b in range(a+1, n):
                # Count common neighbors of a and b in color c
                # Intersection of sets is fast in Python
                common = adj[c][a].intersection(adj[c][b])
                k = len(common)
                if k >= 2:
                    total_c4s += k * (k - 1) // 2
                    
    return total_c4s

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin for 120s limit
    
    # Initialize with a guaranteed valid small n
    # n=4 is always valid because K4 has no C4.
    best_n = 4
    best_colors = [0] * (4 * 3 // 2) # All 0s is fine for K4
    
    # Try to find valid colorings for n = 5 to 15
    for n in range(5, 16):
        if time.time() > deadline:
            break
            
        # Initialize
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        found = False
        restarts = 0
        max_restarts = 200 # Increase restarts for small n
        
        # Time limit per n?
        # Let's give 2 seconds per n for small n, less for larger.
        time_per_n = 2.0 if n < 10 else 0.5
        
        while restarts < max_restarts and time.time() < deadline:
            restarts += 1
            if restarts == 1:
                 pass # Keep initial random
            else:
                 # Perturb
                 for _ in range(5):
                     idx = random.randint(0, num_edges - 1)
                     colors[idx] = random.randint(0, 4)
            
            current_cost = count_c4s(colors, n)
            if current_cost == 0:
                found = True
                break
                
            # Hill Climbing
            steps = 0
            max_steps = 100
            while steps < max_steps and time.time() < deadline:
                steps += 1
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                old_color = colors[idx]
                
                # Try all 5 colors, pick best
                best_new_color = old_color
                best_cost = current_cost
                
                for new_c in range(5):
                    if new_c == old_color:
                        continue
                    colors[idx] = new_c
                    new_cost = count_c4s(colors, n)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_new_color = new_c
                
                if best_cost < current_cost:
                    colors[idx] = best_new_color
                    current_cost = best_cost
                    if current_cost == 0:
                        found = True
                        break
                else:
                    # Revert if no improvement found (keep old_color)
                    # It is already old_color if we didn't change it, 
                    # but we might have changed it during the loop.
                    # Actually, the loop changes colors[idx] multiple times.
                    # So we must restore it.
                    colors[idx] = old_color
                    
        if found:
            best_n = n
            best_colors = colors.copy()
            # Continue to next n
        else:
            # If we fail to find a valid coloring for this n, 
            # we might still try the next n? 
            # Usually if you can't find n, you can't find n+1 easily.
            # But random restarts might succeed later.
            # However, to save time, if we fail for n, we can break 
            # if n is large, or continue if n is small.
            # Let's just break if we've spent too much time.
            pass

    return {'n': best_n, 'colors': best_colors}
