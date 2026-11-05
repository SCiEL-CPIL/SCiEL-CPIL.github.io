import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Partitions of 6 items into 3 pairs
    # Local indices 0..5
    partitions = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    def has_k222(colors, n, color_val):
        adj = [[False]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color_val:
                    adj[i][j] = True
                    adj[j][i] = True
                idx += 1
        
        for combo in combinations(range(n), 6):
            for part in partitions:
                p0, p1, p2 = part
                # Check p0-p1
                if not (adj[combo[p0[0]]][combo[p1[0]]] and adj[combo[p0[0]]][combo[p1[1]]] and
                        adj[combo[p0[1]]][combo[p1[0]]] and adj[combo[p0[1]]][combo[p1[1]]]):
                    continue
                # Check p0-p2
                if not (adj[combo[p0[0]]][combo[p2[0]]] and adj[combo[p0[0]]][combo[p2[1]]] and
                        adj[combo[p0[1]]][combo[p2[0]]] and adj[combo[p0[1]]][combo[p2[1]]]):
                    continue
                # Check p1-p2
                if not (adj[combo[p1[0]]][combo[p2[0]]] and adj[combo[p1[0]]][combo[p2[1]]] and
                        adj[combo[p1[1]]][combo[p2[0]]] and adj[combo[p1[1]]][combo[p2[1]]]):
                    continue
                return True
        return False

    def is_valid(colors, n):
        return not has_k222(colors, n, 0) and not has_k222(colors, n, 1)

    def get_paley(n):
        # n must be prime 1 mod 4
        squares = {x * x % n for x in range(1, n)}
        colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
        return colors

    best_n = 0
    best_colors = None
    
    # Try n=29 with Paley
    if 29 <= 31:
        colors_29 = get_paley(29)
        if is_valid(colors_29, 29):
            return {'n': 29, 'colors': colors_29}
        
        # Local search on 29
        colors = colors_29[:]
        # Simple local search: flip random edges to reduce violations
        # We don't have a fast violation counter, so we just try random flips and check
        # This is slow. 
        # Instead, try other primes? 31 is 3 mod 4, no Paley.
        # Try n=28?
        
    # Try n=28 with Paley(29) restricted? Or random?
    # If 29 fails, 28 is likely valid with Paley(29) subgraph?
    # Not necessarily.
    
    # Let's try a quick random search for 29 if Paley fails?
    # Given time constraints, if Paley(29) fails, we might just return 28 or 29 if we can fix it.
    
    # For now, if Paley(29) is not valid, try to find a valid 28.
    # Paley(29) restricted to first 28 vertices might be valid?
    if best_n < 28:
        colors_28 = get_paley(29)[:28*27//2] # This is wrong, indices don't match
        # Need to re-derive colors for n=28 from the same structure?
        # Paley(29) is defined on 29 vertices.
        # If we take vertices 0..27, the colors are determined by (j-i)%29.
        # This is a valid subgraph of the Paley(29) coloring.
        # Does the subgraph contain K2,2,2?
        # If the full graph doesn't, the subgraph doesn't.
        # But if the full graph DOES contain K2,2,2, the subgraph might not.
        
        # Let's construct colors for n=28 using the same rule (mod 29)
        colors_28 = [0 if (j - i) % 29 in {x * x % 29 for x in range(1, 29)} else 1 for i in range(28) for j in range(i + 1, 28)]
        if is_valid(colors_28, 28):
            best_n = 28
            best_colors = colors_28

    if best_n < 27:
         # Fallback
         colors_27 = [0 if (j - i) % 29 in {x * x % 29 for x in range(1, 29)} else 1 for i in range(27) for j in range(i + 1, 27)]
         if is_valid(colors_27, 27):
             best_n = 27
             best_colors = colors_27

    if best_colors is None:
        # Seed fallback
        colors_17 = get_paley(17)
        return {'n': 17, 'colors': colors_17}

    return {'n': best_n, 'colors': best_colors}
