import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = 50.0  # 50 seconds out of 60
    
    # Helper to get edge index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and the 3 ways to form a cycle.
    # Actually, for a set of 4 vertices, there are 3 distinct 4-cycles.
    # We check all 3.
    cycles = []
    vertices = range(N)
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # 3 cycles on {a,b,c,d}
                    # 1: a-b-c-d-a
                    # 2: a-b-d-c-a
                    # 3: a-c-b-d-a
                    cycles.append([(a,b), (b,c), (c,d), (d,a)])
                    cycles.append([(a,b), (b,d), (d,c), (c,a)])
                    cycles.append([(a,c), (c,b), (b,d), (d,a)])
    
    # Initialize with bisection structure (15 colors)
    # Split 0-7 and 8-15
    # Inter color 0
    # Intra 0-7: Bisection 0-3, 4-7. Inter 1. Intra 0-3: Bisection 0-1, 2-3. Inter 2. Intra 0-1: 3, 2-3: 4.
    # This is getting complicated to hardcode.
    # Let's just start random and search.
    
    colors = [0] * NUM_EDGES
    
    def is_valid(col):
        for cyc in cycles:
            c0 = col[edge_idx(cyc[0][0], cyc[0][1])]
            c1 = col[edge_idx(cyc[1][0], cyc[1][1])]
            c2 = col[edge_idx(cyc[2][0], cyc[2][1])]
            c3 = col[edge_idx(cyc[3][0], cyc[3][1])]
            if len({c0, c1, c2, c3}) == 4:
                return False
        return True

    def num_colors(col):
        return len(set(col))

    best_colors = colors
    best_count = 0
    
    # Simple local search
    # Try to change random edges to new colors
    for _ in range(100000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[idx]
        new_color = random.randint(0, 119)
        
        if old_color == new_color:
            continue
            
        colors[idx] = new_color
        
        if is_valid(colors):
            curr_count = num_colors(colors)
            if curr_count > best_count:
                best_count = curr_count
                best_colors = colors[:]
        else:
            colors[idx] = old_color
            
    return best_colors
