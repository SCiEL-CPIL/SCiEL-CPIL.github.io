import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Seed program's result for fallback
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def seed_coloring():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    best_result = seed_coloring()
    best_n = 32

    # Function to count triangles using bitsets
    # colors is a 2D list colors[i][j] for i < j
    # We maintain masks[i][c] as an integer bitset
    def solve_sa(n, max_time):
        if n <= best_n:
            return None
            
        # Initialize random coloring
        # colors[i][j] for i < j
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j] # Symmetry for access, though we only use i<j logic mostly

        # Build masks
        # masks[v][c] is bitset of neighbors u such that color(v,u) == c
        masks = [[0] * 4 for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)

        # Calculate initial total triangles
        total_tri = 0
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[i][j] == c:
                        total_tri += (masks[i][c] & masks[j][c]).bit_count()
        total_tri //= 3 # Each triangle counted 3 times (once per vertex)

        if total_tri == 0:
            return n, colors

        T = 10.0
        cooling = 0.999
        # Estimate steps based on time
        # Each step is very fast. Let's just run until time limit.
        
        while time.time() < max_time:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = colors[i][j]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue

            # Calculate delta triangles
            # Remove contribution of c_old
            # Add contribution of c_new
            
            # Triangles involving edge (i,j)
            # For a color c, number of triangles is popcount(masks[i][c] & masks[j][c])
            
            # Note: masks[i][c] includes j if color(i,j)==c. 
            # But we are changing color(i,j).
            # The intersection masks[i][c] & masks[j][c] includes k where color(i,k)=c and color(j,k)=c.
            # If k is i or j, it's not a triangle. 
            # Since i is not in masks[i], and j is not in masks[j], the intersection doesn't include i or j.
            # So we just count the common neighbors.
            
            tri_old = (masks[i][c_old] & masks[j][c_old]).bit_count()
            tri_new = (masks[i][c_new] & masks[j][c_new]).bit_count()
            
            delta_tri = tri_new - tri_old
            
            if delta_tri <= 0 or random.random() < math.exp(-delta_tri / T):
                # Accept
                # Update colors
                colors[i][j] = c_new
                colors[j][i] = c_new
                
                # Update masks
                masks[i][c_old] &= ~(1 << j)
                masks[j][c_old] &= ~(1 << i)
                masks[i][c_new] |= (1 << j)
                masks[j][c_new] |= (1 << i)
                
                total_tri += delta_tri
                
                if total_tri == 0:
                    return n, colors

            T *= cooling
            if T < 0.01:
                T = 0.01

        return None

    # Try to solve for descending n
    targets = [50, 48, 45, 42, 40]
    
    for target_n in targets:
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this target
        # If we are trying 50, give it most time.
        # Simple strategy: give each target a fixed chunk, or run until deadline if it's the last.
        # Let's give 50s to 50, 20s to 48, 10s to 45, etc.
        
        time_limit = min(deadline - 5, time.time() + 50)
        if target_n == 48:
             time_limit = min(deadline - 5, time.time() + 20)
        elif target_n == 45:
             time_limit = min(deadline - 5, time.time() + 10)
        else:
             time_limit = min(deadline - 5, time.time() + 5)

        result = solve_sa(target_n, time_limit)
        if result:
            n, colors = result
            # Construct the output list
            out_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    out_colors.append(colors[i][j])
            return {'n': n, 'colors': out_colors}

    return best_result
