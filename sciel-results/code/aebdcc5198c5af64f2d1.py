"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Base 32-vertex solution
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    n_base = 32
    base_colors_flat = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            base_colors_flat.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    n = 50
    # Use 2D array for speed
    colors = [[0]*n for _ in range(n)]
    
    # Embed base
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors[i][j] = base_colors_flat[idx]
            colors[j][i] = base_colors_flat[idx]
            idx += 1
            
    # Randomly initialize rest
    for i in range(n_base, n):
        for j in range(i + 1, n):
            c = random.randint(0, 3)
            colors[i][j] = c
            colors[j][i] = c
    for i in range(n_base):
        for j in range(n_base, n):
            c = random.randint(0, 3)
            colors[i][j] = c
            colors[j][i] = c

    # Count triangles
    # We only care about triangles that are NOT entirely within 0..31
    # Because those are already 0.
    # So we only need to count triangles with at least one vertex >= 32.
    
    def count_relevant_triangles():
        total = 0
        for i in range(n_base, n):
            for j in range(i + 1, n):
                c_ij = colors[i][j]
                for k in range(j + 1, n):
                    if colors[i][k] == c_ij and colors[j][k] == c_ij:
                        total += 1
        for i in range(n_base):
            for j in range(n_base, n):
                c_ij = colors[i][j]
                # Triangles (i, j, k) where k > j
                # k can be < n_base or >= n_base
                # If k < n_base, and i < k, we might double count?
                # Let's just iterate all i<j<k and check if any >= n_base.
                pass
        
        # Simpler: Iterate all i<j<k. If max(i,j,k) >= n_base, check mono.
        # This is 19600 checks. Fast.
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[i][j]
                for k in range(j + 1, n):
                    if i < n_base and j < n_base and k < n_base:
                        continue # Skip base
                    if colors[i][k] == c_ij and colors[j][k] == c_ij:
                        total += 1
        return total

    total_tri = count_relevant_triangles()
    best_total = total_tri
    best_colors = [row[:] for row in colors]
    
    # Local Search
    # Focus on edges that can break triangles.
    # We will pick a random triangle (i,j,k) that is mono and "relevant".
    # Then change one edge.
    
    while time.time() < deadline:
        if total_tri == 0:
            break
            
        # Find a relevant mono triangle
        # Random sampling
        found = False
        for _ in range(50):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            k = random.randint(0, n-1)
            if i == j or j == k or i == k: continue
            # Ensure i < j < k for consistency? Not necessary for check, but for index
            a, b, c = sorted([i, j, k])
            if a < n_base and b < n_base and c < n_base:
                continue # Base triangle, should be 0
                
            ca = colors[a][b]
            cb = colors[b][c]
            cc = colors[a][c]
            
            if ca == cb == cc:
                # Found one
                # Try changing edge (a,b)
                old_c = ca
                # Try other colors
                for new_c in range(4):
                    if new_c == old_c: continue
                    
                    # Check impact on triangles involving (a,b)
                    # Triangles are (a,b,x) for x != a,b
                    delta = 0
                    for x in range(n):
                        if x == a or x == b: continue
                        # Skip if triangle (a,b,x) is base-only?
                        # If a,b,x all < n_base, it's base. But base has no mono triangles.
                        # So delta will be 0 for base triangles.
                        
                        cx_a = colors[a][x]
                        cx_b = colors[b][x]
                        
                        # Old contribution
                        if cx_a == old_c and cx_b == old_c:
                            delta -= 1
                            
                        # New contribution
                        if cx_a == new_c and cx_b == new_c:
                            delta += 1
                            
                    if delta < 0:
                        # Improve
                        colors[a][b] = new_c
                        colors[b][a] = new_c
                        total_tri += delta
                        if total_tri < best_total:
                            best_total = total_tri
                            best_colors = [row[:] for row in colors]
                        found = True
                        break
                    elif delta == 0:
                        if random.random() < 0.05:
                            colors[a][b] = new_c
                            colors[b][a] = new_c
                            total_tri += delta
                            if total_tri < best_total:
                                best_total = total_tri
                                best_colors = [row[:] for row in colors]
                            found = True
                            break
                            
        if not found:
            # Perturbation
            # Pick random relevant edge
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b: continue
            if a < n_base and b < n_base: continue
            old_c = colors[a][b]
            new_c = random.randint(0, 3)
            if new_c != old_c:
                # Recount is expensive, so just do it and let next iteration figure out?
                # No, we need to update total_tri.
                # We can just do a full recount every 1000 perturbation steps?
                # Or just accept that total_tri might be inaccurate for a bit?
                # Better: Just do a full recount if we perturb.
                colors[a][b] = new_c
                colors[b][a] = new_c
                total_tri = count_relevant_triangles()
                if total_tri < best_total:
                    best_total = total_tri
                    best_colors = [row[:] for row in colors]

    if total_tri == 0:
        # Flatten colors
        flat = []
        for i in range(n):
            for j in range(i + 1, n):
                flat.append(colors[i][j])
        return {'n': n, 'colors': flat}
    else:
        # Return base 32
        return {'n': 32, 'colors': base_colors_flat}

# EVOLVE-BLOCK-END
