"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _mul_gf16(a, b):
    """Multiply in GF(16) using irreducible polynomial x^4 + x + 1 (0b10011)."""
    r = 0
    for i in range(4):
        if (b >> i) & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if (r >> i) & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_seed32():
    """Build the Greenwood-Gleason 32-vertex 4-coloring (Klebsch graph doubled)."""
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = _mul_gf16(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _color_index(n, i, j):
    """Index of edge (i,j) with i<j in the colors list."""
    # edges before vertex i: sum over k=0..i-1 of (n-1-k) = i*(n-1) - i*(i-1)//2
    base = i * (n - 1) - i * (i - 1) // 2
    return base + (j - i - 1)

def _count_mono_triangles(colors, n):
    """Count monochromatic triangles in a 4-coloring of K_n."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if colors[_color_index(n, i, k)] == c and colors[_color_index(n, j, k)] == c:
                    count += 1
    return count

def _count_mono_triangles_fast(colors, n):
    """Faster triangle count using adjacency per color."""
    # For each color, build adjacency sets and count triangles
    total = 0
    for c in range(4):
        # For each pair (i,j) with color c, count common neighbors with color c
        for i in range(n):
            for j in range(i + 1, n):
                if colors[_color_index(n, i, j)] == c:
                    # Count k > j such that (i,k) and (j,k) are both color c
                    # This is expensive; use a different approach
                    pass
    # Use the straightforward O(n^3) but with early structure
    count = 0
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if colors[_color_index(n, i, j)] != c:
                    continue
                for k in range(j + 1, n):
                    if colors[_color_index(n, i, k)] == c and colors[_color_index(n, j, k)] == c:
                        count += 1
    return count

def _random_coloring(n, seed=None):
    """Generate a random 4-coloring of K_n."""
    rng = random.Random(seed)
    return [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]

def _local_search(colors, n, max_iters, time_limit, start_time, seed=0):
    """Local search: try to reduce mono triangles to zero. Returns (colors, triangles)."""
    rng = random.Random(seed)
    triangles = _count_mono_triangles(colors, n)
    if triangles == 0:
        return colors, 0
    
    best_colors = colors[:]
    best_triangles = triangles
    
    # Temperature for simulated annealing
    temp = 10.0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge to recolor
        idx = rng.randint(0, len(colors) - 1)
        old_color = colors[idx]
        new_color = rng.randint(0, 3)
        if new_color == old_color:
            continue
        
        colors[idx] = new_color
        new_triangles = _count_mono_triangles(colors, n)
        
        if new_triangles <= best_triangles:
            best_colors = colors[:]
            best_triangles = new_triangles
            temp = max(0.1, temp * 0.99)
        else:
            # Accept with probability exp(-(delta)/temp) if it's not too bad
            delta = new_triangles - triangles
            if delta < 50 and rng.random() < (0.05 * (50 - delta) / 50):
                triangles = new_triangles
            else:
                colors[idx] = old_color
                # Don't update triangles since we reverted
        triangles = _count_mono_triangles(colors, n) if colors[idx] != old_color else triangles
        
        if triangles == 0:
            return colors, 0
    
    # Restore best
    return best_colors, best_triangles

def _try_extend(colors32, n_target, time_budget, seed=0):
    """Try to extend the 32-vertex coloring to n_target vertices."""
    start_time = time.time()
    n = 32
    colors = colors32[:]
    
    for target_n in range(33, n_target + 1):
        if time.time() - start_time > time_budget:
            break
        
        # Add new vertex with random colors
        new_colors = []
        rng = random.Random(seed + target_n)
        for i in range(32):
            new_colors.append(rng.randint(0, 3))
        colors = colors + new_colors
        n = target_n
        
        # Try to fix triangles
        max_iters = 50000
        colors, tri = _local_search(colors, n, max_iters, time_budget - (time.time() - start_time), start_time, seed=seed + target_n)
        
        if tri != 0:
            # Failed to reach target, go back
            n = 32
            colors = colors32[:]
            break
    
    return colors, n

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s
    
    # Start with the known valid n=32 coloring
    colors32 = _build_seed32()
    best_n = 32
    best_colors = colors32[:]
    
    # Verify n=32 is valid
    tri32 = _count_mono_triangles(colors32, 32)
    if tri32 != 0:
        # Should not happen, but just in case
        pass
    
    # Try to extend to higher n using local search
    # Strategy: start from n=32, try to add vertices one by one
    # Use multiple random restarts for the extension
    
    attempts = 0
    for target_n in [33, 34, 35, 36, 37, 38, 40, 42, 45, 50]:
        if time.time() - start_time > time_limit - 30:
            break
        
        # Multiple attempts for each target
        found = False
        for attempt in range(3):
            if time.time() - start_time > time_limit - 30:
                break
            
            remaining = time_limit - (time.time() - start_time)
            colors, n = _try_extend(colors32, target_n, remaining * 0.9, seed=1000 * target_n + attempt)
            
            if n > best_n:
                # Verify
                tri = _count_mono_triangles(colors, n)
                if tri == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
            elif n == target_n:
                tri = _count_mono_triangles(colors, n)
                if tri == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
        
        if found and best_n >= target_n:
            continue
        else:
            # If we failed at this target, likely won't succeed at higher ones
            # But try one more level up before giving up
            if best_n < target_n - 2:
                break
    
    # Final: if we didn't improve, return the n=32 result
    if best_n == 32:
        return {'n': 32, 'colors': best_colors}
    
    # Verify final result
    final_tri = _count_mono_triangles(best_colors, best_n)
    if final_tri != 0:
        # Something went wrong, return safe n=32
        return {'n': 32, 'colors': colors32}
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
