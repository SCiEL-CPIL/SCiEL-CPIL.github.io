"""Model-and-optimize: construct large triangle-free 4-colorings via bitmask local search."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _baseline32():
    """Greenwood-Gleason K16 doubled -> K32."""
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
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

def _build_idx(n):
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    return idx

def _count_triangles(colors, idx, n):
    tri = 0
    for u in range(n):
        for v in range(u + 1, n):
            c1 = colors[idx[u][v]]
            for w in range(v + 1, n):
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    tri += 1
    return tri

def _init_masks(colors, idx, n):
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
    return masks

def _solve_n(n, rng, deadline, init_colors=None, max_restarts=10):
    """
    Attempt to find a valid 4-coloring of K_n such that no monochromatic triangle exists.
    Returns the colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    if m == 0:
        return []
    
    idx = _build_idx(n)
    
    best_tri = float('inf')
    best_colors = None
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Initialize coloring
        if init_colors is not None and restart == 0:
            colors = init_colors[:]
        else:
            colors = [rng.randint(0, 3) for _ in range(m)]
        
        # Initialize masks
        masks = _init_masks(colors, idx, n)
        
        # Calculate initial triangle count
        tri_count = _count_triangles(colors, idx, n)
        
        if tri_count == 0:
            return colors
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_colors = colors[:]
        
        # Simulated Annealing with adaptive cooling
        T = 1.0
        cooling = 0.999
        min_T = 0.005
        steps = 0
        max_steps = 500000
        
        while T > min_T and steps < max_steps and time.time() < deadline - 1.0:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            i_idx = idx[u][v]
            old_c = colors[i_idx]
            new_c = rng.randint(0, 3)
            if old_c == new_c:
                continue
            
            lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
            gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
            
            delta = gained - lost
            
            accept = False
            if delta < 0:
                accept = True
            else:
                if delta > 100 * T:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta / T)
            
            if accept:
                colors[i_idx] = new_c
                tri_count += delta
                
                masks[u][old_c] &= ~(1 << v)
                masks[v][old_c] &= ~(1 << u)
                masks[u][new_c] |= (1 << v)
                masks[v][new_c] |= (1 << u)
                
                if tri_count == 0:
                    return colors
                
                T *= cooling
                steps += 1
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_colors = colors[:]
    
    if best_tri == 0 and best_colors is not None:
        return best_colors
    return None

def _extend_vertex(colors, idx, n, new_v, masks, rng):
    """
    Add a new vertex new_v to an existing valid coloring of K_n.
    Returns (new_colors, new_masks) if successful, else (None, None).
    """
    # This is a placeholder - we'll use SA instead
    return None, None

def _validate(colors, n):
    """Check if the coloring is valid (no monochromatic triangles)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    idx = _build_idx(n)
    for u in range(n):
        for v in range(u + 1, n):
            c1 = colors[idx[u][v]]
            for w in range(v + 1, n):
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    return False
    return True

def _try_extend(base_n, target_n, base_colors, rng, deadline):
    """
    Try to extend a valid coloring of K_{base_n} to K_{target_n}.
    Uses SA with the base coloring as initialization.
    """
    if base_n >= target_n:
        return base_colors[:]
    
    m_target = target_n * (target_n - 1) // 2
    m_base = base_n * (base_n - 1) // 2
    
    # Create initial coloring: base + random for new edges
    init_colors = base_colors[:]
    while len(init_colors) < m_target:
        init_colors.append(rng.randint(0, 3))
    
    # Run SA with this initialization
    result = _solve_n(target_n, rng, deadline, init_colors=init_colors, max_restarts=3)
    return result

def construct():
    start = time.time()
    deadline = start + 115.0
    rng = random.Random(42)
    
    # Start with baseline
    best = _baseline32()
    best_n = 32
    
    # Try to extend from 32 to larger n
    current_n = 33
    while current_n <= 62:
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time for this n
        remaining = deadline - time.time() - 2.0
        if remaining < 3.0:
            break
        
        sub_deadline = time.time() + min(remaining * 0.7, 15.0)
        
        res = None
        
        # Strategy 1: Try to extend from best known
        if best_n >= 32:
            res = _try_extend(best_n, current_n, best['colors'], rng, sub_deadline)
        
        # Strategy 2: Random initialization
        if res is None and time.time() < deadline - 3.0:
            res = _solve_n(current_n, rng, deadline - 1.0, max_restarts=5)
        
        if res is not None:
            # Validate
            if _validate(res, current_n):
                best = {'n': current_n, 'colors': res}
                best_n = current_n
                current_n += 1
            else:
                pass
        else:
            # Failed to find coloring for current_n
            # Try a couple more times with different seeds
            success = False
            for attempt in range(3):
                if time.time() > deadline - 3.0:
                    break
                rng2 = random.Random(42 + attempt * 1000)
                res2 = _solve_n(current_n, rng2, deadline - 1.0, max_restarts=5)
                if res2 is not None and _validate(res2, current_n):
                    best = {'n': current_n, 'colors': res2}
                    best_n = current_n
                    current_n += 1
                    success = True
                    break
            if not success:
                break
    
    # Final validation
    if time.time() < deadline - 1.0:
        n = best['n']
        colors = best['colors']
        if not _validate(colors, n):
            best = _baseline32()
    
    return best

# EVOLVE-BLOCK-END
