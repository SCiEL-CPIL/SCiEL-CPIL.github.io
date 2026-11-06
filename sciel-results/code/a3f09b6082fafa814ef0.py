"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _mul(a, b):
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_k32_colors():
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return colors

def _edge_index(n, i, j):
    """Return the linear index of edge (i,j) in a K_n coloring stored in lex order."""
    if i > j:
        i, j = j, i
    # Index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _is_valid_incremental(base_colors, base_n, new_vertex_edges):
    """
    Check if adding a new vertex (index base_n) with given edge colors to 0..base_n-1
    creates any monochromatic triangle.
    """
    n = base_n + 1
    # For each color c, get the set of vertices connected to new vertex with color c
    # Then check if any pair in that set has edge color c in the base
    for c in range(4):
        verts_c = [i for i in range(base_n) if new_vertex_edges[i] == c]
        for a in range(len(verts_c)):
            for b in range(a + 1, len(verts_c)):
                i, j = verts_c[a], verts_c[b]
                idx = _edge_index(base_n, i, j)
                if base_colors[idx] == c:
                    return False
    return True

def _try_add_vertex(current_colors, current_n, rng, deadline, max_attempts=20000):
    """
    Try to add one vertex to the current coloring using random restarts.
    Returns (new_colors, new_n) on success, (None, current_n) on failure.
    """
    new_vertex = current_n
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        new_edges = [rng.randint(0, 3) for _ in range(current_n)]
        if _is_valid_incremental(current_colors, current_n, new_edges):
            new_colors = current_colors + new_edges
            return new_colors, current_n + 1
    return None, current_n

def _local_search_improve(colors, n, rng, deadline):
    """
    Try to improve the coloring by flipping some edge colors to allow further extension.
    This is a heuristic: we try to find a coloring of K_n that is "more extendable".
    """
    if time.time() > deadline:
        return colors, n
    
    cur_colors = colors[:]
    cur_n = n
    
    # Try to flip random edges and see if we can extend further
    # This is a simple hill-climbing: flip an edge, try to extend, keep if it helps
    for step in range(100):
        if time.time() > deadline:
            break
        
        # Pick a random edge to flip
        i = rng.randint(0, cur_n - 2)
        j = rng.randint(i + 1, cur_n - 1)
        idx = _edge_index(cur_n, i, j)
        old_color = cur_colors[idx]
        new_color = rng.randint(0, 3)
        if new_color == old_color:
            continue
        
        cur_colors[idx] = new_color
        
        # Check if this is still valid (no monochromatic triangles)
        # Quick check: only need to check triangles involving edge (i,j)
        valid = True
        for k in range(cur_n):
            if k == i or k == j:
                continue
            c_ik = cur_colors[_edge_index(cur_n, i, k)]
            c_jk = cur_colors[_edge_index(cur_n, j, k)]
            if c_ik == c_jk == new_color:
                valid = False
                break
        
        if not valid:
            cur_colors[idx] = old_color
            continue
        
        # Try to extend this coloring
        test_colors, test_n = cur_colors[:], cur_n
        extended = False
        for target in range(cur_n + 1, cur_n + 5):
            if time.time() > deadline:
                break
            new_c, new_n = _try_add_vertex(test_colors, test_n, rng, deadline, max_attempts=500)
            if new_c is None:
                break
            test_colors = new_c
            test_n = new_n
            extended = True
        
        if extended and test_n > cur_n:
            cur_colors = test_colors
            cur_n = test_n
        else:
            # Revert the flip
            cur_colors[idx] = old_color
    
    return cur_colors, cur_n

def construct():
    deadline = time.time() + 140  # Safety margin: 10s before 150s kill
    rng = random.Random(42)
    
    # Start with 32-vertex coloring
    base_colors = _build_k32_colors()
    best_colors = base_colors[:]
    best_n = 32
    
    # Try to extend to larger n using random restarts
    current_colors = base_colors[:]
    current_n = 32
    
    while current_n < 62 and time.time() < deadline - 10:
        new_colors, new_n = _try_add_vertex(current_colors, current_n, rng, deadline, max_attempts=50000)
        if new_colors is None:
            # Try local search to find a more extendable coloring
            if time.time() < deadline - 20:
                improved_colors, improved_n = _local_search_improve(current_colors, current_n, rng, deadline)
                if improved_n > current_n:
                    current_colors = improved_colors
                    current_n = improved_n
                    # Try extending again
                    continue
            break
        
        current_colors = new_colors
        current_n = new_n
        
        if current_n > best_n:
            best_colors = current_colors[:]
            best_n = current_n
    
    # Try multiple random restarts for the base extension
    if time.time() < deadline - 5:
        for seed in range(1, 20):
            if time.time() > deadline - 2:
                break
            rng2 = random.Random(seed * 7919)
            cur_c = base_colors[:]
            cur_n = 32
            for _ in range(10):
                if time.time() > deadline - 2:
                    break
                new_c, new_n = _try_add_vertex(cur_c, cur_n, rng2, deadline, max_attempts=10000)
                if new_c is None:
                    break
                cur_c = new_c
                cur_n = new_n
            
            if cur_n > best_n:
                best_colors = cur_c[:]
                best_n = cur_n
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
