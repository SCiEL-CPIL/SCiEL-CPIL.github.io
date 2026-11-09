"""Incremental C4-free 5-coloring via cost-minimizing edge assignment + local search."""
import time
import random

_START_TIME = time.time()
DEADLINE = 145.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _get_idx(i, j, n):
    """Flat index for edge (i,j), i<j."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate(n, colors):
    """Check no monochromatic C4. True if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    adj = [0] * (5 * n)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            c = colors[idx]
            adj[c * n + a] |= (1 << b)
            adj[c * n + b] |= (1 << a)
            idx += 1
    for c in range(5):
        base = c * n
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[base + u] & adj[base + v]
                if common & (common - 1):
                    return False
    return True

def _count_violations(n, colors):
    """Count C4 violations using bitmasks."""
    adj = [0] * (5 * n)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            c = colors[idx]
            adj[c * n + a] |= (1 << b)
            adj[c * n + b] |= (1 << a)
            idx += 1
    total = 0
    for c in range(5):
        base = c * n
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[base + u] & adj[base + v]
                if common & (common - 1):
                    total += 1
    return total

def _construct_lazebnik_woldar(n):
    """
    Lazebnik-Woldar construction for n = q^2 + 1 using affine plane AG(2,q).
    For q=5, n=26. Vertices 0..24 are points in GF(5)^2, vertex 25 is infinity.
    Colors: 5 slopes (0,1,2,3,4) for finite-finite edges.
    Edges to infinity get color based on the 'row' (y-coordinate) or a fixed mapping.
    
    Actually the standard construction: color of edge (p,q) for finite points is
    the slope (q_y - p_y)/(q_x - p_x) mod 5, and edges to infinity get color
    based on... Let me use a simpler valid construction.
    
    For AG(2,5): 25 points, 5 parallel classes of 5 lines each = 25 lines per class,
    5 classes. Each class partitions the 25 points into 5 lines of 5.
    Color a finite-finite edge by which parallel class the line through them belongs to.
    This gives a C4-free coloring because any C4 in one color would be a parallelogram,
    but in AG(2,q) there are no parallelograms in a single parallel class... 
    Actually, in AG(2,q), a C4 in one parallel class would require 4 points forming a 
    parallelogram with all 4 edges in the same slope class, which is impossible since
    a parallel class has exactly one line through each pair.
    
    Wait, that's not right. Let me think again. In AG(2,q), the lines of one parallel
    class are disjoint. So if we color edge (u,v) by the parallel class of the line
    containing u and v, then a monochromatic C4 would need 4 vertices u,v,w,x such
    that all 4 edges uv, vw, wx, xu are in the same parallel class. But in a parallel
    class, lines are disjoint, so u and v are on one line, v and w on another line
    of the same class - but these two lines share vertex v, contradiction since lines
    in a parallel class are disjoint. So this is C4-free!
    
    For n=26, we add vertex 25 (infinity). We need to color edges (25, i) for i=0..24
    with colors 0..4 such that no C4 is formed. This is tricky. Let's just use the
    25-point construction and try to extend.
    """
    if n != 26:
        return None
    
    # Vertices 0..24: map to (x, y) in GF(5)^2
    # Vertex 25: infinity
    
    # Parallel classes: slopes 0,1,2,3,4 (including "vertical" as slope 4 or use a 5th)
    # Actually in GF(5), slopes are 0,1,2,3,4 (all elements) plus infinity for vertical.
    # That's 6 slopes. We only have 5 colors.
    # 
    # Alternative: use 5 parallel classes by merging two slopes.
    # Or: use the fact that for q=5, we can use a different decomposition.
    #
    # Simpler approach: just use a known valid 25-coloring and extend to 26.
    # For now, return None and use the greedy approach.
    
    # Actually, let's just do a simple modular construction and hope it works
    colors = [0] * (n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            if i < 25 and j < 25:
                # Finite-finite: use slope
                x1, y1 = i % 5, i // 5
                x2, y2 = j % 5, j // 5
                dx = (x2 - x1) % 5
                dy = (y2 - y1) % 5
                if dx == 0:
                    slope = 4  # vertical, map to color 4
                else:
                    slope = (dy * pow(dx, 3, 5)) % 5  # dy/dx mod 5
                colors[_get_idx(i, j, n)] = slope
            else:
                # Edge to infinity (vertex 25)
                # Color based on the finite vertex's x-coordinate
                finite = i if j == 25 else j
                colors[_get_idx(i, j, n)] = finite % 5
    
    return colors

def _greedy_build(n, deadline, seed=42):
    """
    Build coloring edge-by-edge using incremental common-neighbor tracking.
    For each edge, choose the color that minimizes the number of new at-risk pairs
    (pairs with exactly 1 common neighbor in that color) while avoiding C4.
    """
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    colors = [-1] * n_edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # common[c][u][v] = number of common neighbors of u,v in color c
    common = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    for idx, (i, j) in enumerate(edges):
        if time.time() > deadline:
            return None
        
        best_color = -1
        best_cost = float('inf')
        valid_colors = []
        
        for c in range(5):
            if common[c][i][j] >= 2:
                continue  # Would create C4
            
            # Compute cost: number of pairs that go from 0 to 1 common neighbor
            cost = 0
            feasible = True
            for k in range(n):
                if k == i or k == j:
                    continue
                idx_ik = _get_idx(i, k, n)
                idx_jk = _get_idx(j, k, n)
                
                if colors[idx_ik] == c:
                    # (i,k) is color c, so adding (i,j) color c makes j a common neighbor of i,k
                    if common[c][j][k] >= 1:
                        feasible = False
                        break
                    # common[c][j][k] was 0, becomes 1 -> at-risk
                    cost += 1
                
                if colors[idx_jk] == c:
                    # (j,k) is color c, so adding (i,j) color c makes i a common neighbor of j,k
                    if common[c][i][k] >= 1:
                        feasible = False
                        break
                    # common[c][i][k] was 0, becomes 1 -> at-risk
                    cost += 1
            
            if feasible:
                valid_colors.append((cost, c))
        
        if not valid_colors:
            return None
        
        valid_colors.sort()
        min_cost = valid_colors[0][0]
        candidates = [c for (cost, c) in valid_colors if cost == min_cost]
        best_color = rng.choice(candidates)
        colors[idx] = best_color
        c = best_color
        
        # Update common counts
        for k in range(n):
            if k == i or k == j:
                continue
            idx_ik = _get_idx(i, k, n)
            idx_jk = _get_idx(j, k, n)
            
            if colors[idx_ik] == c:
                common[c][j][k] += 1
                common[c][k][j] += 1
            if colors[idx_jk] == c:
                common[c][i][k] += 1
                common[c][k][i] += 1
    
    if _validate(n, colors):
        return colors
    return None

def _local_search(n, colors, deadline):
    """Local search: try recoloring edges to reduce C4 violations."""
    colors = list(colors)
    current_viol = _count_violations(n, colors)
    if current_viol == 0:
        return colors
    
    best_viol = current_viol
    best_colors = colors[:]
    n_edges = n * (n - 1) // 2
    
    while time.time() < deadline:
        eidx = random.randint(0, n_edges - 1)
        old_color = colors[eidx]
        new_color = (old_color + 1 + random.randint(0, 3)) % 5
        
        colors[eidx] = new_color
        new_viol = _count_violations(n, colors)
        
        if new_viol <= current_viol:
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
            if current_viol == 0:
                break
        else:
            colors[eidx] = old_color
    
    return best_colors

def construct():
    random.seed(42)
    
    # Start with a small valid coloring
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try from high n down
    for target_n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if _time_left() < 10.0:
            break
        if target_n <= best_n:
            break
        
        found = None
        
        # Try Lazebnik-Woldar construction first for n=26
        if target_n == 26:
            lw = _construct_lazebnik_woldar(26)
            if lw is not None:
                refined = _local_search(26, lw, time.time() + 5)
                if _validate(26, refined):
                    found = refined
        
        # Try greedy build with multiple seeds
        if found is None:
            for seed in [42, 123, 777, 2024, 555, 888]:
                if _time_left() < 8.0:
                    break
                result = _greedy_build(target_n, time.time() + 8, seed)
                if result is not None:
                    found = result
                    break
        
        # If greedy failed, try random + local search
        if found is None:
            for attempt in range(10):
                if _time_left() < 6.0:
                    break
                n_edges = target_n * (target_n - 1) // 2
                colors = [random.randint(0, 4) for _ in range(n_edges)]
                refined = _local_search(target_n, colors, time.time() + 4)
                if _validate(target_n, refined):
                    found = refined
                    break
        
        if found is not None:
            best_n = target_n
            best_colors = found
    
    return {'n': best_n, 'colors': best_colors}
