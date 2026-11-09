"""Four-colourings of K_n without monochromatic triangles.

Strategy:
- Primary: Construct 49-vertex coloring using GF(7^2) with multiplicative
  group partitioned into cosets of cubes (order 48, divisible by 3).
- Fallback: 32-vertex Clebsch-based construction if GF(7^2) fails.
- Extension: Backtracking + local-search repair to add vertices beyond base.
- Deadline guard: return best valid result before 120 s.
"""

import time
import random
import sys
import copy

# ---------------------------------------------------------------------------
# GF(7^2) construction for 49 vertices
# ---------------------------------------------------------------------------

def _gf49_construction():
    """
    Construct a 4-coloring of K_49 using GF(7^2).
    
    GF(7^2) has 49 elements. The multiplicative group has order 48.
    Since 48 is divisible by 3, we partition the nonzero elements into
    3 cosets of the subgroup of cubes (order 16).
    
    For vertices x, y in GF(49):
    - color based on which coset of cubes (x-y) belongs to
    
    We use colors 0,1,2 for the 3 cosets. Color 3 is unused in the base
    but available for extensions.
    """
    # GF(7^2) = GF(7)[x] / (x^2 + 1), alpha^2 = -1 = 6 mod 7
    
    def gf49_add(x, y):
        return ((x[0] + y[0]) % 7, (x[1] + y[1]) % 7)
    
    def gf49_sub(x, y):
        return ((x[0] - y[0]) % 7, (x[1] - y[1]) % 7)
    
    def gf49_mul(x, y):
        a, b = x
        c, d = y
        real = (a * c + 6 * b * d) % 7
        imag = (a * d + b * c) % 7
        return (real, imag)
    
    elements = [(a, b) for a in range(7) for b in range(7)]
    
    def gf49_pow(x, n):
        result = (1, 0)
        base = x
        while n > 0:
            if n & 1:
                result = gf49_mul(result, base)
            base = gf49_mul(base, base)
            n >>= 1
        return result
    
    # Find primitive element (order 48)
    primitive = None
    for e in elements:
        if e == (0, 0):
            continue
        if gf49_pow(e, 24) != (1, 0) and gf49_pow(e, 16) != (1, 0):
            primitive = e
            break
    
    if primitive is None:
        return None
    
    # Generate powers
    powers = []
    current = (1, 0)
    for i in range(48):
        powers.append(current)
        current = gf49_mul(current, primitive)
    
    # Cubes subgroup (order 16)
    cubes = [powers[3 * i] for i in range(16)]
    cube_set = set(cubes)
    
    # Precompute coset lookup for all nonzero elements
    coset_lookup = {}
    inv_prim = powers[47]  # primitive^(-1)
    for p in powers:
        if p in cube_set:
            coset_lookup[p] = 0
        else:
            e1 = gf49_mul(p, inv_prim)
            if e1 in cube_set:
                coset_lookup[p] = 1
            else:
                coset_lookup[p] = 2
    
    # Build 2D color matrix
    n = 49
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            diff = gf49_sub(elements[i], elements[j])
            if diff == (0, 0):
                c = 0
            else:
                c = coset_lookup[diff]
            graph[i][j] = c
            graph[j][i] = c
    
    # Convert to flat list
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(graph[i][j])
    
    return colors, graph

# ---------------------------------------------------------------------------
# Fast triangle-free verification using bitsets
# ---------------------------------------------------------------------------

def _is_triangle_free_fast(colors, n):
    """
    colors: flat list in lex order.
    Returns True iff no monochromatic triangle.
    """
    # Build adjacency bitmasks
    adj = [[0] * n for _ in range(4)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    for c in range(4):
        for v in range(n):
            nbrs = adj[c][v]
            n1 = nbrs
            while n1:
                b1 = n1 & (-n1)
                n1 -= b1
                u = b1.bit_length() - 1
                if adj[c][u] & nbrs & ~(1 << v):
                    return False
    return True

def _is_triangle_free_graph(graph, n):
    """Check triangle-free using 2D graph matrix."""
    adj = [[0] * n for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = graph[i][j]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    
    for c in range(4):
        for v in range(n):
            nbrs = adj[c][v]
            n1 = nbrs
            while n1:
                b1 = n1 & (-n1)
                n1 -= b1
                u = b1.bit_length() - 1
                if adj[c][u] & nbrs & ~(1 << v):
                    return False
    return True

# ---------------------------------------------------------------------------
# 32-vertex Clebsch base
# ---------------------------------------------------------------------------

def _base32():
    """Return (colors, graph) for the 32-vertex Clebsch-doubled coloring."""
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
    
    n = 32
    graph = [[0] * n for _ in range(n)]
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 16 == j // 16:
                c = log[(i % 16) ^ (j % 16)] % 3
            else:
                c = 3
            graph[i][j] = c
            graph[j][i] = c
            colors.append(c)
    
    return colors, graph

# ---------------------------------------------------------------------------
# Extension: add vertices via backtracking with local-search repair
# ---------------------------------------------------------------------------

def _try_add_vertex(graph, n, deadline):
    """
    Try to add a new vertex (index n) to the current coloring.
    Returns (success, new_graph) where new_graph has n+1 vertices.
    Uses backtracking with pruning; if backtracking fails, tries local search.
    """
    new_n = n + 1
    new_graph = [row[:] + [0] for row in graph]
    new_graph.append([0] * new_n)
    
    # Precompute for backtracking: for each vertex i, the set of colors
    # that are "forbidden" given current assignments
    assignments = [0] * n
    
    def backtrack(i, depth):
        if time.time() > deadline:
            return False
        if i == n:
            # Verify no monochromatic triangle among new edges
            for a in range(n):
                for b in range(a + 1, n):
                    if graph[a][b] == assignments[a] == assignments[b]:
                        return False
            return True
        
        # Try each color
        for c in range(4):
            assignments[i] = c
            # Prune: check triangles (k, i, new) for k < i
            valid = True
            for k in range(i):
                if graph[k][i] == assignments[k] == c:
                    valid = False
                    break
            if valid:
                if backtrack(i + 1, depth + 1):
                    return True
        return False
    
    if backtrack(0, 0):
        for i in range(n):
            new_graph[i][n] = assignments[i]
            new_graph[n][i] = assignments[i]
        return True, new_graph
    
    # Backtracking failed - try local search repair
    # Randomly assign colors, then iteratively fix violations
    if time.time() > deadline - 2:
        return False, None
    
    for attempt in range(50):
        if time.time() > deadline - 1:
            break
        
        # Random assignment
        for i in range(n):
            assignments[i] = random.randint(0, 3)
        
        # Local search: find and fix monochromatic triangles
        for it in range(2000):
            if time.time() > deadline - 1:
                break
            
            # Find a violation
            found_violation = False
            for a in range(n):
                for b in range(a + 1, n):
                    c_ab = graph[a][b]
                    if assignments[a] == c_ab and assignments[b] == c_ab:
                        # Violation: triangle (a, b, new) is monochromatic
                        # Flip one of the two edges
                        if random.random() < 0.5:
                            assignments[a] = (assignments[a] + 1) % 4
                        else:
                            assignments[b] = (assignments[b] + 1) % 4
                        found_violation = True
                        break
                if found_violation:
                    break
            
            if not found_violation:
                # No violations - check full validity
                valid = True
                for a in range(n):
                    for b in range(a + 1, n):
                        if graph[a][b] == assignments[a] == assignments[b]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for i in range(n):
                        new_graph[i][n] = assignments[i]
                        new_graph[n][i] = assignments[i]
                    return True, new_graph
    
    return False, None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, safety margin for 120s limit
    
    # Try GF(49) construction first
    best_colors = None
    best_graph = None
    best_n = 0
    
    gf49_result = _gf49_construction()
    if gf49_result is not None:
        colors, graph = gf49_result
        if _is_triangle_free_graph(graph, 49):
            best_colors = colors
            best_graph = graph
            best_n = 49
    
    # Fallback to 32-vertex
    if best_n < 32:
        colors, graph = _base32()
        best_colors = colors
        best_graph = graph
        best_n = 32
    
    # Try to extend beyond current best_n
    if best_n < 62 and time.time() < deadline - 10:
        graph = best_graph
        n = best_n
        
        # Try to add vertices one by one
        consecutive_failures = 0
        max_consecutive_failures = 20
        
        while n < 62 and time.time() < deadline - 5:
            if consecutive_failures >= max_consecutive_failures:
                break
            
            success, new_graph = _try_add_vertex(graph, n, deadline - 2)
            
            if success:
                graph = new_graph
                n += 1
                best_n = n
                # Update best_colors
                best_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        best_colors.append(graph[i][j])
                best_graph = graph
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                # Occasionally try a "kick": randomly modify some edges
                # and see if we can still add a vertex
                if consecutive_failures % 5 == 0 and time.time() < deadline - 10:
                    kick_graph = copy.deepcopy(graph)
                    num_changes = max(1, n * n // 40)
                    for _ in range(num_changes):
                        i = random.randint(0, n - 1)
                        j = random.randint(0, n - 1)
                        if i != j:
                            kick_graph[i][j] = random.randint(0, 3)
                            kick_graph[j][i] = kick_graph[i][j]
                    
                    # Check if kick_graph is still valid
                    if _is_triangle_free_graph(kick_graph, n):
                        # Try to add vertex to kick_graph
                        success2, new_graph2 = _try_add_vertex(kick_graph, n, deadline - 2)
                        if success2:
                            graph = new_graph2
                            n += 1
                            best_n = n
                            best_colors = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    best_colors.append(graph[i][j])
                            best_graph = graph
                            consecutive_failures = 0
    
    # Final deadline check
    if time.time() > deadline:
        pass  # Just return what we have
    
    # Ensure we have a valid result
    if best_colors is None:
        # Absolute fallback: n=1
        best_n = 1
        best_colors = []
    
    return {'n': best_n, 'colors': best_colors}
