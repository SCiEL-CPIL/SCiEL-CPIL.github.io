"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _has_c4_in_color(color, adj, n):
    """Check if color class has a C4. adj[color] is list of sets of neighbors per vertex."""
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[color][u]:
                common = adj[color][u] & adj[color][v]
                if len(common) >= 2:
                    return True
    return False

def _validate_coloring(n, colors):
    """Validate that no color class contains a C4."""
    adj = [defaultdict(set) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    for c in range(5):
        if _has_c4_in_color(c, adj, n):
            return False
    return True

def _lazebnik_woldar_26():
    """
    Lazebnik-Woldar construction for R_5(C4) >= 26.
    Uses finite field F_26 = GF(2^4) * GF(13)? Actually 26 = 2*13.
    
    The construction: vertices are pairs (a, b) where a in Z_2, b in Z_13.
    Edge coloring based on differences.
    
    Actually, the standard LW construction for 5 colors and 26 vertices:
    V = Z_2 x Z_13
    For edge {(a,b), (c,d)}, let delta = (a-c mod 2, b-d mod 13)
    Color based on which of 5 sets the delta belongs to.
    
    The 5 color classes correspond to:
    - delta_1 = 0 (a=c): then color by (b-d) mod 13 in specific subsets
    - delta_1 != 0: color by (b-d) mod 13 in other subsets
    
    Let me use a known explicit construction:
    Colors 0-4 partition Z_13^* into 5 sets of sizes... 13 is prime.
    Actually 12 nonzero elements in Z_13, so we can't split evenly into 5.
    
    Let me try a different approach: use the known result that R_5(C4) >= 26.
    I'll implement a greedy construction targeting n=26 first, then try to extend.
    """
    n = 26
    colors = []
    # Use a greedy approach with the constraint that no C4 in any color
    # Start with all edges uncolored, assign greedily
    adj = [defaultdict(set) for _ in range(5)]
    
    # Precompute: for each pair of vertices and each color, check if adding would create C4
    # This is expensive for n=26, so use a smarter approach
    
    # Actually, let me try the cyclic-like construction on Z_26
    # But Z_26 is not prime, so cyclic won't work well.
    
    # Let me try: vertices 0..25, color edge (i,j) based on (j-i mod 26)
    # Map each nonzero residue mod 26 to one of 5 colors
    # There are 25 nonzero residues, so 5 colors with 5 residues each
    
    # Residues mod 26: 1,2,...,25
    # Assign: color c gets residues {5c+1, 5c+2, 5c+3, 5c+4, 5c+5}
    # But this creates C4s likely.
    
    # Better: use the fact that we need C4-free graphs.
    # A C4-free graph on 26 vertices can have at most ~26^(3/2)/2 ≈ 66 edges.
    # K_26 has 325 edges, so each color class has ~65 edges on average.
    
    # Let me try a random greedy with backtracking for small n, 
    # and for n=26 use a more structured approach.
    
    # Structured approach: use Z_2 x Z_13
    # Vertex = (a, b) where a in {0,1}, b in Z_13
    # Map vertex index v to (v % 2, v // 2)
    
    def v_to_ab(v):
        return (v % 2, v // 2)
    
    def ab_to_v(a, b):
        return a + 2 * b
    
    # For edge between (a,b) and (c,d):
    # delta_a = (a - c) % 2, delta_b = (b - d) % 13
    # If delta_a == 0: color based on delta_b
    # If delta_a == 1: color based on delta_b
    
    # We need to partition the 25 nonzero pairs (da, db) into 5 groups of 5
    # such that each group forms a C4-free graph.
    
    # The 25 nonzero deltas: (0,1)..(0,12), (1,0)..(1,12)
    # Group 0: (0,1), (0,2), (0,3), (0,4), (0,5)
    # Group 1: (0,6), (0,7), (0,8), (0,9), (0,10)
    # Group 2: (0,11), (0,12), (1,0), (1,1), (1,2)
    # Group 3: (1,3), (1,4), (1,5), (1,6), (1,7)
    # Group 4: (1,8), (1,9), (1,10), (1,11), (1,12)
    
    # Let me try a different partition that might avoid C4s
    # Use quadratic residues mod 13: QR = {1, 3, 4, 9, 10, 12}
    # Non-QR = {2, 5, 6, 7, 8, 11}
    
    # 6 QR, 6 NQR, and 1 zero.
    # So we have:
    # da=0, db in QR: 6 edges
    # da=0, db in NQR: 6 edges  
    # da=0, db=0: 1 edge (but db=0 with da=0 means same vertex, skip)
    # da=1, db in QR: 6 edges
    # da=1, db in NQR: 6 edges
    # da=1, db=0: 1 edge
    
    # Total: 6+6+6+6+1 = 25 nonzero deltas. Good.
    
    # Assign:
    # Color 0: (0, QR) = 6 deltas
    # Color 1: (0, NQR) = 6 deltas
    # Color 2: (1, QR) = 6 deltas
    # Color 3: (1, NQR) = 6 deltas
    # Color 4: (1, 0) = 1 delta... only 1 edge, too few.
    
    # Let me rebalance:
    # Color 0: (0, QR) = 6
    # Color 1: (0, NQR) = 6
    # Color 2: (1, QR) = 6
    # Color 3: (1, NQR) = 6
    # Color 4: (1, 0) + some from others?
    
    # This doesn't balance well. Let me just use a simple assignment and validate.
    
    QR = {1, 3, 4, 9, 10, 12}
    NQR = {2, 5, 6, 7, 8, 11}
    
    def get_color(da, db):
        if da == 0:
            if db == 0:
                return 0  # shouldn't happen for distinct vertices
            if db in QR:
                return 0
            else:
                return 1
        else:  # da == 1
            if db == 0:
                return 4
            if db in QR:
                return 2
            else:
                return 3
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            a1, b1 = v_to_ab(i)
            a2, b2 = v_to_ab(j)
            da = (a1 - a2) % 2
            db = (b1 - b2) % 13
            c = get_color(da, db)
            colors.append(c)
    
    if _validate_coloring(n, colors):
        return n, colors
    
    # If validation fails, fall back to greedy for smaller n
    return _greedy_construct(15)

def _greedy_construct(target_n):
    """Greedy edge coloring trying to reach target_n."""
    n = target_n
    colors = [-1] * (n * (n - 1) // 2)
    adj = [defaultdict(set) for _ in range(5)]
    
    def idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def can_add(i, j, c):
        # Check if adding edge (i,j) with color c creates a C4
        for k in adj[c][i]:
            if k in adj[c][j]:
                return False
        for k in adj[c][j]:
            if k in adj[c][i]:
                return False
        # Also check: for any pair (u,v) both adjacent to i and j in color c
        common = adj[c][i] & adj[c][j]
        if len(common) >= 1:
            return False
        return True
    
    # Process edges in order
    success = True
    for i in range(n):
        for j in range(i + 1, n):
            placed = False
            for c in range(5):
                if can_add(i, j, c):
                    colors[idx(i, j)] = c
                    adj[c][i].add(j)
                    adj[c][j].add(i)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if not success:
            break
    
    if success:
        return n, colors
    else:
        # Return what we have
        return _partial_result(n, colors, adj)

def _partial_result(n, colors, adj):
    """Extract the largest valid prefix."""
    # Find the largest m such that K_m is valid
    best_m = 1
    for m in range(2, n + 1):
        # Check if all edges in K_m are colored and valid
        # This is expensive, so just return what we can
        pass
    # For simplicity, return n=1 if we can't validate
    return 1, []

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    # First try the LW construction for n=26
    n26, colors26 = _lazebnik_woldar_26()
    
    if time.time() > deadline - 10:
        return {'n': n26, 'colors': colors26}
    
    # Try to extend to n=27 using greedy
    n27, colors27 = _try_extend(n26, colors26, 27)
    if n27 >= 27:
        return {'n': n27, 'colors': colors27}
    
    # Try n=28
    if time.time() < deadline - 20:
        n28, colors28 = _try_extend(n26, colors26, 28)
        if n28 >= 28:
            return {'n': n28, 'colors': colors28}
    
    # Fall back to best we have
    return {'n': n26, 'colors': colors26}

def _try_extend(n_base, colors_base, n_target):
    """Try to extend a valid coloring of K_n_base to K_n_target."""
    n = n_target
    m_base = n_base * (n_base - 1) // 2
    m_target = n * (n - 1) // 2
    
    # Start with the base coloring, add new edges
    colors = list(colors_base) + [-1] * (m_target - m_base)
    
    # Rebuild adjacency for base
    adj = [defaultdict(set) for _ in range(5)]
    idx_base = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = colors[idx_base]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx_base += 1
    
    # Now add edges involving new vertices
    def idx_full(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def can_add(i, j, c):
        common = adj[c][i] & adj[c][j]
        if len(common) >= 1:
            return False
        return True
    
    # Add edges for new vertices
    for new_v in range(n_base, n):
        for old_v in range(n_base):
            placed = False
            for c in range(5):
                if can_add(old_v, new_v, c):
                    colors[idx_full(old_v, new_v)] = c
                    adj[c][old_v].add(new_v)
                    adj[c][new_v].add(old_v)
                    placed = True
                    break
            if not placed:
                return n_base, colors_base
    
        # Add edges between new vertices
        for new_v2 in range(new_v + 1, n):
            placed = False
            for c in range(5):
                if can_add(new_v, new_v2, c):
                    colors[idx_full(new_v, new_v2)] = c
                    adj[c][new_v].add(new_v2)
                    adj[c][new_v2].add(new_v)
                    placed = True
                    break
            if not placed:
                return n_base, colors_base
    
    # Validate the full coloring
    if _validate_coloring(n, colors):
        return n, colors
    
    return n_base, colors_base
# EVOLVE-BLOCK-END
