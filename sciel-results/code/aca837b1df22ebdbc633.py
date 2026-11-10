"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

N = 16
NEDGES = N*(N-1)//2

def _edge_idx(i, j):
    """Map pair (i,j) with i<j to lexicographic edge index."""
    # Edges in lex order: (0,1),(0,2),...,(0,15),(1,2),...,(14,15)
    # For vertex i, edges to j=i+1..N-1
    # Count edges before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    return i*(N-1) - i*(i-1)//2 + (j - i - 1)

def _check_rainbow(colors):
    """Check no 4-cycle has 4 distinct colors. Returns True if valid."""
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}
        # Cycle 1: a-b-c-d-a
        e1 = (colors[_edge_idx(a,b)], colors[_edge_idx(b,c)],
              colors[_edge_idx(c,d)], colors[_edge_idx(d,a)])
        if len(set(e1)) == 4:
            return False
        # Cycle 2: a-b-d-c-a
        e2 = (colors[_edge_idx(a,b)], colors[_edge_idx(b,d)],
              colors[_edge_idx(c,d)], colors[_edge_idx(a,c)])
        if len(set(e2)) == 4:
            return False
        # Cycle 3: a-c-b-d-a
        e3 = (colors[_edge_idx(a,c)], colors[_edge_idx(b,c)],
              colors[_edge_idx(b,d)], colors[_edge_idx(a,d)])
        if len(set(e3)) == 4:
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _try_parametric(family, params):
    """Try a parametric coloring family, return (colors, distinct) or None."""
    colors = [0] * NEDGES
    for i in range(N):
        for j in range(i+1, N):
            colors[_edge_idx(i,j)] = family(i, j, params)
    if not _check_rainbow(colors):
        return None
    return colors, _count_distinct(colors)

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    best_colors = [0] * NEDGES
    best_distinct = 0
    
    # Family 1: color(i,j) = (a*i + b*j + c*i*j + d) % M
    for M in range(2, 120):
        if time.time() > deadline:
            break
        # Try small coefficients
        for a in range(0, min(M, 8)):
            if time.time() > deadline:
                break
            for b in range(0, min(M, 8)):
                if time.time() > deadline:
                    break
                for c in range(0, min(M, 4)):
                    if time.time() > deadline:
                        break
                    def fam(i, j, p, a=a, b=b, c=c, M=M):
                        return (a*i + b*j + c*i*j) % M
                    res = _try_parametric(fam, None)
                    if res:
                        colors, distinct = res
                        if distinct > best_distinct:
                            best_distinct = distinct
                            best_colors = colors[:].copy()
    
    # Family 2: color(i,j) = (i*j) % M
    for M in range(2, 120):
        if time.time() > deadline:
            break
        def fam(i, j, p, M=M):
            return (i*j) % M
        res = _try_parametric(fam, None)
        if res:
            colors, distinct = res
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:].copy()
    
    # Family 3: color(i,j) = (i + j) % M
    for M in range(2, 120):
        if time.time() > deadline:
            break
        def fam(i, j, p, M=M):
            return (i + j) % M
        res = _try_parametric(fam, None)
        if res:
            colors, distinct = res
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:].copy()
    
    # Family 4: color(i,j) = (i*i + j*j + i*j) % M
    for M in range(2, 120):
        if time.time() > deadline:
            break
        def fam(i, j, p, M=M):
            return (i*i + j*j + i*j) % M
        res = _try_parametric(fam, None)
        if res:
            colors, distinct = res
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:].copy()
    
    # Family 5: group-based. Partition into 4 groups of 4.
    # Within group g: use 6 distinct colors offset by g*6
    # Between group g1 and g2: use colors based on (g1,g2) pair
    if time.time() < deadline:
        group_size = 4
        ngroups = N // group_size
        colors = [0] * NEDGES
        # Within-group edges
        for g in range(ngroups):
            start = g * group_size
            for di in range(group_size):
                for dj in range(di+1, group_size):
                    i = start + di
                    j = start + dj
                    colors[_edge_idx(i,j)] = g * 6 + (di*3 + dj - 1)  # 6 distinct per group
        # Between-group edges
        for g1 in range(ngroups):
            for g2 in range(g1+1, ngroups):
                base = g1 * ngroups + g2  # 0..6
                # 16 edges between two groups of 4
                for di in range(group_size):
                    for dj in range(group_size):
                        i = g1 * group_size + di
                        j = g2 * group_size + dj
                        # Use base + di + dj*4 % something... let's use distinct per pair
                        colors[_edge_idx(i,j)] = base * 16 + di + dj * 4
        if _check_rainbow(colors):
            distinct = _count_distinct(colors)
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:].copy()
    
    # Family 6: color(i,j) = (i * (j - i)) % M
    for M in range(2, 120):
        if time.time() > deadline:
            break
        def fam(i, j, p, M=M):
            return (i * (j - i)) % M
        res = _try_parametric(fam, None)
        if res:
            colors, distinct = res
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:].copy()
    
    # Family 7: color(i,j) = (i * j + i) % M
    for M in range(2, 120):
        if time.time() > deadline:
            break
        def fam(i, j, p, M=M):
            return (i * j + i) % M
        res = _try_parametric(fam, None)
        if res:
            colors, distinct = res
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:].copy()
    
    # Family 8: color(i,j) = ((i+1) * (j+1)) % M
    for M in range(2, 120):
        if time.time() > deadline:
            break
        def fam(i, j, p, M=M):
            return ((i+1) * (j+1)) % M
        res = _try_parametric(fam, None)
        if res:
            colors, distinct = res
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:].copy()
    
    # Ensure all colors are in 0..119
    best_colors = [c % 120 for c in best_colors]
    
    return best_colors

# EVOLVE-BLOCK-END
