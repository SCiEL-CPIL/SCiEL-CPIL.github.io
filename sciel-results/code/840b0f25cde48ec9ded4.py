"""
Four-colourings of complete graphs without monochromatic C6.
Uses simulated annealing with conflict-focused restarts to find valid
colourings of K_n for large n, maximising n.
"""
import time
import random
import sys

# Deadline: evaluator kills at 150s; keep safety margin.
DEADLINE = 125.0  # seconds
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _build_adj(n, colors):
    """Build adjacency lists for each colour class."""
    adj = [[set() for _ in range(n)] for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    return adj


def _count_c6(adj, n):
    """Count number of (not necessarily induced) 6-cycles in each colour class.
    A 6-cycle is a set of 6 distinct vertices v0..v5 with edges
    v0-v1, v1-v2, v2-v3, v3-v4, v4-v5, v5-v0.
    We count each cycle once by fixing v0 as the smallest vertex and
    v1 < v5 (to break rotational/reflection symmetry).
    """
    total = 0
    for c in range(4):
        ac = adj[c]
        for v0 in range(n):
            nbrs0 = ac[v0]
            for v1 in nbrs0:
                if v1 <= v0:
                    continue
                for v5 in nbrs0:
                    if v5 >= v1:
                        continue
                    # Now find v2 in nbrs(v1) \ {v0}, v3 in nbrs(v2) \ {v1,v5}, v4 in nbrs(v3) \ {v2,v5}
                    # such that v4 is in nbrs(v5)
                    for v2 in ac[v1]:
                        if v2 == v0:
                            continue
                        for v3 in ac[v2]:
                            if v3 == v1 or v3 == v5:
                                continue
                            for v4 in ac[v3]:
                                if v4 == v2 or v4 == v5:
                                    continue
                                if v4 in ac[v5]:
                                    total += 1
    return total


def _count_c6_fast(adj, n):
    """Faster C6 count using matrix-free approach.
    For each colour, count 6-cycles by: for each pair (v0,v5) with v0<v5 and edge,
    count paths of length 4 from v0 to v5 through v1,v2,v3,v4 with all distinct.
    This is still expensive; use a different strategy:
    Count closed walks of length 6 and subtract, or use the fact that for sparse graphs
    this is manageable.
    
    Actually, let's use: for each colour c, for each edge (a,b), count the number of
    paths of length 5 from a to b that don't revisit a or b internally.
    A 6-cycle through edge (a,b) corresponds to a path of length 5 from a to b
    with all 6 vertices distinct.
    
    For n<=20, let's just do the direct enumeration but optimize with early termination.
    """
    total = 0
    for c in range(4):
        ac = adj[c]
        # Precompute neighbor lists
        nbrs = [list(ac[i]) for i in range(n)]
        for v0 in range(n):
            n0 = nbrs[v0]
            for v1 in n0:
                if v1 <= v0:
                    continue
                n1 = nbrs[v1]
                for v2 in n1:
                    if v2 == v0:
                        continue
                    n2 = nbrs[v2]
                    for v3 in n2:
                        if v3 == v1 or v3 == v5_placeholder if False else v3 == v0:
                            continue
                        # We need v3 != v0, v3 != v1
                        if v3 == v0 or v3 == v1:
                            continue
                        n3 = nbrs[v3]
                        for v4 in n3:
                            if v4 == v2 or v4 == v0 or v4 == v1:
                                continue
                            # Check v4 connected to v5 where v5 in n0, v5 < v1
                            # Actually we need v5 in n0 with v5 < v1 and v4 in nbrs[v5]
                            for v5 in n0:
                                if v5 >= v1 or v5 == v0:
                                    continue
                                if v4 in ac[v5]:
                                    total += 1
    return total


def _count_c6_direct(adj, n):
    """Direct enumeration of 6-cycles. For n up to 20, this is feasible.
    A 6-cycle: v0-v1-v2-v3-v4-v5-v0, all distinct.
    Fix v0 = min vertex. Then v1, v5 are neighbors of v0.
    To avoid double counting: v1 < v5.
    Then v2 in N(v1)\\{v0}, v4 in N(v5)\\{v0}, v3 in N(v2)∩N(v4)\\{v0,v1,v5}.
    """
    total = 0
    for c in range(4):
        ac = adj[c]
        nbrs = [list(ac[i]) for i in range(n)]
        for v0 in range(n):
            n0 = nbrs[v0]
            if len(n0) < 2:
                continue
            # Pairs (v1, v5) with v1 < v5, both in n0
            for ii in range(len(n0)):
                v1 = n0[ii]
                if v1 <= v0:
                    continue
                for jj in range(ii + 1, len(n0)):
                    v5 = n0[jj]
                    if v5 <= v0:
                        continue
                    # v2 in N(v1) \ {v0}
                    for v2 in nbrs[v1]:
                        if v2 == v0:
                            continue
                        # v4 in N(v5) \ {v0}
                        for v4 in nbrs[v5]:
                            if v4 == v0:
                                continue
                            # v3 in N(v2) ∩ N(v4) \ {v0, v1, v5}
                            # and v3 != v2, v3 != v4
                            common = nbrs[v2]
                            for v3 in common:
                                if v3 == v0 or v3 == v1 or v3 == v5 or v3 == v2 or v3 == v4:
                                    continue
                                if v3 in ac[v4]:
                                    total += 1
    return total


def _is_valid(n, colors):
    """Check if the coloring has no monochromatic C6."""
    adj = _build_adj(n, colors)
    return _count_c6_direct(adj, n) == 0


def _sa_search(n, max_iter=200000, restarts=30, seed=None):
    """Simulated annealing to find a valid 4-coloring of K_n with no monochromatic C6."""
    if seed is not None:
        random.seed(seed)
    
    best_conflicts = float('inf')
    best_colors = None
    
    for restart in range(restarts):
        if _time_left() < 5.0:
            break
        
        # Initialize: random coloring
        m = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        
        adj = _build_adj(n, colors)
        conflicts = _count_c6_direct(adj, n)
        
        if conflicts == 0:
            return colors
        
        T = 1.0
        T_min = 0.001
        cooling = 0.9995
        
        for it in range(max_iter):
            if _time_left() < 3.0:
                break
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_colors = colors[:]
            
            if conflicts == 0:
                return colors
            
            # Pick a random edge to recolor
            idx = random.randint(0, m - 1)
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Determine which vertices this edge connects
            # We need to map idx back to (i,j)
            # Precompute this mapping
            # For efficiency, let's precompute edge->(i,j) mapping
            # Actually, let's just recompute conflicts incrementally
            # For simplicity in this version, recompute fully (slow but correct)
            # Better: incremental update
            
            # Let's do incremental: remove old edge from adj, add new edge
            # Find (i,j) for idx
            # idx = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
            # Let's precompute
            pass
        
        # This is getting complex; let's use a simpler but effective approach:
        # Local search with random restarts, using a fast conflict check
        
    return best_colors


def _edge_to_idx(i, j, n):
    """Map (i,j) with i<j to index in lexicographic list."""
    # Number of edges before vertex i: (n-1) + (n-2) + ... + (n-i)
    # = i*n - i*(i+1)//2
    # Then within vertex i, edges to i+1..j: j-i-1
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def _idx_to_edge(idx, n):
    """Map index back to (i,j)."""
    # Find i such that cumulative edges up to i-1 <= idx < cumulative up to i
    # cum(i) = i*n - i*(i+1)//2
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cum = mid * n - mid * (mid + 1) // 2
        if cum <= idx:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    cum = i * n - i * (i + 1) // 2
    j = i + 1 + (idx - cum)
    return i, j


def _local_search(n, max_iter=50000, seed=None):
    """Local search: start random, flip colors to reduce C6 count."""
    if seed is not None:
        random.seed(seed)
    
    m = n * (n - 1) // 2
    colors = [random.randint(0, 3) for _ in range(m)]
    
    adj = _build_adj(n, colors)
    conflicts = _count_c6_direct(adj, n)
    
    if conflicts == 0:
        return colors
    
    T = 1.0
    T_min = 0.005
    cooling = 0.9997
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        
        if conflicts == 0:
            return colors
        
        # Pick random edge
        idx = random.randint(0, m - 1)
        i, j = _idx_to_edge(idx, n)
        old_c = colors[idx]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
        
        # Compute delta in conflicts
        # Remove edge (i,j) from color old_c, add to color new_c
        # This is complex; for now, do full recompute (acceptable for n<=20 with few iterations)
        
        # Try the flip
        colors[idx] = new_c
        adj2 = _build_adj(n, colors)
        new_conflicts = _count_c6_direct(adj2, n)
        
        delta = new_conflicts - conflicts
        if delta < 0 or random.random() < (0.0001 / T):
            conflicts = new_conflicts
            adj = adj2
        else:
            colors[idx] = old_c
        
        T = max(T * cooling, T_min)
    
    return colors


def _greedy_init(n):
    """Greedy initialization: assign colors to minimize immediate C6 creation."""
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[set() for _ in range(n)] for _ in range(4)]
    
    for idx in range(m):
        i, j = _idx_to_edge(idx, n)
        best_c = 0
        best_score = float('inf')
        for c in range(4):
            # Count how many C6s would be created by adding edge (i,j) to color c
            # A C6 containing this edge: i-v2-v3-v4-j-i or i-j-v2-v3-v4-i etc.
            # Simpler: just count neighbors
            score = len(adj[c][i]) + len(adj[c][j])
            if score < best_score:
                best_score = score
                best_c = c
        colors[idx] = best_c
        adj[best_c][i].add(j)
        adj[best_c][j].add(i)
    
    return colors


def construct():
    """Find the largest n such that K_n has a 4-coloring with no monochromatic C6."""
    global START_TIME
    START_TIME = time.time()
    
    best_n = 9
    best_colors = None
    
    # Start with known valid coloring for n=9
    n0 = 9
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    best_n = n0
    
    # Try to extend to larger n using local search
    for target_n in range(10, 21):
        if _time_left() < 15.0:
            break
        
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and not found and _time_left() > 10.0:
            attempts += 1
            
            # Initialize
            if target_n <= 12:
                colors = _greedy_init(target_n)
            else:
                colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Local search
            colors = _local_search(target_n, max_iter=30000)
            
            if _is_valid(target_n, colors):
                best_n = target_n
                best_colors = colors
                found = True
                break
        
        if not found and target_n > 12:
            # If we can't find n=13, maybe we can't find larger; but try a few more
            # Actually, continue trying larger n with more attempts
            pass
    
    # Final validation
    if best_colors is not None and best_n >= 1:
        if not _is_valid(best_n, best_colors):
            # Fallback to known valid
            best_n = 9
            best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    return {'n': best_n, 'colors': best_colors}
