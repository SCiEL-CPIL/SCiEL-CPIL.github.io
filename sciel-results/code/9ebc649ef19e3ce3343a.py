import time
import itertools
import random

_DEADLINE = time.time() + 135  # safety margin before 150s wall-clock kill


def _elapsed_ok():
    return time.time() < _DEADLINE


def _verify(n, colors):
    """Check that no colour class contains a C4 (two vertices with two common neighbours).
    Efficient bitset check. Returns True if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # adjacency bitmasks per colour: adj[c][v] = bitmask of neighbours of v in colour c
    adj = [ [0]*n for _ in range(5) ]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]; idx += 1
            if c < 0 or c > 4:
                return False
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):  # at least two common neighbours -> C4
                    return False
    return True


def _cyclic(n, k):
    """Colour edge (i,j) by circular distance 1..k (k = n//2, requires n odd)."""
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            dist = min(d, n - d)
            out.append(dist - 1)
    return out


def _greedy_construct(n, seed):
    """Greedy randomized construction with dynamic forbidden-pair tracking.
    
    Processes edges in a specific order. For each edge (i,j), tries each color
    and counts how many new C4 violations would be created. Picks the color
    with minimum new violations (with random tie-breaking).
    
    Returns (colors, valid) where valid is True if no C4 exists.
    """
    if not _elapsed_ok():
        return None, False
    
    rng = random.Random(seed)
    
    # adj[c][v] = bitmask of neighbors of v in color c
    adj = [[0]*n for _ in range(5)]
    # For each color c and vertex v, track which other vertices have exactly 1 common neighbor with v in color c
    # fp[c][v][w] = count of common neighbors of v and w in color c (we only care if it's 0, 1, or >=2)
    # To save memory/time, use: for each color c, for each pair (u,v), count common neighbors
    # But that's O(n^2) per color = O(5*n^2) which is fine for n<=29
    
    # Actually, let's track: for each color c, common_count[c][u][v] = number of common neighbors
    # We need this to be efficient. For n=29, n^2 = 841, times 5 = 4205 entries. Fine.
    
    # Better approach: for each color c, maintain adj[c][v] as bitmask.
    # When adding edge (i,j) with color c:
    #   new common neighbors created = for each pair (u,v) where u is neighbor of i in c and v is neighbor of j in c
    #   Actually, adding edge (i,j) in color c creates a C4 if there exists a vertex k such that
    #   k is already adjacent to both i and j in color c. That's adj[c][i] & adj[c][j] != 0.
    #   More precisely, a C4 is formed if |adj[c][i] & adj[c][j]| >= 1 when we add the edge.
    #   Wait no - a C4 in the color-c subgraph means two vertices with two common neighbors.
    #   Adding edge (i,j) in color c: the new C4s are those where i and j become two common neighbors
    #   of some pair (u,v), OR where some existing common neighbor of i and j forms a C4.
    #   
    #   Let me think more carefully. In the color-c subgraph, a C4 is a cycle of length 4.
    #   When we add edge (i,j) to the color-c graph, new C4s are created if:
    #   - There exists a vertex k such that k-i and k-j are both edges in color c (i.e., k is a common neighbor of i and j). Then i-k-j plus the new edge i-j forms a triangle, not a C4.
    #   - Actually, a C4 involves 4 distinct vertices. Adding edge (i,j) creates a C4 if there exist vertices u,v (distinct from i,j) such that u-i, u-v, v-j, and now i-j form a C4? No, that's not right either.
    #   
    #   Let me reconsider. A C4 in color c is four vertices a,b,c,d with edges ab, bc, cd, da all in color c.
    #   When we add edge (i,j) in color c, this edge participates in a new C4 if there exists a path of length 3 from i to j in the existing color-c graph (i.e., i-x-y-j for distinct x,y).
    #   Equivalently, there exists a vertex x adjacent to i in color c and a vertex y adjacent to j in color c such that x-y is also an edge in color c, and x != y, x != i, y != j.
    #   
    #   So: new C4s from adding (i,j) in color c = number of pairs (x,y) where:
    #     x in adj[c][i], y in adj[c][j], x != y, edge (x,y) in color c
    #   
    #   This is: for each x in adj[c][i], count neighbors of x in color c that are also in adj[c][j] and not equal to x.
    #   
    #   Let common = adj[c][i] & adj[c][j]  (vertices adjacent to both i and j)
    #   For x in adj[c][i] \ common: new C4s += popcount(adj[c][x] & adj[c][j] & ~(1<<x) & ~(1<<i) & ~(1<<j))
    #   For x in common: new C4s += popcount(adj[c][x] & adj[c][j] & ~(1<<x) & ~(1<<i) & ~(1<<j))
    #   Wait, this is getting complicated. Let me simplify.
    #
    #   Actually, the standard way: adding edge (i,j) creates a C4 iff there's a path i-x-y-j in the current graph.
    #   Number of such C4s = sum over x in adj[c][i] of (number of neighbors of x in color c that are in adj[c][j], excluding i and j)
    #   But we need x != y and the path to be i-x-y-j with all distinct.
    #   
    #   Let me use: for each x in adj[c][i]:
    #     candidates = adj[c][x] & adj[c][j] & ~(1<<i) & ~(1<<j) & ~(1<<x)
    #     new_c4s += popcount(candidates)
    #   But this counts each C4 twice (once for x and once for y in the path i-x-y-j).
    #   Actually no, it counts each C4 exactly once per choice of x. A C4 i-x-y-j is counted when x is the first intermediate vertex.
    #   But wait, the C4 could also be traversed as i-y-x-j, which would be counted when y is chosen.
    #   So we're double counting. Let's just use the count as a heuristic (it's proportional to actual C4s).
    #
    #   For simplicity and speed, let me just count: for each x in adj[c][i], popcount(adj[c][x] & adj[c][j])
    #   This overcounts but is a valid proxy for "how bad is this assignment".

    colors = [0] * (n * (n - 1) // 2)
    
    # Edge ordering: process edges in a way that might help. 
    # One option: process by "distance" or use a fixed order.
    # Let's use the natural lexicographic order for simplicity, but we could also
    # try processing high-degree edges first or random order.
    
    # For the greedy approach, let's process edges in order (0,1), (0,2), ..., (0,n-1), (1,2), ...
    # This is the same as the output order.
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not _elapsed_ok():
                return None, False
            
            best_color = -1
            best_score = float('inf')
            
            # Try each color, compute the "cost" (number of new C4s created)
            for c in range(5):
                # Compute how many C4s would be created by adding (i,j) in color c
                score = 0
                # For each neighbor x of i in color c, count neighbors of x in color c that are also neighbors of j
                mask_i = adj[c][i]
                mask_j = adj[c][j]
                
                # Iterate over set bits in mask_i
                x = mask_i
                while x:
                    lb = x & (-x)
                    x ^= lb
                    x_idx = lb.bit_length() - 1
                    if x_idx == j:
                        continue
                    # Count neighbors of x_idx in color c that are in mask_j
                    # Exclude i, j, x_idx itself
                    common = adj[c][x_idx] & mask_j & ~(1 << i) & ~(1 << j) & ~(1 << x_idx)
                    # Count bits
                    cnt = 0
                    tmp = common
                    while tmp:
                        tmp &= tmp - 1
                        cnt += 1
                    score += cnt
                
                if score < best_score:
                    best_score = score
                    best_color = c
                elif score == best_score:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_color = c
            
            # Assign the best color
            colors[idx] = best_color
            adj[best_color][i] |= (1 << j)
            adj[best_color][j] |= (1 << i)
            idx += 1
    
    valid = _verify(n, colors)
    return colors, valid


def _greedy_with_restarts(n, max_restarts=50):
    """Try multiple random restarts of the greedy construction."""
    best_colors = None
    best_valid = False
    
    for attempt in range(max_restarts):
        if not _elapsed_ok():
            break
        seed = attempt * 7919 + 13
        colors, valid = _greedy_construct(n, seed)
        if colors is None:
            break
        if valid:
            return colors, True
        # Track the "best" invalid coloring (fewest violations) as fallback
        # For now, just keep trying
    
    return None, False


def _build_cyclic_ext(n):
    """Build a C4-free 5-colouring for n=26 using product structure."""
    if n == 26:
        colors = []
        for i in range(26):
            for j in range(i + 1, 26):
                a1, b1 = divmod(i, 13)
                a2, b2 = divmod(j, 13)
                d = (b2 - b1) % 13
                if d == 0:
                    d = 13
                if a1 == a2:
                    c = d % 5
                else:
                    c = (d + 2) % 5
                colors.append(c)
        return colors
    return None


def _local_search(n, colors, max_iter=100000):
    """Local search to eliminate C4 violations."""
    if _verify(n, colors):
        return colors
    
    random.seed(12345)
    
    cur = colors[:]
    
    def build_adj(cols, n):
        adj = [[0]*n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = cols[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj
    
    def count_violations(adj, n):
        cnt = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[c][u] & adj[c][v]
                    if common & (common - 1):
                        cnt += 1
        return cnt
    
    adj = build_adj(cur, n)
    cur_viol = count_violations(adj, n)
    
    for it in range(max_iter):
        if not _elapsed_ok():
            break
        if cur_viol == 0:
            return cur
        
        i = random.randint(0, n-2)
        j = random.randint(i+1, n-1)
        idx = 0
        for ii in range(n):
            for jj in range(ii+1, n):
                if ii == i and jj == j:
                    break
                idx += 1
            else:
                continue
            break
        
        old_c = cur[idx]
        best_c = -1
        best_v = cur_viol
        
        for new_c in range(5):
            if new_c == old_c:
                continue
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            v = count_violations(adj, n)
            
            adj[new_c][i] &= ~(1 << j)
            adj[new_c][j] &= ~(1 << i)
            adj[old_c][i] |= (1 << j)
            adj[old_c][j] |= (1 << i)
            
            if v < best_v:
                best_v = v
                best_c = new_c
        
        if best_c >= 0:
            cur[idx] = best_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[best_c][i] |= (1 << j)
            adj[best_c][j] |= (1 << i)
            cur_viol = best_v
    
    return cur


def _build():
    best = None
    best_n = 0

    # 1) cyclic construction for odd n
    for n in range(3, 29, 2):
        if not _elapsed_ok():
            break
        cols = _cyclic(n, n // 2)
        if _verify(n, cols):
            if n > best_n:
                best_n = n
                best = (n, cols)

    # 2) Try n=26 with product construction
    if _elapsed_ok():
        cols26 = _build_cyclic_ext(26)
        if cols26 is not None:
            if _verify(26, cols26):
                if 26 > best_n:
                    best_n = 26
                    best = (26, cols26)
            else:
                cols26_fix = _local_search(26, cols26, max_iter=50000)
                if _verify(26, cols26_fix):
                    if 26 > best_n:
                        best_n = 26
                        best = (26, cols26_fix)

    # 3) Greedy randomized construction for larger n
    if _elapsed_ok():
        for target in range(27, 29):
            if not _elapsed_ok():
                break
            cols, valid = _greedy_with_restarts(target, max_restarts=30)
            if valid:
                if target > best_n:
                    best_n = target
                    best = (target, cols)
                break  # If we found a valid one for 27, try 28
    
    # 4) Try to extend to 27, 28 with local search on greedy result
    if best is not None and _elapsed_ok():
        for target in range(best_n + 1, 29):
            if not _elapsed_ok():
                break
            if target % 2 == 1:
                cand = _cyclic(target, target // 2)
                if _verify(target, cand):
                    if target > best_n:
                        best_n = target
                        best = (target, cand)
                    continue
            random.seed(target)
            base_n, base_cols = best
            cand = base_cols[:]
            for i in range(base_n):
                cand.append(random.randint(0, 4))
            cand_fix = _local_search(target, cand, max_iter=30000)
            if _verify(target, cand_fix):
                if target > best_n:
                    best_n = target
                    best = (target, cand_fix)

    if best is None:
        n = 11
        cols = _cyclic(n, n // 2)
        best = (n, cols)

    return {'n': best[0], 'colors': best[1]}


def construct():
    try:
        return _build()
    except Exception:
        n = 11
        return {'n': n, 'colors': _cyclic(n, n // 2)}
