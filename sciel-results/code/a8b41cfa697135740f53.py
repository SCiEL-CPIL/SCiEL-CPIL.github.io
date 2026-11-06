"""Rewritten program: search for large K2,2,2-free two-colourings of K_n."""
import time
import random
import numpy as np

# Deadline guard: keep a safety margin before the 150s kill.
_START = time.time()
_DEADLINE = _START + 140.0  # 10s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _has_k222(adj):
    """
    Check if adjacency matrix adj (n x n, symmetric, 0/1) contains K2,2,2.
    K2,2,2 has 6 vertices partitioned into 3 pairs {a1,a2},{b1,b2},{c1,c2}
    with all 12 cross-edges present.
    
    Efficient check: for each pair of vertices (u,v) that are NOT adjacent,
    count common neighbours. If two disjoint non-adjacent pairs share >= 2 common neighbours
    that are also non-adjacent to each other... 
    
    Actually, K2,2,2 as a subgraph means: there exist 6 distinct vertices 
    {a1,a2,b1,b2,c1,c2} such that every vertex in {a1,a2} is adjacent to every 
    vertex in {b1,b2} and to every vertex in {c1,c2}, and every vertex in {b1,b2} 
    is adjacent to every vertex in {c1,c2}.
    
    So we need three pairs where each pair is a non-edge (or not required to be non-edge? 
    Wait - K2,2,2 is the complete 3-partite graph with parts of size 2. The vertices 
    within each part are NOT connected. So the 6 vertices form a subgraph isomorphic 
    to K_{2,2,2}, meaning:
    - a1-a2 not necessarily non-edge (they just don't need to be connected for the 
      K2,2,2 structure; actually in a subgraph, edges present in K2,2,2 must be present,
      but edges NOT in K2,2,2 can be present or absent)
    
    Wait, re-reading: "Neither colour class may contain K2,2,2 as a (not necessarily induced) subgraph."
    
    So for colour c, we look at the graph G_c where edge (i,j) has colour c. 
    G_c must NOT contain K2,2,2 as a subgraph. A subgraph means: there exist 6 vertices 
    such that all edges of K2,2,2 are present in G_c. The edges of K2,2,2 are the 12 
    cross-edges between the three pairs. The 3 within-pair edges are NOT part of K2,2,2.
    
    So we need: vertices a1,a2,b1,b2,c1,c2 (all distinct) such that:
    - a1-b1, a1-b2, a2-b1, a2-b2 all present (4 edges between pair A and pair B)
    - a1-c1, a1-c2, a2-c1, a2-c2 all present (4 edges between pair A and pair C)
    - b1-c1, b1-c2, b2-c1, b2-c2 all present (4 edges between pair B and pair C)
    
    The within-pair edges (a1-a2, b1-b2, c1-c2) don't matter.
    
    Efficient algorithm:
    For each pair (u,v) [these will be one of the "pairs" in K2,2,2], 
    compute N(u) ∩ N(v) = common neighbours.
    Then we need to find two more pairs (w1,w2) and (x1,x2) from common neighbours 
    such that w1,w2 are both adjacent to x1,x2.
    
    Actually a cleaner approach: 
    For each unordered pair {u,v}, let C = N(u) ∩ N(v) (common neighbours).
    We need to find in the subgraph induced by C, an edge {w,x} (w≠x, both in C),
    and then another vertex y in C \ {w,x} such that y is adjacent to both w and x.
    Wait no, that's not quite right either.
    
    Let me think again. We need three pairs. Fix one pair {u,v}. The other 4 vertices 
    must all be in N(u) ∩ N(v). Among these 4 vertices {w1,w2,x1,x2}, we need:
    - w1 adjacent to x1 and x2
    - w2 adjacent to x1 and x2
    
    So in the subgraph induced by C = N(u)∩N(v), we need two vertices w1,w2 
    (the "B pair") that both have at least 2 common neighbours in C (the "C pair" 
    vertices x1,x2 must be adjacent to both w1 and w2).
    
    So: for pair {u,v}, C = N(u)∩N(v). In G[C], find if there exist w1≠w2 in C 
    such that |N(w1) ∩ N(w2) ∩ C| >= 2.
    
    This is O(n^2 * n^2) worst case but with bitsets it's fast.
    """
    n = adj.shape[0]
    # Use bitsets for speed
    # adj is a numpy boolean array n x n
    for u in range(n):
        for v in range(u + 1, n):
            # Common neighbours of u and v
            C = adj[u] & adj[v]
            C[u] = False  # exclude u itself (should already be false if no self-loops)
            C[v] = False
            c_count = C.sum()
            if c_count < 4:
                continue
            # In the subgraph induced by C, check if any pair w1,w2 has >= 2 common neighbours in C
            # Get indices in C
            c_idx = np.where(C)[0]
            for i in range(len(c_idx)):
                w1 = c_idx[i]
                for j in range(i + 1, len(c_idx)):
                    w2 = c_idx[j]
                    # Common neighbours of w1 and w2 within C
                    common = adj[w1] & adj[w2] & C
                    if common.sum() >= 2:
                        return True
    return False

def _has_k222_fast(adj):
    """Faster K2,2,2 check using integer bitsets."""
    n = adj.shape[0]
    # Convert to list of integer bitsets
    bits = []
    for i in range(n):
        b = 0
        for j in range(n):
            if adj[i, j]:
                b |= (1 << j)
        bits.append(b)
    
    for u in range(n):
        for v in range(u + 1, n):
            C = bits[u] & bits[v]
            C &= ~(1 << u)
            C &= ~(1 << v)
            c_count = C.bit_count()
            if c_count < 4:
                continue
            # Extract set bits
            c_idx = []
            temp = C
            while temp:
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                c_idx.append(idx)
                temp ^= lb
            # Check pairs in C
            for i in range(len(c_idx)):
                w1 = c_idx[i]
                for j in range(i + 1, len(c_idx)):
                    w2 = c_idx[j]
                    common = bits[w1] & bits[w2] & C
                    if common.bit_count() >= 2:
                        return True
    return False

def _build_adj(n, colors):
    """Build adjacency matrices for both colours."""
    adj0 = np.zeros((n, n), dtype=bool)
    adj1 = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i, j] = adj0[j, i] = True
            else:
                adj1[i, j] = adj1[j, i] = True
            idx += 1
    return adj0, adj1

def _is_valid(n, colors):
    """Check if the colouring is valid (no K2,2,2 in either colour)."""
    adj0, adj1 = _build_adj(n, colors)
    if _has_k222_fast(adj0):
        return False
    if _has_k222_fast(adj1):
        return False
    return True

def _paley(n):
    """
    Paley-type construction. For prime p ≡ 1 (mod 4), the Paley graph on p vertices
    is a self-complementary graph. We try various primes and also generalizations.
    """
    # For the Paley graph of order p (p ≡ 1 mod 4 prime):
    # Edge (i,j) exists if (j-i) is a quadratic residue mod p.
    # This gives a graph where both it and its complement are strongly regular.
    # For K2,2,2-freeness, we need the graph to be K2,2,2-free.
    
    # Known: Paley(17) works (n=17).
    # Let's try to find larger constructions.
    
    # Actually, for this task we need BOTH colour classes to be K2,2,2-free.
    # The Paley graph of order 17 has both the graph and complement K2,2,2-free.
    
    # For larger n, we can try:
    # 1. Direct product / blow-up constructions
    # 2. Random initialisation + local search
    # 3. Cyclic constructions
    
    # Let's implement a cyclic construction:
    # For n vertices in a cycle, colour (i,j) based on (j-i) mod n.
    # We need to find a subset S of {1,...,n-1} with S = -S (for symmetry)
    # such that both the graph with edges at distances in S and the complement are K2,2,2-free.
    
    return None  # Placeholder, we'll use local search instead

def _local_search(n, colors, max_iters=50000):
    """
    Local search: flip random edges to try to eliminate K2,2,2 violations.
    Returns (success, colors).
    """
    colors = list(colors)
    adj0, adj1 = _build_adj(n, colors)
    
    # Count violations
    def count_violations(adj):
        """Count number of K2,2,2 subgraphs (approximate)."""
        # Just check if any exist
        return 1 if _has_k222_fast(adj) else 0
    
    v0 = count_violations(adj0)
    v1 = count_violations(adj1)
    
    if v0 == 0 and v1 == 0:
        return True, colors
    
    best_v = v0 + v1
    best_colors = list(colors)
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        idx = i * n - i * (i + 1) // 2 + j - i - 1
        # Correct index calculation:
        # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # In row i, j-th edge (0-indexed from i+1): j - i - 1
        idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
        old_c = colors[idx]
        new_c = 1 - old_c
        colors[idx] = new_c
        
        # Update adjacency
        if new_c == 0:
            adj0[i, j] = adj0[j, i] = True
            adj1[i, j] = adj1[j, i] = False
        else:
            adj1[i, j] = adj1[j, i] = True
            adj0[i, j] = adj0[j, i] = False
        
        v0 = count_violations(adj0)
        v1 = count_violations(adj1)
        v = v0 + v1
        
        if v < best_v:
            best_v = v
            best_colors = list(colors)
            if v == 0:
                return True, colors
        else:
            # Revert
            colors[idx] = old_c
            if old_c == 0:
                adj0[i, j] = adj0[j, i] = True
                adj1[i, j] = adj1[j, i] = False
            else:
                adj1[i, j] = adj1[j, i] = True
                adj0[i, j] = adj0[j, i] = False
    
    return best_v == 0, best_colors

def _random_coloring(n):
    """Generate a random colouring."""
    num_edges = n * (n - 1) // 2
    return [random.randint(0, 1) for _ in range(num_edges)]

def _cyclic_coloring(n, S):
    """
    Cyclic colouring: colour (i,j) = 0 if (j-i) mod n is in S, else 1.
    S must be symmetric: if d in S then n-d in S.
    """
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(0 if d in S else 1)
    return colors

def _try_cyclic(n):
    """Try to find a valid cyclic colouring of K_n."""
    # S must be a subset of {1, ..., n-1} with d in S iff n-d in S
    # So S is determined by its intersection with {1, ..., (n-1)//2} (and possibly n/2 if n even)
    
    half = (n - 1) // 2
    # For each subset of {1,...,half}, form S and check validity
    # This is 2^half possibilities - too many for large n
    # Instead, try greedy / random subsets
    
    best = None
    for attempt in range(200):
        if _time_left() < 10:
            break
        S = set()
        for d in range(1, half + 1):
            if random.random() < 0.5:
                S.add(d)
                S.add(n - d)
        if n % 2 == 0:
            if random.random() < 0.5:
                S.add(n // 2)
        colors = _cyclic_coloring(n, S)
        if _is_valid(n, colors):
            return colors
    return None

def _expand(n_start, colors_start, target_n):
    """
    Try to expand a valid colouring of K_{n_start} to K_{target_n}
    by adding vertices one at a time.
    """
    n = n_start
    colors = list(colors_start)
    
    while n < target_n:
        if _time_left() < 10:
            break
        # Try to add vertex n
        # For each new vertex, we need to assign colours to edges (i, n) for i in 0..n-1
        # such that no K2,2,2 is created in either colour.
        
        # Greedy: for each new edge, try both colours and pick one that doesn't create a violation
        new_colors = [None] * n  # colours for edges (i, n)
        success = True
        
        for i in range(n):
            # Try colour 0
            found_valid = False
            for c in [0, 1]:
                new_colors[i] = c
                # Build full colouring
                full_colors = colors + new_colors[:i+1] + [0]*(n - i - 1)  # placeholder
                # Actually we need to check incrementally
                # Let's just try all 2^n assignments for small n... no that's too slow.
                # Instead, check if adding edge (i,n) with colour c creates a K2,2,2.
                
                # For a K2,2,2 to be created involving the new vertex n:
                # n must be in one of the three pairs. WLOG, n is in pair A = {n, a}.
                # Then we need vertices b1,b2 (pair B) and c1,c2 (pair C) all adjacent to n and a.
                # So: a must be adjacent to n (by our choice), and b1,b2,c1,c2 must all be 
                # adjacent to both n and a.
                
                # Case 1: n and a are in the same pair.
                # We need 4 other vertices b1,b2,c1,c2 all adjacent to both n and a.
                # So |N(n) ∩ N(a) \\ {n,a}| >= 4, and among those 4, we need two pairs 
                # where each pair's vertices are mutually adjacent to each other... 
                # Wait no. The 4 vertices form pairs B and C. Edges between B and C must exist.
                # b1-c1, b1-c2, b2-c1, b2-c2 must all be edges.
                # So in the subgraph induced by {b1,b2,c1,c2}, we need a K_{2,2} (complete bipartite).
                
                # This is getting complex. Let me just do a full check.
                pass
            
            # Simplified: just do full validity check after assigning all new edges
            pass
        
        # This approach is too slow. Let's use a different strategy:
        # Randomly assign colours to new edges, then do local search.
        
        new_edge_colors = [random.randint(0, 1) for _ in range(n)]
        full_colors = colors + new_edge_colors
        
        # Local search to fix violations
        ok, fixed_colors = _local_search(n + 1, full_colors, max_iters=10000)
        if ok:
            colors = fixed_colors
            n += 1
        else:
            # Try a few more random initializations
            found = False
            for _ in range(5):
                if _time_left() < 10:
                    break
                new_edge_colors = [random.randint(0, 1) for _ in range(n)]
                full_colors = colors + new_edge_colors
                ok, fixed_colors = _local_search(n + 1, full_colors, max_iters=10000)
                if ok:
                    colors = fixed_colors
                    n += 1
                    found = True
                    break
            if not found:
                break
    
    return n, colors

def construct():
    """
    Find the largest n such that K_n can be two-coloured with no monochromatic K2,2,2.
    """
    random.seed(42)
    
    best_n = 0
    best_colors = None
    
    # Strategy 1: Start from Paley(17) and try to expand
    squares = {x * x % 17 for x in range(1, 17)}
    paley_colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    
    if _is_valid(17, paley_colors):
        best_n = 17
        best_colors = paley_colors
    else:
        # Shouldn't happen, but fallback
        best_n = 0
        best_colors = None
    
    # Try to expand from 17
    if _time_left() > 30:
        expanded_n, expanded_colors = _expand(17, best_colors, 31)
        if expanded_n > best_n:
            best_n = expanded_n
            best_colors = expanded_colors
    
    # Strategy 2: Try random initialisations + local search for various n
    for target_n in range(min(best_n + 1, 20), 32):
        if _time_left() < 30:
            break
        found = False
        for attempt in range(10):
            if _time_left() < 15:
                break
            colors = _random_coloring(target_n)
            ok, fixed = _local_search(target_n, colors, max_iters=30000)
            if ok:
                best_n = target_n
                best_colors = fixed
                found = True
                break
        if not found and target_n > best_n:
            # Can't reach this n, likely can't reach larger ones either
            # But keep trying a bit more
            pass
    
    # Strategy 3: Try cyclic constructions
    for n in range(18, 32):
        if _time_left() < 20:
            break
        if n <= best_n:
            continue
        colors = _try_cyclic(n)
        if colors is not None:
            best_n = n
            best_colors = colors
    
    # Final verification
    if best_colors is not None and best_n > 0:
        if not _is_valid(best_n, best_colors):
            # Fallback to known valid
            best_n = 17
            best_colors = paley_colors
    
    return {'n': best_n, 'colors': best_colors}
