import time
import random

def _has_k222_bitmask(adj):
    """
    Check if the graph represented by bitmask adjacency list 'adj' contains K2,2,2.
    adj[i] is an integer bitmask of neighbors of i.
    Returns True if K2,2,2 is found, False otherwise.
    """
    n = len(adj)
    # Iterate over pairs (u, v) that are NOT connected (non-edges in this graph)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue # u and v are connected, cannot form a part of K2,2,2 together
            
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            
            # We need to find two disjoint non-edges (w,x) and (y,z) in 'common'
            # such that all edges between {w,x} and {y,z} are present.
            # This implies {w,x} and {y,z} must be independent sets in the subgraph induced by 'common'.
            # And every vertex in {w,x} is connected to every vertex in {y,z}.
            
            # Extract list of vertices in common
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                mask ^= lsb
            
            # Iterate over pairs (w, x) in verts that are NOT connected
            for i in range(len(verts)):
                w = verts[i]
                for j in range(i + 1, len(verts)):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue # w and x are connected, cannot form a part together
                    
                    # Now we have Part A={u,v}, Part B={w,x}
                    # We need Part C={y,z} such that:
                    # 1. y, z are in common (connected to u, v)
                    # 2. y, z are NOT connected to each other
                    # 3. y, z are connected to w and x
                    
                    # Candidates for C must be in common, and connected to w and x
                    cand = common & adj[w] & adj[x]
                    
                    # Extract candidates
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        idx = lsb.bit_length() - 1
                        cverts.append(idx)
                        cmask ^= lsb
                        
                    # Check if cverts contains a non-edge pair
                    for k in range(len(cverts)):
                        y = cverts[k]
                        for l in range(k + 1, len(cverts)):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                return True
    return False

def _count_k222_bitmask(adj):
    """
    Count the number of K2,2,2 subgraphs. Used for optimization heuristic.
    """
    n = len(adj)
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            
            common = adj[u] & adj[v]
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                mask ^= lsb
            
            for i in range(len(verts)):
                w = verts[i]
                for j in range(i + 1, len(verts)):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue
                    
                    cand = common & adj[w] & adj[x]
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        idx = lsb.bit_length() - 1
                        cverts.append(idx)
                        cmask ^= lsb
                        
                    for k in range(len(cverts)):
                        y = cverts[k]
                        for l in range(k + 1, len(cverts)):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                count += 1
    return count

def _build_adj_bitmask(colors, n):
    """Build bitmask adjacency list from flat color list."""
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _is_valid_bitmask(colors, n):
    """Check if coloring is valid (no K2,2,2 in either color)."""
    adj1 = _build_adj_bitmask(colors, n)
    if _has_k222_bitmask(adj1):
        return False
    
    # Build adj0 (complement)
    # adj0[i] has bits set for j where colors[i,j] == 0
    adj0 = [0] * n
    # Full mask for n vertices
    # We can compute adj0 by taking ~adj1 and masking
    # Or simply iterate again
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            idx += 1
            
    if _has_k222_bitmask(adj0):
        return False
    return True

def _paley_colors(n, p):
    """
    Generate Paley coloring for n vertices using prime p >= n.
    Color 1 if (j-i) mod p is a quadratic residue, else 0.
    """
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_n = 0
    best_colors = None
    
    # Primes suitable for Paley (p % 4 == 1)
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149]
    
    # 1. Try to find a valid coloring for n=30
    # Since R(K2,2,2) = 31, a valid coloring for n=30 should exist.
    # We try local search starting from Paley(37) or Paley(29) extended?
    # Paley(29) is for n=29. We can try to extend it or just use Paley(37) truncated.
    
    # Strategy for n=30:
    # Initialize with Paley(37) truncated to 30 vertices.
    # Then use local search to remove any K2,2,2s.
    
    target_n = 30
    if time.time() < deadline - 5:
        colors30 = _paley_colors(30, 37)
        
        # Local search
        adj1 = _build_adj_bitmask(colors30, 30)
        adj0 = [0] * 30
        idx = 0
        for i in range(30):
            for j in range(i + 1, 30):
                if colors30[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                idx += 1
        
        # Count initial violations
        violations = _count_k222_bitmask(adj1) + _count_k222_bitmask(adj0)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try to find a K2,2,2 and flip an edge in it
            # We do a random search for a K2,2,2
            # If found, flip a random edge in that K2,2,2
            # If not found, we are done (valid)
            
            found_violation = False
            # Sample some random edges to check if they are in a K2,2,2?
            # Or just check validity first
            if not (_has_k222_bitmask(adj1) or _has_k222_bitmask(adj0)):
                # Valid!
                if 30 > best_n:
                    best_n = 30
                    best_colors = colors30[:]
                break
            
            # Find a K2,2,2 in adj1 or adj0
            # We'll use a helper to find one specific K2,2,2
            # For simplicity, we'll just flip a random edge if violations > 0
            # This is slow. Better to find the specific edges.
            
            # Let's implement a quick find_k222_edge
            # We'll reuse the logic from _has_k222_bitmask but return the edge indices
            
            def find_k222_edges(adj):
                n = len(adj)
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            continue
                        common = adj[u] & adj[v]
                        verts = []
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            idx = lsb.bit_length() - 1
                            verts.append(idx)
                            mask ^= lsb
                        for i in range(len(verts)):
                            w = verts[i]
                            for j in range(i + 1, len(verts)):
                                x = verts[j]
                                if adj[w] & (1 << x):
                                    continue
                                cand = common & adj[w] & adj[x]
                                cverts = []
                                cmask = cand
                                while cmask:
                                    lsb = cmask & -cmask
                                    idx = lsb.bit_length() - 1
                                    cverts.append(idx)
                                    cmask ^= lsb
                                for k in range(len(cverts)):
                                    y = cverts[k]
                                    for l in range(k + 1, len(cverts)):
                                        z = cverts[l]
                                        if not (adj[y] & (1 << z)):
                                            # Found K2,2,2 with parts {u,v}, {w,x}, {y,z}
                                            # Edges are (u,v), (w,x), (y,z) non-edges in adj
                                            # But we need to flip an edge in the COLORING
                                            # The edges of K2,2,2 in the coloring are the cross edges.
                                            # Let's return a random cross edge index
                                            # Cross edges: (u,w), (u,x), (u,y), (u,z), (v,w)...
                                            # We need the flat index
                                            def get_idx(a, b):
                                                if a > b: a, b = b, a
                                                # i * (2n - i - 3) // 2 + (j - i - 1)
                                                return a * (2 * n - a - 3) // 2 + (b - a - 1)
                                            
                                            return random.choice([
                                                get_idx(u, w), get_idx(u, x), get_idx(u, y), get_idx(u, z),
                                                get_idx(v, w), get_idx(v, x), get_idx(v, y), get_idx(v, z),
                                                get_idx(w, y), get_idx(w, z), get_idx(x, y), get_idx(x, z)
                                            ])
                return None

            # Check adj1
            edge_idx = find_k222_edges(adj1)
            if edge_idx is None:
                # Check adj0
                edge_idx = find_k222_edges(adj0)
            
            if edge_idx is not None:
                # Flip the edge
                colors30[edge_idx] = 1 - colors30[edge_idx]
                # Update adj bitmasks
                # We need to know which color it was
                # If it was 1, it becomes 0. Remove from adj1, add to adj0.
                # If it was 0, it becomes 1. Remove from adj0, add to adj1.
                
                # Rebuilding adj is O(n^2) which is small (900 edges).
                # Let's just rebuild for safety and simplicity in the loop, 
                # as n=30 is small.
                adj1 = _build_adj_bitmask(colors30, 30)
                adj0 = [0] * 30
                idx = 0
                for i in range(30):
                    for j in range(i + 1, 30):
                        if colors30[idx] == 0:
                            adj0[i] |= (1 << j)
                            adj0[j] |= (1 << i)
                        idx += 1
                improved = True
            else:
                # No K2,2,2 found? Should have been caught by _has_k222_bitmask
                break
        
        if best_n == 30:
            return {'n': 30, 'colors': best_colors}

    # 2. If n=30 failed, try n=29
    # Paley(29) is a known valid coloring for n=29
    if time.time() < deadline - 5:
        colors29 = _paley_colors(29, 29)
        if _is_valid_bitmask(colors29, 29):
            if 29 > best_n:
                best_n = 29
                best_colors = colors29[:]
            return {'n': 29, 'colors': best_colors}
        
        # If Paley(29) somehow failed (it shouldn't), try local search on it
        # But Paley(29) is theoretically valid for K2,2,2?
        # Actually, R(K2,2,2) = 31 means Paley(29) is valid.
        
    # 3. Fallback: Random search for n=29 or smaller
    # This is unlikely to be reached if Paley works.
    for n in range(28, 16, -1):
        if time.time() > deadline:
            break
        for _ in range(100):
            colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            if _is_valid_bitmask(colors, n):
                if n > best_n:
                    best_n = n
                    best_colors = colors[:]
                break
    
    if best_colors is None:
        # Absolute fallback
        best_n = 17
        best_colors = _paley_colors(17, 17)
        
    return {'n': best_n, 'colors': best_colors}
