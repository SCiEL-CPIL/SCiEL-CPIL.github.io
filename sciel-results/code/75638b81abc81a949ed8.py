"""Girth-constrained random regular graph via configuration model with dynamic edge rejection."""
import random
import sys
import time
from collections import deque

# EVOLVE-BLOCK-START

def _short_cycle_exists(adj, u, v, g):
    """Return True if there is a path from u to v of length < g-1 in adj (excluding the direct edge).
    Adding edge (u,v) would then create a cycle of length < g."""
    if u == v:
        return True
    # BFS from u, but we need to avoid using the direct edge (u,v) if it exists.
    # Since we're checking BEFORE adding the edge, the edge isn't in adj yet.
    # But if we're repairing, we've removed it. So just BFS normally.
    dist = {u: 0}
    dq = deque([u])
    while dq:
        cur = dq.popleft()
        d = dist[cur]
        if d >= g - 1:
            break
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = d + 1
                if nb == v:
                    return True
                dq.append(nb)
    return False

def _verify_girth(adj, n, g):
    """Verify that no cycle shorter than g exists using non-backtracking BFS from each vertex."""
    for start in range(n):
        # Non-backtracking walk: state is (current, prev)
        # We want to detect if we can return to start with a walk of length < g
        # without immediately backtracking.
        # BFS over (node, prev) states
        visited = set()
        visited.add((start, -1))
        dq = deque()
        for nb in adj[start]:
            dq.append((nb, start, 1))
        while dq:
            node, prev, depth = dq.popleft()
            if depth >= g:
                continue
            if node == start:
                return False  # Found a cycle of length depth < g
            for nb in adj[node]:
                if nb == prev:
                    continue  # No backtracking
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    dq.append((nb, node, depth + 1))
    return True

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using configuration model
    with dynamic edge rejection and local repair.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    timeout = 115  # Leave some margin for evaluation overhead
    
    # Moore bound for (3,16): 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510
    # n must be even for k=3 regular graph
    # Known best: 936. Best lower bound: 512.
    
    # Try increasing n from the Moore bound upward
    # For each n, try multiple random configurations with repair
    
    for n in range(512, 1000, 2):
        if time.time() - start_time > timeout:
            break
        
        max_attempts = 50 if n <= 600 else 30
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Create stubs: vertex i has k stubs
            stubs = []
            for i in range(n):
                for j in range(k):
                    stubs.append((i, j))
            
            # Randomly pair up stubs
            random.shuffle(stubs)
            pairs = [(stubs[i], stubs[i+1]) for i in range(0, len(stubs), 2)]
            
            # Build initial edge set, identifying bad edges
            # An edge is bad if: loop (u==v), multi-edge, or creates cycle < g
            adj = [[] for _ in range(n)]
            edge_count = {}  # (min(u,v), max(u,v)) -> count
            bad_edges = set()  # indices of pairs that are bad
            good_pairs = []
            
            for idx, (s1, s2) in enumerate(pairs):
                u, _ = s1
                v, _ = s2
                
                if u == v:
                    bad_edges.add(idx)
                    continue
                
                ekey = (min(u, v), max(u, v))
                edge_count[ekey] = edge_count.get(ekey, 0) + 1
                if edge_count[ekey] > 1:
                    bad_edges.add(idx)
                    continue
                
                # Check girth: is there a path from u to v of length < g-1?
                if _short_cycle_exists(adj, u, v, g):
                    bad_edges.add(idx)
                    continue
                
                # This edge is good so far
                adj[u].append(v)
                adj[v].append(u)
                good_pairs.append(idx)
            
            # Now repair bad edges by re-pairing their stubs
            # The stubs involved in bad edges need to be re-matched
            bad_stubs = []
            for idx in bad_edges:
                s1, s2 = pairs[idx]
                bad_stubs.append(s1)
                bad_stubs.append(s2)
            
            # Remove the bad edges from adj
            for idx in bad_edges:
                s1, s2 = pairs[idx]
                u, _ = s1
                v, _ = s2
                if u != v:
                    # Remove edge from adj
                    if v in adj[u]:
                        adj[u].remove(v)
                    if u in adj[v]:
                        adj[v].remove(u)
            
            # Now we need to re-pair bad_stubs among themselves (and possibly with good stubs if odd)
            # For simplicity, just shuffle and re-pair the bad stubs
            # If odd number, we'd have a problem, but since total stubs is even and 
            # we remove pairs, bad_stubs should be even.
            
            # Actually, let's do iterative repair:
            # Repeatedly find bad edges, remove them, re-pair their stubs randomly,
            # and check again.
            
            repaired = False
            for repair_iter in range(100):
                if time.time() - start_time > timeout:
                    break
                
                # Check current state for bad edges
                # Rebuild check
                current_edges = []
                for idx in good_pairs:
                    s1, s2 = pairs[idx]
                    u, _ = s1
                    v, _ = s2
                    current_edges.append((u, v))
                
                # Also check the re-paired edges from bad_stubs
                # Let's just do a full re-check
                
                # Actually, let's take a simpler approach:
                # Just re-pair all bad stubs randomly and check if all edges are valid.
                
                if len(bad_stubs) == 0:
                    # No bad edges, verify
                    break
                
                random.shuffle(bad_stubs)
                new_pairs = [(bad_stubs[i], bad_stubs[i+1]) for i in range(0, len(bad_stubs), 2)]
                
                # Check if new pairs are valid
                all_valid = True
                for s1, s2 in new_pairs:
                    u, _ = s1
                    v, _ = s2
                    if u == v:
                        all_valid = False
                        break
                    ekey = (min(u, v), max(u, v))
                    # Check if this edge already exists in good_pairs
                    for idx in good_pairs:
                        gs1, gs2 = pairs[idx]
                        gu, _ = gs1
                        gv, _ = gs2
                        gkey = (min(gu, gv), max(gu, gv))
                        if gkey == ekey:
                            all_valid = False
                            break
                    if not all_valid:
                        break
                    if _short_cycle_exists(adj, u, v, g):
                        all_valid = False
                        break
                
                if all_valid:
                    # Add these edges
                    for s1, s2 in new_pairs:
                        u, _ = s1
                        v, _ = s2
                        adj[u].append(v)
                        adj[v].append(u)
                        pairs[len(pairs)] = (s1, s2)
                    bad_stubs = []
                    repaired = True
                    break
                else:
                    # Keep trying to repair
                    # Re-shuffle and try again
                    continue
            
            if not bad_stubs:
                # All edges should be valid now. Verify.
                # Check degrees
                degree_ok = all(len(adj[i]) == k for i in range(n))
                if degree_ok:
                    # Verify girth thoroughly
                    if _verify_girth(adj, n, g):
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return n, edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
