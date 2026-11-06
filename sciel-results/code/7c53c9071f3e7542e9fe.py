"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles for fast evaluation
    # A 4-cycle is (u, v, w, x) where u-v-w-x-u are edges
    # We enumerate by choosing two non-adjacent opposite vertices (u, w)
    # and two common neighbors (v, x)
    all_cycles = []
    for u in range(n):
        for w in range(u + 1, n):
            # Common neighbors of u and w
            common = [v for v in range(n) if v != u and v != w]
            # Filter to actual neighbors
            # We'll build this dynamically, but for precomputation we need adjacency
            # Actually, let's precompute all potential 4-cycles (as vertex tuples)
            # Then filter by adjacency at evaluation time
            # Better: precompute all valid 4-cycles in K_n
            pass
    
    # Precompute all 4-cycles in K_n (complete graph)
    # A 4-cycle u-v-w-x-u: u,v,w,x all distinct, u-v, v-w, w-x, x-u are edges
    # In K_n, all such tuples work
    # To avoid duplicates, we fix u < w (opposite vertices) and v < x (other opposite)
    # Actually, let's just enumerate all 4-cycles uniquely
    all_4cycles = []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for w in range(n):
                if w == u or w == v:
                    continue
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    # To avoid duplicates, require u = min(u,v,w,x) and v < x
                    # Actually, a 4-cycle u-v-w-x-u is the same as w-x-u-v-w, etc.
                    # Let's canonicalize: the cycle is determined by the set of 4 vertices
                    # and the pairing of opposite vertices
                    # For 4 vertices {a,b,c,d}, there are 3 ways to form a 4-cycle:
                    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
                    # Let's just enumerate all and use a set
                    pass
    
    # Simpler: enumerate all 4-cycles by choosing 4 vertices and 3 pairings
    # For 4 vertices sorted (a<b<c<d), the 4-cycles are:
    # a-b-c-d-a (edges: ab, bc, cd, da)
    # a-b-d-c-a (edges: ab, bd, dc, ca)
    # a-c-b-d-a (edges: ac, cb, bd, da)
    from itertools import combinations
    all_4cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Three 4-cycles on these 4 vertices
        all_4cycles.append((a, b, c, d))  # a-b-c-d-a
        all_4cycles.append((a, b, d, c))  # a-b-d-c-a
        all_4cycles.append((a, c, b, d))  # a-c-b-d-a
    
    # For each cycle (u,v,w,x), edges are (u,v), (v,w), (w,x), (x,u)
    # Precompute edge indices for each cycle
    cycle_edges = []
    for cyc in all_4cycles:
        u, v, w, x = cyc
        edges = [(min(u,v), max(u,v)), (min(v,w), max(v,w)), (min(w,x), max(w,x)), (min(x,u), max(x,u))]
        cycle_edges.append(edges)
    
    def is_rainbow(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) == 4
    
    def count_bad_cycles(adj, colors):
        """Count 4-cycles where colors are not all distinct."""
        bad = 0
        for i, (cyc, edges) in enumerate(zip(all_4cycles, cycle_edges)):
            # Check if all 4 edges exist
            if not all(adj[e[0], e[1]] for e in edges):
                continue
            c1 = colors[edges[0][0], edges[0][1]]
            c2 = colors[edges[1][0], edges[1][1]]
            c3 = colors[edges[2][0], edges[2][1]]
            c4 = colors[edges[3][0], edges[3][1]]
            if not is_rainbow(c1, c2, c3, c4):
                bad += 1
        return bad
    
    def get_bad_cycles_list(adj, colors):
        """Return list of bad 4-cycle indices."""
        bad = []
        for i, (cyc, edges) in enumerate(zip(all_4cycles, cycle_edges)):
            if not all(adj[e[0], e[1]] for e in edges):
                continue
            c1 = colors[edges[0][0], edges[0][1]]
            c2 = colors[edges[1][0], edges[1][1]]
            c3 = colors[edges[2][0], edges[2][1]]
            c4 = colors[edges[3][0], edges[3][1]]
            if not is_rainbow(c1, c2, c3, c4):
                bad.append(i)
        return bad
    
    def count_bad_cycles_fast(adj, colors):
        """Vectorized count of bad 4-cycles."""
        bad = 0
        for i, (cyc, edges) in enumerate(zip(all_4cycles, cycle_edges)):
            if not all(adj[e[0], e[1]] for e in edges):
                continue
            c1 = colors[edges[0][0], edges[0][1]]
            c2 = colors[edges[1][0], edges[1][1]]
            c3 = colors[edges[2][0], edges[2][1]]
            c4 = colors[edges[3][0], edges[3][1]]
            if not is_rainbow(c1, c2, c3, c4):
                bad += 1
        return bad
    
    # Initialize: start with complete graph, random colors
    adj = np.ones((n, n), dtype=np.uint8)
    np.fill_diagonal(adj, 0)
    colors = np.zeros((n, n), dtype=np.int32)
    for u in range(n):
        for v in range(u + 1, n):
            c = random.randint(0, num_colors - 1)
            colors[u, v] = c
            colors[v, u] = c
    
    current_bad = count_bad_cycles_fast(adj, colors)
    current_edges = int(np.sum(adj) // 2)
    
    best_bad = current_bad
    best_edges = current_edges
    best_adj = adj.copy()
    best_colors = colors.copy()
    
    # Greedy phase: while bad > 0, remove the edge that appears in the most bad cycles
    while current_bad > 0 and time.time() < deadline - 5:
        bad_cycles = get_bad_cycles_list(adj, colors)
        if not bad_cycles:
            current_bad = 0
            break
        
        # Count how many bad cycles each edge participates in
        edge_count = Counter()
        for idx in bad_cycles:
            edges = cycle_edges[idx]
            for e in edges:
                edge_count[e] += 1
        
        # Remove the most frequent edge
        if edge_count:
            max_edge = max(edge_count, key=edge_count.get)
            u, v = max_edge
            adj[u, v] = 0
            adj[v, u] = 0
            colors[u, v] = 0
            colors[v, u] = 0
            current_edges -= 1
            current_bad = count_bad_cycles_fast(adj, colors)
            if current_bad < best_bad or (current_bad == best_bad and current_edges > best_edges):
                best_bad = current_bad
                best_edges = current_edges
                best_adj = adj.copy()
                best_colors = colors.copy()
        else:
            break
    
    # If still bad, do local search with recoloring
    if current_bad > 0 and time.time() < deadline - 2:
        # Try to fix by recoloring
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            bad_cycles = get_bad_cycles_list(adj, colors)
            if not bad_cycles:
                current_bad = 0
                break
            
            # Pick a random bad cycle
            idx = random.choice(bad_cycles)
            edges = cycle_edges[idx]
            # Try recoloring each edge in the cycle
            best_improvement = None
            best_new_bad = current_bad
            for e in edges:
                u, v = e
                old_c = colors[u, v]
                for new_c in range(num_colors):
                    if new_c == old_c:
                        continue
                    colors[u, v] = new_c
                    colors[v, u] = new_c
                    new_bad = count_bad_cycles_fast(adj, colors)
                    if new_bad < best_new_bad:
                        best_new_bad = new_bad
                        best_improvement = (u, v, new_c)
                    colors[u, v] = old_c
                    colors[v, u] = old_c
            
            if best_improvement:
                u, v, new_c = best_improvement
                colors[u, v] = new_c
                colors[v, u] = new_c
                current_bad = best_new_bad
                if current_bad < best_bad or (current_bad == best_bad and current_edges > best_edges):
                    best_bad = current_bad
                    best_edges = current_edges
                    best_adj = adj.copy()
                    best_colors = colors.copy()
            else:
                # No improvement from recoloring, try removing an edge
                idx = random.choice(bad_cycles)
                edges = cycle_edges[idx]
                e = random.choice(edges)
                u, v = e
                adj[u, v] = 0
                adj[v, u] = 0
                colors[u, v] = 0
                colors[v, u] = 0
                current_edges -= 1
                current_bad = count_bad_cycles_fast(adj, colors)
                if current_bad < best_bad or (current_bad == best_bad and current_edges > best_edges):
                    best_bad = current_bad
                    best_edges = current_edges
                    best_adj = adj.copy()
                    best_colors = colors.copy()
    
    # If still bad, fallback: remove edges until valid
    if best_bad > 0:
        adj = best_adj.copy()
        colors = best_colors.copy()
        current_bad = count_bad_cycles_fast(adj, colors)
        current_edges = int(np.sum(adj) // 2)
        
        while current_bad > 0 and time.time() < deadline - 1:
            bad_cycles = get_bad_cycles_list(adj, colors)
            if not bad_cycles:
                break
            edge_count = Counter()
            for idx in bad_cycles:
                edges = cycle_edges[idx]
                for e in edges:
                    edge_count[e] += 1
            if edge_count:
                max_edge = max(edge_count, key=edge_count.get)
                u, v = max_edge
                adj[u, v] = 0
                adj[v, u] = 0
                colors[u, v] = 0
                colors[v, u] = 0
                current_edges -= 1
                current_bad = count_bad_cycles_fast(adj, colors)
            else:
                break
        
        if current_bad == 0 and current_edges > best_edges:
            best_adj = adj.copy()
            best_colors = colors.copy()
            best_edges = current_edges
            best_bad = 0
    
    # Ultimate fallback: C4-free graph (always valid with any coloring)
    if best_bad > 0:
        adj = np.zeros((n, n), dtype=np.uint8)
        edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            # Check if adding edge (u,v) creates a 4-cycle
            # A 4-cycle u-v-w-x-u exists if there's a common neighbor w of u and v
            # such that x is also a common neighbor and x != u, x != v
            # Actually, adding edge (u,v) creates a 4-cycle if there exist w, x
            # such that u-w-v-x-u or similar
            # Simpler: check if there's a path of length 3 from u to v
            # i.e., u-a-b-v for some a, b
            has_c4 = False
            for a in range(n):
                if a == u or a == v or not adj[u, a]:
                    continue
                for b in range(n):
                    if b == u or b == v or b == a or not adj[a, b]:
                        continue
                    if adj[b, v]:
                        has_c4 = True
                        break
                if has_c4:
                    break
            if not has_c4:
                adj[u, v] = 1
                adj[v, u] = 1
        best_adj = adj
        best_colors = np.zeros((n, n), dtype=np.int32)
        best_edges = int(np.sum(adj) // 2)
        best_bad = 0
    
    # Construct output
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                c = int(best_colors[u, v])
                result.append([u, v, c])
    
    return result

# EVOLVE-BLOCK-END
