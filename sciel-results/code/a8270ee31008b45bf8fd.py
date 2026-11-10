import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    q = 5
    # Build PG(2, q) points: 31 points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    adj = [set() for _ in range(n)]
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                edge_set.add((i, j))
    
    # Build adjacency matrix as numpy for fast computation
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in adj[i]:
            A[i, j] = 1
            A[j, i] = 1
    
    # Number of C4s using a specific edge (u,v):
    # = number of pairs (a,b) with a in N(u), b in N(v), a~b, a!=b
    # This equals (A[u] * A[v].T)[0] - common_neighbors(u,v)
    # Because A[u] @ A[v] counts all pairs (a,b) where a~u and b~v
    # Among these, pairs where a==b are the common neighbors
    # So C4s using edge (u,v) = sum(A[u] * A[v]) - |N(u) ∩ N(v)|
    
    # Precompute for all pairs
    # c4_count[u,v] = number of 4-cycles that use edge (u,v)
    c4_count = np.zeros((n, n), dtype=np.int32)
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 1:
                # Edge exists: count C4s using this edge
                # sum(A[u] * A[v]) counts pairs (a,b) with a~u, b~v
                # subtract common neighbors (a==b cases)
                prod = A[u] * A[v]  # element-wise product
                total = prod.sum()
                common = np.sum((A[u] & A[v]) > 0)
                c4_count[u, v] = total - common
                c4_count[v, u] = total - common
            else:
                # Non-edge: this is the cost of ADDING edge (u,v)
                prod = A[u] * A[v]
                total = prod.sum()
                common = np.sum((A[u] & A[v]) > 0)
                c4_count[u, v] = total - common
                c4_count[v, u] = total - common
    
    # Current total C4 count (verify)
    total_c4 = 0
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 1:
                total_c4 += c4_count[u, v]
    total_c4 //= 2  # each C4 is counted twice (once for each of its 4 edges... wait)
    
    # Actually each C4 has 4 edges, so it's counted 4 times total
    total_c4 = 0
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 1:
                total_c4 += c4_count[u, v]
    total_c4 //= 4
    
    # Strategy 1: Just add the best non-edge (minimize added C4s)
    best_add_edge = None
    best_add_cost = float('inf')
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 0:
                if c4_count[u, v] < best_add_cost:
                    best_add_cost = c4_count[u, v]
                    best_add_edge = (u, v)
    
    # Strategy 2: Remove one edge, add one non-edge (swap)
    # Net change = c4_count[new_edge] - c4_count[old_edge]
    # But removing an edge also changes the C4 counts of other edges!
    # For a single swap, the interaction is:
    # - Removing edge (a,b) destroys c4_count[a,b] C4s
    # - Adding edge (c,d) creates c4_count[c,d] C4s
    # - But some of the C4s created by adding (c,d) might use (a,b) or vice versa
    # For simplicity, first try the naive approach, then refine
    
    best_swap = None  # (remove_edge, add_edge, net_change)
    best_swap_change = float('inf')
    
    # For each edge to remove, find best non-edge to add
    # To be efficient, pre-sort non-edges by c4_count
    non_edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 0:
                non_edges_list.append((c4_count[u, v], u, v))
    non_edges_list.sort()
    
    # For each edge, check if swapping with a low-cost non-edge helps
    # Net benefit = c4_count[remove_edge] - c4_count[add_edge] (approximately)
    # We want this to be positive (net reduction)
    for (cost_add, u, v) in non_edges_list:
        if time.time() > deadline - 30:
            break
        for (cost_rem, a, b) in [(c4_count[x, y], x, y) for x in range(n) for y in range(x+1, n) if A[x, y] == 1]:
            pass  # This is too slow, let me restructure
    
    # Better: for each non-edge to add, find the best edge to remove
    # Best edge to remove is the one with highest c4_count
    edge_costs_rem = []
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 1:
                edge_costs_rem.append((c4_count[u, v], u, v))
    edge_costs_rem.sort(reverse=True)
    
    # The best swap: add the non-edge with lowest cost, remove the edge with highest cost
    # But they might interact (the added edge's C4s might use the removed edge)
    # Let's just try the top candidates
    
    best_swap_net = float('inf')
    best_swap_pair = None
    
    for add_idx in range(min(20, len(non_edges_list))):
        cost_add, u, v = non_edges_list[add_idx]
        for rem_idx in range(min(20, len(edge_costs_rem))):
            cost_rem, a, b = edge_costs_rem[rem_idx]
            # Check interaction: does the new edge (u,v) create C4s that use (a,b)?
            # A C4 using both (u,v) and (a,b) would be u-a-b-v or u-b-a-v etc.
            # This requires: a~u, b~v, a~b (for u-a-b-v) OR a~v, b~u, a~b (for u-b-a-v)
            interaction = 0
            # C4: u-a-b-v: need a~u, b~v, a~b, and a!=v, b!=u
            if A[u, a] == 1 and A[v, b] == 1 and A[a, b] == 1 and a != v and b != u:
                interaction += 1
            # C4: u-b-a-v: need b~u, a~v, a~b, and b!=v, a!=u
            if A[u, b] == 1 and A[v, a] == 1 and A[a, b] == 1 and b != v and a != u:
                interaction += 1
            # Net change = cost_add - cost_rem + interaction (interaction C4s are counted in both)
            # Actually: when we remove (a,b), we lose cost_rem C4s. Some of those C4s also use (u,v)?
            # No - (u,v) is not an edge yet, so no C4 uses both.
            # When we add (u,v), we create cost_add C4s. Some of those might use (a,b)?
            # Yes! If a C4 uses both (u,v) and (a,b), it was not counted in cost_rem (since (u,v) wasn't an edge)
            # but it IS counted in cost_add. So the net is: -cost_rem + cost_add
            # But wait, after removing (a,b), the graph changes, so cost_add might be different.
            # This is getting complex. Let's just use the approximation for now.
            net = cost_add - cost_rem
            if net < best_swap_net:
                best_swap_net = net
                best_swap_pair = ((a, b), (u, v))
    
    # Decide: use best add-only or best swap
    if best_swap_pair is not None and best_swap_net < best_add_cost:
        # Do the swap
        remove_edge = best_swap_pair[0]
        add_edge = best_swap_pair[1]
        edges = [e for e in edges if tuple(sorted(e)) != tuple(sorted(remove_edge))]
        edges.append(list(add_edge))
    else:
        # Just add the best non-edge
        edges.append([best_add_edge[0], best_add_edge[1]])
    
    # Ensure exactly 91 edges
    if len(edges) != 91:
        existing = set()
        for e in edges:
            existing.add(tuple(sorted(e)))
        for i in range(n):
            for j in range(i+1, n):
                if len(edges) >= 91:
                    break
                if (i, j) not in existing:
                    edges.append([i, j])
                    existing.add((i, j))
            if len(edges) >= 91:
                break
    
    return edges[:91]
