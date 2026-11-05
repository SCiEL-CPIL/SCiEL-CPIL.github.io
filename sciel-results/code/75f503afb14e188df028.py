"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n = 31
    m = 91
    q = 5
    
    # Build the polarity graph of PG(2, q) for q=5
    # Points in PG(2, q): (x, y, z) up to scalar, not all zero
    # For q=5: q^2 + q + 1 = 31 points
    
    # Enumerate all points in PG(2, 5)
    points = []
    for z in range(5):
        for y in range(5):
            for x in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                pt = [x, y, z]
                for i in range(3):
                    if pt[i] != 0:
                        inv = pow(pt[i], q - 2, q)
                        pt = [(pt[j] * inv) % q for j in range(3)]
                        break
                if pt not in points:
                    points.append(pt)
    
    # Should have 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define the polarity: x^2 + y^2 + z^2 = 0 (or use a non-degenerate quadratic form)
    # For q odd, use the form x^2 + y^2 + z^2 = 0
    # Two points are adjacent if they are orthogonal under the bilinear form
    
    # Actually, for a polarity graph, we use a symmetric bilinear form B(u,v) = u_0*v_0 + u_1*v_1 + u_2*v_2
    # u ~ v if B(u,v) = 0
    
    def is_orthogonal(u, v):
        return (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q == 0
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if is_orthogonal(points[i], points[j]):
                A[i, j] = 1
                A[j, i] = 1
    
    # Count edges in the polarity graph
    edge_count = int(np.sum(A) // 2)
    
    # The polarity graph of PG(2,5) should have exactly q(q+1)/2 * (q+1) = ... 
    # Let's just work with what we have and adjust to get exactly 91 edges
    
    # Get list of edges
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                edges.add((i, j))
    
    # We need exactly 91 edges
    # If we have more, remove some; if fewer, add some
    # The polarity graph for q=5 should have q(q+1)/2 = 15 edges per vertex? No...
    # Actually, in PG(2,q), each point is orthogonal to exactly q+1 points (including itself if self-orthogonal)
    # So degree is q+1 or q, giving total edges around 31*6/2 = 93 or so
    
    # Let's count
    if edge_count > m:
        # Remove edges to get to exactly m
        edge_list = list(edges)
        random.seed(42)
        random.shuffle(edge_list)
        to_remove = edge_list[:edge_count - m]
        for e in to_remove:
            edges.remove(e)
            A[e[0], e[1]] = 0
            A[e[1], e[0]] = 0
    elif edge_count < m:
        # Add edges to get to exactly m
        # Add edges that minimize C4 count
        all_possible = set()
        for i in range(n):
            for j in range(i+1, n):
                all_possible.add((i, j))
        
        missing = all_possible - edges
        missing_list = list(missing)
        random.seed(42)
        random.shuffle(missing_list)
        
        # Add edges that create fewest C4s
        for e in missing_list:
            if len(edges) >= m:
                break
            # Check how many C4s this edge would create
            u, v = e
            common_neighbors = 0
            for w in range(n):
                if A[u, w] == 1 and A[v, w] == 1:
                    common_neighbors += 1
            # Prefer edges with fewer common neighbors
            edges.add(e)
            A[u, v] = 1
            A[v, u] = 1
    
    # Now we have exactly 91 edges
    assert len(edges) == m, f"Expected {m} edges, got {len(edges)}"
    
    # Count C4s in the current graph
    def count_c4s(adj):
        """Count number of 4-cycles in the graph."""
        n = adj.shape[0]
        # A 4-cycle is u-v-w-x-u where u,v,w,x are distinct
        # Count by: for each pair of vertices (u,w), count common neighbors
        # Each C4 contributes 2 to the sum over pairs of common neighbors
        # Actually, each C4 u-v-w-x has pairs (u,w) and (v,x) as opposite vertices
        # For each pair (u,w), let c(u,w) = number of common neighbors
        # Then number of C4s = sum over all pairs (u,w) of C(c(u,w), 2) / 2
        # Wait, let me think more carefully.
        
        # A 4-cycle u-v-w-x-u: the pairs of opposite vertices are (u,w) and (v,x)
        # For a fixed pair (u,w), each pair of common neighbors (v,x) forms a 4-cycle u-v-w-x-u
        # But we need v and x to be distinct from u,w and from each other
        # Also, each 4-cycle is counted twice (once for each pair of opposite vertices)
        
        total = 0
        for u in range(n):
            for w in range(u+1, n):
                if adj[u, w] == 1:
                    continue  # Skip if u,w are adjacent (would create triangles, not 4-cycles)
                # Count common neighbors
                common = np.sum(adj[u] & adj[w])
                # Each pair of common neighbors forms a 4-cycle
                total += common * (common - 1) // 2
        return total // 2  # Each 4-cycle counted twice
    
    current_c4 = count_c4s(A)
    
    # Try to improve via edge swaps
    # An edge swap: remove edge (a,b), add edge (c,d)
    # This keeps the edge count constant
    
    def c4_change_remove_add(A, rem_u, rem_v, add_u, add_v):
        """Estimate change in C4 count when removing (rem_u,rem_v) and adding (add_u,add_v)."""
        # This is approximate - for exact, we'd need to recompute
        # For now, use a heuristic: 
        # Removing an edge reduces C4s by the number of C4s containing that edge
        # Adding an edge increases C4s by the number of C4s the new edge would form
        
        # Number of C4s containing edge (u,v):
        # For each w != u,v, if (u,w) and (v,w) are edges... no
        # A C4 containing edge (u,v) has form u-v-w-x-u
        # So w must be a common neighbor of... no, w is adjacent to v, x is adjacent to u and w
        # Actually: u-v-w-x-u means v-w edge, w-x edge, x-u edge
        # So for edge (u,v): count pairs (w,x) where v-w, w-x, x-u are all edges
        # That is: for each w in N(v)\{u}, for each x in N(u)\{v}, if w-x is an edge
        
        # Simpler heuristic: 
        # C4s through edge (u,v) ≈ |N(u) ∩ N(v)| * something
        # Actually, exact count: for edge (u,v), a 4-cycle u-v-w-x-u requires
        # w ∈ N(v)\{u}, x ∈ N(u)\{v}, and (w,x) ∈ E
        # This is the number of paths of length 2 from v to u not using edge (u,v)
        # which equals (A^2)[u,v] - 1 (since A^2[u,v] counts paths of length 2, one of which is the direct edge... no)
        
        # Let's use: C4s containing edge (u,v) = number of w such that (v,w) and (w,x) and (x,u) for some x
        # This is complex. Let's just use a simpler proxy.
        
        # Proxy: use the number of common neighbors as a proxy for C4 contribution
        common_rem = np.sum(A[rem_u] & A[rem_v])
        common_add = np.sum(A[add_u] & A[add_v])
        
        # Removing edge with many common neighbors is good (reduces C4s more)
        # Adding edge with few common neighbors is good (adds fewer C4s)
        return common_add - common_rem
    
    # Local search: try edge swaps
    best_c4 = current_c4
    best_A = A.copy()
    best_edges = set(edges)
    
    # Build edge list for fast lookup
    edge_set = set(best_edges)
    
    # Precompute neighbor sets
    neighbors = [set(np.where(best_A[i] == 1)[0]) for i in range(n)]
    
    # Try random edge swaps
    all_edges = list(edge_set)
    non_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) not in edge_set:
                non_edges.add((i,j))
    
    non_edge_list = list(non_edges)
    
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        random.shuffle(all_edges)
        random.shuffle(non_edge_list)
        
        for rem_e in all_edges:
            if time.time() > deadline - 5:
                break
            for add_e in non_edge_list:
                if time.time() > deadline - 5:
                    break
                # Skip if add_e is same as rem_e or shares a vertex in a way that's invalid
                if add_e == rem_e:
                    continue
                # Check that adding add_e doesn't create a duplicate
                if add_e in edge_set:
                    continue
                
                # Estimate improvement
                delta = c4_change_remove_add(best_A, rem_e[0], rem_e[1], add_e[0], add_e[1])
                
                if delta < 0:  # C4 count decreases
                    # Apply the swap
                    ru, rv = rem_e
                    au, av = add_e
                    
                    best_A[ru, rv] = 0
                    best_A[rv, ru] = 0
                    best_A[au, av] = 1
                    best_A[av, au] = 1
                    
                    best_edges.remove(rem_e)
                    best_edges.add(add_e)
                    edge_set = best_edges
                    
                    # Update neighbors
                    neighbors[ru].discard(rv)
                    neighbors[rv].discard(ru)
                    neighbors[au].add(av)
                    neighbors[av].add(au)
                    
                    # Update lists
                    all_edges.remove(rem_e)
                    all_edges.add(add_e)
                    non_edge_list.remove(add_e)
                    non_edge_list.append(rem_e)
                    
                    # Recompute C4 count
                    current_c4 = count_c4s(best_A)
                    best_c4 = current_c4
                    improvement_found = True
                    break
            if improvement_found:
                break
    
    # Return the best edges found
    result = [[u, v] for u, v in sorted(best_edges)]
    assert len(result) == 91
    return result
# EVOLVE-BLOCK-END
