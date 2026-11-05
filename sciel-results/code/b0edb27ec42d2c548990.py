"""Polarity graph C4-free core + optimized extra edge + local search for C4 minimization."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Construct polarity graph on PG(2,5)
    # Points of PG(2,5) are 1D subspaces of F_5^3
    # There are (5^3 - 1)/(5 - 1) = 31 points
    # A polarity graph connects points if they are orthogonal under the polarity
    
    # Generate all 31 points in PG(2,5) - projective points
    points = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                idx = -1
                for i in range(3):
                    if vec[i] != 0:
                        idx = i
                        break
                inv = pow(vec[idx], -1, 5) if vec[idx] != 0 else 1
                vec = [(v * inv) % 5 for v in vec]
                key = tuple(vec)
                if key not in points:
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define a polarity: a symmetric bilinear form
    # Use the standard polarity: x1*x2 + x2*x3 + x3*x1 (or similar)
    # Actually, let's use the Hermitian-like form: x1^2 + x2^2 + x3^2 = 0 in F5
    # Wait, for a polarity we need a symmetric bilinear form B such that B(x,x)=0 has exactly 31 solutions (the absolute points)
    
    # Let's use the form: B((x1,x2,x3),(y1,y2,y3)) = x1*y2 + x2*y1 + x2*y3 + x3*y2 + x3*y1 + x1*y3
    # That's the "symmetric" form. Actually let me use a standard polarity of PG(2,5).
    
    # For q=5 (odd), a standard polarity is given by the symmetric matrix:
    # [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    # Two points are adjacent if B(p, q) = 0
    
    B_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int)
    
    def are_orthogonal(p, q):
        # B(p, q) = p^T * B_matrix * q mod 5
        val = 0
        for i in range(3):
            for j in range(3):
                val += p[i] * B_matrix[i, j] * q[j]
        return (val % 5) == 0
    
    # Build the polarity graph adjacency
    polarity_edges = []
    polarity_adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if are_orthogonal(points[i], points[j]):
                polarity_edges.append([i, j])
                polarity_adj[i, j] = True
                polarity_adj[j, i] = True
    
    # The polarity graph on PG(2,5) should have exactly 90 edges (it's C4-free)
    # Let's verify
    num_polarity_edges = len(polarity_edges)
    
    # If we don't have exactly 90, we need to adjust
    # For now, let's work with what we have
    
    if num_polarity_edges < 90:
        # Add random edges until we have 90
        existing = set(tuple(sorted(e)) for e in polarity_edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for e in all_possible:
            if len(polarity_edges) >= 90:
                break
            if e not in existing:
                polarity_edges.append(list(e))
                existing.add(e)
                polarity_adj[e[0], e[1]] = True
                polarity_adj[e[1], e[0]] = True
    elif num_polarity_edges > 90:
        # Remove random edges to get to 90
        random.shuffle(polarity_edges)
        polarity_edges = polarity_edges[:90]
        polarity_adj = np.zeros((n, n), dtype=bool)
        for e in polarity_edges:
            polarity_adj[e[0], e[1]] = True
            polarity_adj[e[1], e[0]] = True
    
    # Now we have 90 edges. Add the 91st edge optimally.
    # Count C4s for each possible additional edge
    def count_c4s(adj):
        """Count number of 4-cycles in the graph."""
        A = adj.astype(int)
        A2 = A @ A
        # Number of C4s = (sum of squared off-diagonal entries of A^2 - sum of degrees) / 8
        # Actually: number of closed walks of length 4 = sum(A^4)
        # Number of C4s (as subgraphs) = (sum_{i≠j} (A^2)_{ij}^2 - sum_i deg(i)) / 8
        # Let me use: C4 count = (tr(A^4) - 2*sum_{i≠j}(A^2)_{ij} - 2*sum_i deg(i)^2 + 2*sum_i deg(i)) / 8
        # Simpler: for each pair of vertices (i,j), the number of common neighbors is (A^2)_{ij}
        # A 4-cycle is determined by two non-adjacent pairs that share 2 common neighbors
        # Actually, the number of C4s = sum over all pairs {i,j} of C(common_neighbors(i,j), 2)
        # but we need to be careful about what counts as a C4
        
        # Let's use the formula: number of 4-cycles = (1/8) * sum_{i≠j} (A^2)_{ij}^2 - (1/4) * sum_i deg(i)^2 + (1/2) * sum_i deg(i)
        # Wait, let me think more carefully.
        
        # The number of (not necessarily induced) 4-cycles as subgraphs:
        # For each unordered pair of vertices {u,v}, let c(u,v) = number of common neighbors
        # Then the number of 4-cycles containing edge {u,v} is... no that's not right either.
        
        # A 4-cycle is a sequence u-v-w-x-u where u,v,w,x are distinct.
        # Number of such cycles = (1/8) * sum over all ordered 4-tuples (u,v,w,x) with u≠v≠w≠x≠u and v≠x
        # of [A(u,v)*A(v,w)*A(w,x)*A(x,u)]
        # = (1/8) * (tr(A^4) - 2*sum_{i≠j}(A^2)_{ij} - 2*sum_i deg(i)^2 + 2*sum_i deg(i))
        
        # Actually, let me just use: C4_count = sum over all unordered pairs {i,j} of C(common_neighbors(i,j), 2)
        # This counts each 4-cycle exactly once? No...
        
        # Let me think again. A 4-cycle u-v-w-x-u has two pairs of opposite vertices: {u,w} and {v,x}.
        # For pair {u,w}, the common neighbors include v and x (and possibly others).
        # So if I sum C(common_neighbors(i,j), 2) over all pairs {i,j}, I count each 4-cycle twice
        # (once for each pair of opposite vertices).
        # But wait, if there are other common neighbors, I might count additional "cycles" that aren't simple 4-cycles.
        # Actually no - if i and j have k common neighbors, then C(k,2) counts the number of ways to pick two
        # common neighbors, which gives a 4-cycle i-a-j-b-i (or i-b-j-a-i). Each such choice gives exactly one 4-cycle.
        # And each 4-cycle is counted exactly twice (once for each pair of opposite vertices).
        
        A2 = A @ A
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                c = A2[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    # Find the best edge to add
    existing_edges = set(tuple(sorted(e)) for e in polarity_edges)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edge = None
    best_c4_count = float('inf')
    
    # Precompute A^2 for the current graph
    A = polarity_adj.astype(int)
    A2 = A @ A
    
    for i, j in all_possible:
        if (i, j) in existing_edges:
            continue
        # If we add edge (i,j), how many new C4s are created?
        # A new C4 must use the new edge (i,j). So it's of the form i-j-x-y-i where x,y are neighbors of both i and j.
        # Wait no: i-j-x-y-i means j-x, x-y, y-i must be edges.
        # So x must be a neighbor of j, y must be a neighbor of i, and x-y must be an edge.
        # Also x≠i, y≠j, x≠y.
        # Actually, the C4 is i-j-x-y-i, so edges are (i,j), (j,x), (x,y), (y,i).
        # x is a common neighbor of j and... no, x is neighbor of j, y is neighbor of i, and (x,y) is an edge.
        
        # Number of new C4s = number of pairs (x,y) such that:
        # - x is neighbor of j (x ≠ i)
        # - y is neighbor of i (y ≠ j)  
        # - (x,y) is an edge
        # - x ≠ y
        # - x ≠ i, y ≠ j (already ensured)
        
        # Wait, I also need to make sure this isn't already a C4. Since (i,j) wasn't an edge before,
        # any C4 using (i,j) is new.
        
        new_c4s = 0
        for x in range(n):
            if x == i or not A[j, x]:
                continue
            for y in range(n):
                if y == j or y == x or not A[i, y] or not A[x, y]:
                    continue
                new_c4s += 1
        
        # Each C4 is counted once here (fixed edge i-j, then x is neighbor of j, y is neighbor of i, x-y edge)
        # Actually each C4 i-j-x-y-i is counted exactly once with this fixed ordering.
        
        if new_c4s < best_c4_count:
            best_c4_count = new_c4s
            best_edge = (i, j)
    
    if best_edge is None:
        # Fallback: add any edge
        for i, j in all_possible:
            if (i, j) not in existing_edges:
                best_edge = (i, j)
                break
    
    if best_edge:
        polarity_edges.append(list(best_edge))
        polarity_adj[best_edge[0], best_edge[1]] = True
        polarity_adj[best_edge[1], best_edge[0]] = True
    
    # Now we have 91 edges. Verify.
    assert len(polarity_edges) == 91, f"Expected 91 edges, got {len(polarity_edges)}"
    
    # Local search: try swaps to reduce C4 count
    A = polarity_adj.astype(int)
    current_c4s = count_c4s(A)
    
    def try_swap(adj, edge1, edge2, time_limit):
        """Try swapping edge1 for edge2. Return new adj if it improves, else None."""
        if adj[edge1[0], edge1[1]] == 0:
            return None
        if adj[edge2[0], edge2[1]] == 1:
            return None
        
        new_adj = adj.copy()
        new_adj[edge1[0], edge1[1]] = 0
        new_adj[edge1[1], edge1[0]] = 0
        new_adj[edge2[0], edge2[1]] = 1
        new_adj[edge2[1], edge2[0]] = 1
        
        new_c4s = count_c4s(new_adj)
        return new_adj, new_c4s
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        A = polarity_adj.astype(int)
        
        # Try all possible swaps
        # For each edge in the graph, try removing it and adding a non-edge
        edges_in_graph = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j]]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not A[i, j]]
        
        random.shuffle(edges_in_graph)
        random.shuffle(non_edges)
        
        for e1 in edges_in_graph:
            if time.time() > deadline:
                break
            for e2 in non_edges:
                if time.time() > deadline:
                    break
                result = try_swap(A, e1, e2, deadline)
                if result is not None:
                    new_adj, new_c4s = result
                    if new_c4s < current_c4s:
                        polarity_adj = new_adj.astype(bool)
                        current_c4s = new_c4s
                        improved = True
                        A = new_adj
                        # Update edges and non-edges
                        edges_in_graph = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j]]
                        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not A[i, j]]
                        break
    
    # Build final edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if polarity_adj[i, j]:
                final_edges.append([i, j])
    
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    
    return final_edges
