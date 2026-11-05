"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a graph with 31 vertices and 91 edges that minimizes the number of C4s.
    
    Strategy: Use the polarity graph of PG(2,5) which has 90 edges and 0 C4s,
    then add the edge that creates the fewest new C4s.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin
    
    n = 31
    q = 5
    
    # Define points in PG(2, 5)
    # We represent points as (x, y, z) where not all are zero, modulo scalar multiples
    # For q=5, we have 31 points
    points = []
    
    # Points with z=0: (x, y, 0) ~ (1, t, 0) for t in GF(5), plus (0, 1, 0)
    # That's 6 points: (1,0,0), (1,1,0), (1,2,0), (1,3,0), (1,4,0), (0,1,0)
    for t in range(5):
        points.append((1, t, 0))
    points.append((0, 1, 0))
    
    # Points with z=1: (x, y, 1) for x,y in GF(5)
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Bilinear form for polarity: B(u,v) = u0*v0 + u1*v1 + u2*v2 (mod 5)
    # Two points are adjacent if B(u,v) == 0 mod 5
    # Note: For a polarity graph, we need to be careful about self-adjacency
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ui, vi, wi = points[i]
            uj, vj, wj = points[j]
            # Bilinear form
            val = (ui * uj + vi * vj + wi * wj) % q
            if val == 0:
                A[i, j] = 1
    
    # Make symmetric and remove self-loops
    A = (A + A.T) // 2
    np.fill_diagonal(A, 0)
    
    # Count edges
    edge_count = A.sum() // 2
    print(f"Polarity graph has {edge_count} edges")
    
    # The polarity graph of PG(2,5) should have 90 edges
    # If it has exactly 90 edges, we add one more edge
    if edge_count == 90:
        # Find the non-edge that creates the fewest new C4s
        A3 = A @ A @ A
        
        # Find the non-edge with minimum A3[u,v]
        min_c4s = float('inf')
        best_edge = None
        
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    c4s = int(A3[u, v])
                    if c4s < min_c4s:
                        min_c4s = c4s
                        best_edge = (u, v)
        
        print(f"Best edge to add: {best_edge}, creates {min_c4s} new C4s")
        
        # Add the edge
        if best_edge is not None:
            u, v = best_edge
            A[u, v] = 1
            A[v, u] = 1
        
        # Verify we have 91 edges
        final_edge_count = A.sum() // 2
        print(f"Final graph has {final_edge_count} edges")
        
        # Now try to improve via local search: swap edges to reduce C4 count
        def count_c4s(adj):
            """Count number of C4 subgraphs using matrix multiplication."""
            # C4 count = (1/2) * sum over all pairs (u,v) of C(common_neighbors(u,v), 2)
            # = (1/2) * sum_{u,v} [C(adj[u]&adj[v], 2)]
            # More efficiently: for each pair of common neighbors, count paths of length 2
            # Actually: number of C4s = (1/2) * sum_{i,j} C((A^2)[i,j], 2) - corrections for edges
            # Simpler: use the formula sum over unordered pairs {u,v} of C(cn(u,v), 2) / 2
            # where cn(u,v) = number of common neighbors
            A2 = adj @ adj
            total = 0
            for u in range(n):
                for v in range(u + 1, n):
                    cn = int(A2[u, v])
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
            return total // 2
        
        current_c4s = count_c4s(A)
        print(f"Initial C4 count after adding edge: {current_c4s}")
        
        best_c4s = current_c4s
        best_A = A.copy()
        
        # Local search: try edge swaps
        deadline_check = time.time()
        while time.time() < deadline_check + 100:
            improved = False
            # Get current edge and non-edge lists
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if best_A[i, j] == 1]
            non_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if best_A[i, j] == 0]
            
            if not non_edge_list:
                break
            
            # Try random swaps
            np.random.shuffle(edge_list)
            np.random.shuffle(non_edge_list)
            
            for e_out in edge_list:
                for e_in in non_edge_list:
                    if set(e_in) == set(e_out):
                        continue
                    
                    # Try the swap
                    u1, v1 = e_out
                    u2, v2 = e_in
                    best_A[u1, v1] = 0
                    best_A[v1, u1] = 0
                    best_A[u2, v2] = 1
                    best_A[v2, u2] = 1
                    
                    new_c4s = count_c4s(best_A)
                    
                    if new_c4s < best_c4s:
                        best_c4s = new_c4s
                        improved = True
                        print(f"Improved to {best_c4s} C4s")
                    else:
                        # Revert
                        best_A[u1, v1] = 1
                        best_A[v1, u1] = 1
                        best_A[u2, v2] = 0
                        best_A[v2, u2] = 0
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        print(f"Final C4 count after local search: {best_c4s}")
        
        # Build the edge list from best_A
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_A[i, j] == 1:
                    edges.append([i, j])
        
        assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
        
        # Verify all vertices are in range
        for e in edges:
            assert 0 <= e[0] <= 30 and 0 <= e[1] <= 30
            assert e[0] != e[1]
        
        # Check for duplicates
        edge_set = set()
        for e in edges:
            key = tuple(sorted(e))
            assert key not in edge_set, f"Duplicate edge: {e}"
            edge_set.add(key)
        
        return edges
    
    else:
        # If the polarity graph doesn't have exactly 90 edges, 
        # we need a fallback strategy
        print(f"Warning: polarity graph has {edge_count} edges, using fallback")
        
        # Fallback: use a greedy approach
        # Start with empty graph, add edges that minimize C4s
        # This is a heuristic and may not be optimal
        
        A_fallback = np.zeros((n, n), dtype=np.int8)
        edges = []
        
        # Try to build a graph with 91 edges
        # Use a random initialization and local search
        np.random.seed(42)
        
        # Initialize with a random sparse graph
        all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        np.random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges[:91]:
            A_fallback[u, v] = 1
            A_fallback[v, u] = 1
            edges.append([u, v])
        
        # Local search: try edge swaps to reduce C4 count
        def count_c4s(adj):
            """Count number of C4 subgraphs."""
            c4_count = 0
            # For each pair of vertices, count common neighbors
            # Each C4 is counted 4 times (once for each pair of opposite vertices)
            # Actually, each C4 has 2 pairs of opposite vertices
            # So C4 count = (1/2) * sum over all pairs (u,v) of C(common_neighbors(u,v), 2)
            for u in range(n):
                for v in range(u + 1, n):
                    # Common neighbors of u and v
                    common = np.sum(adj[u] & adj[v])
                    if common >= 2:
                        c4_count += common * (common - 1) // 2
            return c4_count // 2
        
        current_c4s = count_c4s(A_fallback)
        print(f"Initial C4 count: {current_c4s}")
        
        # Local search
        best_c4s = current_c4s
        best_A = A_fallback.copy()
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            # Try swapping an edge for a non-edge
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A_fallback[i, j] == 1]
            non_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A_fallback[i, j] == 0]
            
            np.random.shuffle(edge_list)
            np.random.shuffle(non_edge_list)
            
            for e_out in edge_list[:50]:
                for e_in in non_edge_list[:50]:
                    # Check that e_in is not the same as e_out
                    if set(e_in) == set(e_out):
                        continue
                    
                    # Try the swap
                    A_fallback[e_out[0], e_out[1]] = 0
                    A_fallback[e_out[1], e_out[0]] = 0
                    A_fallback[e_in[0], e_in[1]] = 1
                    A_fallback[e_in[1], e_in[0]] = 1
                    
                    new_c4s = count_c4s(A_fallback)
                    
                    if new_c4s < best_c4s:
                        best_c4s = new_c4s
                        best_A = A_fallback.copy()
                        improved = True
                        # Keep the swap
                    else:
                        # Revert
                        A_fallback[e_out[0], e_out[1]] = 1
                        A_fallback[e_out[1], e_out[0]] = 1
                        A_fallback[e_in[0], e_in[1]] = 0
                        A_fallback[e_in[1], e_in[0]] = 0
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        print(f"Final C4 count after local search: {best_c4s}")
        
        # Rebuild edge list from best_A
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_A[i, j] == 1:
                    edges.append([i, j])
        
        assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
        
        return edges

# EVOLVE-BLOCK-END
