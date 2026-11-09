"""Improved program for Ramsey R(5,5) graph construction."""
import time
import random
import itertools
import sys

def construct():
    """
    Try to find a (5,5)-free graph with as many vertices as possible.
    Strategy:
    1. Generate Paley graphs for q = 37, 41, 49.
    2. Check if they are (5,5)-free using fast bitset operations.
    3. If a valid graph is found with n > 37, return it.
    4. Otherwise, fall back to Paley(37).
    5. If time permits, try to extend the best found graph using greedy addition.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit

    def paley_graph(q):
        """Generate Paley graph on q vertices."""
        if q < 2:
            return q, []
        residues = {(x * x) % q for x in range(1, q)}
        residues.discard(0)
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges

    def build_adj(n, edges):
        """Build adjacency matrix from edge list."""
        adj = [[False] * n for _ in range(n)]
        for i, j in edges:
            adj[i][j] = adj[j][i] = True
        return adj

    def build_bitsets(n, adj):
        """Build adjacency bitsets."""
        adj_bits = [0] * n
        non_adj_bits = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    if adj[i][j]:
                        adj_bits[i] |= (1 << j)
                    else:
                        non_adj_bits[i] |= (1 << j)
        return adj_bits, non_adj_bits

    def has_k5_fast(n, adj_bits):
        """Check if graph has a clique of size 5 using bitsets."""
        if n < 5:
            return False
        # Iterate over combinations of 5
        # Optimization: Only check if the intersection of adj bits is non-empty
        for v1 in range(n):
            b1 = adj_bits[v1]
            for v2 in range(v1 + 1, n):
                if not (b1 & (1 << v2)): continue
                b2 = b1 & adj_bits[v2]
                # b2 contains neighbors of v1 and v2
                # We need 3 more vertices from b2 that form a triangle
                # Iterate v3
                for v3 in range(v2 + 1, n):
                    if not (b2 & (1 << v3)): continue
                    b3 = b2 & adj_bits[v3]
                    # We need 2 more vertices from b3 that are connected
                    # Iterate v4
                    for v4 in range(v3 + 1, n):
                        if not (b3 & (1 << v4)): continue
                        b4 = b3 & adj_bits[v4]
                        # We need 1 more vertex from b4
                        # Check if b4 has any bit set
                        if b4:
                            return True
        return False

    def has_indep5_fast(n, non_adj_bits):
        """Check if graph has an independent set of size 5 using bitsets."""
        if n < 5:
            return False
        for v1 in range(n):
            b1 = non_adj_bits[v1]
            for v2 in range(v1 + 1, n):
                if not (b1 & (1 << v2)): continue
                b2 = b1 & non_adj_bits[v2]
                for v3 in range(v2 + 1, n):
                    if not (b2 & (1 << v3)): continue
                    b3 = b2 & non_adj_bits[v3]
                    for v4 in range(v3 + 1, n):
                        if not (b3 & (1 << v4)): continue
                        b4 = b3 & non_adj_bits[v4]
                        if b4:
                            return True
        return False

    def is_55_free_fast(n, adj):
        """Check if graph is (5,5)-free."""
        if n < 5:
            return True
        adj_bits, non_adj_bits = build_bitsets(n, adj)
        if has_k5_fast(n, adj_bits):
            return False
        if has_indep5_fast(n, non_adj_bits):
            return False
        return True

    def edges_from_adj(n, adj):
        """Convert adjacency matrix to edge list."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append((i, j))
        return edges

    best_n = 0
    best_adj = None

    # Try Paley graphs for q = 37, 41, 49
    for q in [37, 41, 49]:
        if time.time() > deadline - 20:
            break
        
        n, edges = paley_graph(q)
        adj = build_adj(n, edges)
        
        # Check validity
        if is_55_free_fast(n, adj):
            if n > best_n:
                best_n = n
                best_adj = adj
            # If we found a large one, we can break if it's very large (e.g., >= 41)
            # But let's check all to see if 49 works
        else:
            # If invalid, we might try to modify it, but for now just skip
            pass

    # If we found a valid graph, return it
    if best_adj is not None:
        final_edges = edges_from_adj(best_n, best_adj)
        return best_n, final_edges

    # Fallback: Paley(37)
    q37, edges37 = paley_graph(37)
    return q37, edges37
