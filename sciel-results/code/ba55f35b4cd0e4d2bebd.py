import sys
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Identify all edges
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
    
    # 2. Identify all squares (2-faces)
    # A square is defined by 2 dimensions i, j and a base vertex u 
    # where the bits i and j are 0? No, just fixed bits.
    # Simpler: A square is a set of 4 edges.
    # We can find squares by iterating over pairs of dimensions (i, j) and base vertices.
    # For fixed i, j, the square is formed by varying bits i and j.
    # The 4 vertices are u, u^i, u^j, u^i^j where u has i,j bits 0?
    # Actually, for any u, the edges (u, u^i), (u^j, u^i^j), (u, u^j), (u^i, u^i^j) form a square?
    # No, (u, u^i) and (u, u^j) share u.
    # The square is u - u^i - u^i^j - u^j - u.
    # Edges: (u, u^i), (u^i, u^i^j), (u^i^j, u^j), (u^j, u).
    
    squares = []
    square_to_edges = {}
    edge_to_squares = [set() for _ in range(n_edges)]
    
    # To avoid duplicates, we can iterate u from 0 to n_vertices-1
    # and only consider squares where u is the "minimum" vertex in some sense?
    # Or just collect all and deduplicate.
    # There are C(7,2) * 2^5 = 21 * 32 = 672 squares.
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of fixed bits
            # We can just iterate u from 0 to n_vertices-1
            # and check if the square formed by i,j at u is valid
            # To avoid duplicates, we require that u has bit i=0 and bit j=0?
            # No, if we fix i,j to be the varying bits, the square is unique for each assignment of other bits.
            # The "base" u should have bits i and j set to 0 to uniquely identify the square?
            # Yes. If we vary i and j, the square is determined by the values of the other d-2 bits.
            # So we iterate u such that bit i is 0 and bit j is 0.
            for u in range(n_vertices):
                if (u >> i) & 1 == 0 and (u >> j) & 1 == 0:
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    
                    # The 4 edges
                    e1 = (min(u, v1), max(u, v1))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v3, v2), max(v3, v2))
                    e4 = (min(u, v2), max(u, v2))
                    
                    sq_edges = sorted([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]])
                    sq_key = tuple(sq_edges)
                    
                    if sq_key not in square_to_edges:
                        sq_idx = len(squares)
                        squares.append(sq_key)
                        square_to_edges[sq_key] = sq_idx
                        for e_idx in sq_edges:
                            edge_to_squares[e_idx].add(sq_idx)
    
    n_squares = len(squares)
    
    # 3. Prepare bitmasks
    # square_mask[e] : bitmask of squares covered by edge e
    square_mask = [0] * n_edges
    for e in range(n_edges):
        mask = 0
        for s in edge_to_squares[e]:
            mask |= (1 << s)
        square_mask[e] = mask
    
    # edges_in_square[s] : list of edge indices
    edges_in_square = [[] for _ in range(n_squares)]
    for s in range(n_squares):
        for e in squares[s]:
            edges_in_square[s].append(e)
    
    # 4. Greedy Initial Solution
    # We want to MINIMIZE removed edges.
    # Greedy: Pick edge covering most uncovered squares.
    
    def greedy_hitting_set():
        uncovered = (1 << n_squares) - 1
        removed = []
        while uncovered:
            best_e = -1
            best_count = 0
            for e in range(n_edges):
                count = bin(uncovered & square_mask[e]).count('1')
                if count > best_count:
                    best_count = count
                    best_e = e
            if best_e == -1: break
            removed.append(best_e)
            uncovered &= ~square_mask[best_e]
        return len(removed)
    
    best_removed = greedy_hitting_set()
    
    # 5. Branch and Bound
    # To avoid recursion depth issues and speed up, we use a stack or limit depth.
    # Given the small branching factor (4) and strong pruning, recursion should be fine.
    
    import math
    
    def popcount(x):
        return bin(x).count('1')
        
    # Optimization: Precompute popcount for masks? No, too big.
    # But we can use int.bit_count() in Python 3.8+
    
    def solve(uncovered, depth):
        nonlocal best_removed
        
        if uncovered == 0:
            if depth < best_removed:
                best_removed = depth
            return
            
        if depth >= best_removed:
            return
            
        # Lower bound: ceil(popcount(uncovered) / 6)
        # Max coverage per edge is 6.
        lb = (popcount(uncovered) + 5) // 6
        if depth + lb >= best_removed:
            return
            
        # Select square s
        # Pick the lowest set bit
        s = (uncovered & -uncovered).bit_length() - 1
        
        candidates = edges_in_square[s]
        
        # Sort candidates by coverage of 'uncovered'
        # This helps prune earlier
        candidates.sort(key=lambda e: popcount(square_mask[e] & uncovered), reverse=True)
        
        for e in candidates:
            new_uncovered = uncovered & ~square_mask[e]
            solve(new_uncovered, depth + 1)
            
    # Start with all squares uncovered
    full_mask = (1 << n_squares) - 1
    solve(full_mask, 0)
    
    # The result is n_edges - best_removed
    max_edges = n_edges - best_removed
    
    # Construct the edge list
    # We need to know WHICH edges were removed to return the valid set.
    # The B&B above only finds the NUMBER.
    # We need to modify it to store the solution.
    
    # Let's re-run with solution tracking or modify the solver.
    # For simplicity and speed, if we just need the count to certify, we return a valid set.
    # But we must return a LIST of edges.
    
    # We need to recover the set of removed edges.
    
    best_solution = None
    
    def solve_with_track(uncovered, depth, removed_set):
        nonlocal best_removed, best_solution
        
        if uncovered == 0:
            if depth < best_removed:
                best_removed = depth
                best_solution = set(removed_set)
            return
            
        if depth >= best_removed:
            return
            
        lb = (popcount(uncovered) + 5) // 6
        if depth + lb >= best_removed:
            return
            
        s = (uncovered & -uncovered).bit_length() - 1
        candidates = edges_in_square[s]
        candidates.sort(key=lambda e: popcount(square_mask[e] & uncovered), reverse=True)
        
        for e in candidates:
            new_uncovered = uncovered & ~square_mask[e]
            removed_set.add(e)
            solve_with_track(new_uncovered, depth + 1, removed_set)
            removed_set.remove(e)
            
    solve_with_track(full_mask, 0, set())
    
    if best_solution is None:
        # Fallback to greedy if B&B failed to find better (shouldn't happen)
        # Re-run greedy to get the set
        uncovered = full_mask
        removed_set = set()
        while uncovered:
            best_e = -1
            best_count = 0
            for e in range(n_edges):
                count = popcount(uncovered & square_mask[e])
                if count > best_count:
                    best_count = count
                    best_e = e
            if best_e == -1: break
            removed_set.add(best_e)
            uncovered &= ~square_mask[best_e]
        best_solution = removed_set
        best_removed = len(removed_set)
        
    # Create list of kept edges
    kept_edges = []
    for i, (u, v) in enumerate(edges):
        if i not in best_solution:
            kept_edges.append([u, v])
            
    return kept_edges

# EVOLVE-BLOCK-END
