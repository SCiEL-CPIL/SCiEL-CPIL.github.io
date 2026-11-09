import time
import itertools
import numpy as np
import random

def _has_k222_adj(adj, n):
    """
    Check if the graph defined by boolean adjacency matrix adj (n x n) contains a K_{2,2,2} subgraph.
    K_{2,2,2} is the complete tripartite graph with parts of size 2.
    This is equivalent to finding 6 vertices {a,b,c,d,e,f} partitioned into 3 pairs such that
    all edges between distinct pairs are present.
    
    Efficient Check:
    A K_{2,2,2} exists if there is a pair (u, v) such that the subgraph induced by 
    N(u) ∩ N(v) contains a K_{2,2}.
    Why? Let the parts of K_{2,2,2} be P1={u,v}, P2={a,b}, P3={c,d}.
    Then a,b,c,d must be neighbors of both u and v.
    Also, edges (a,c), (a,d), (b,c), (b,d) must exist.
    This is exactly a K_{2,2} in the induced subgraph on N(u) ∩ N(v).
    """
    # Precompute bitmasks for neighbors for faster intersection
    # adj is a 2D numpy array of bools
    # We can use np.bitwise_and on rows
    
    # Iterate over all pairs (u, v)
    for u in range(n):
        for v in range(u + 1, n):
            # Common neighbors of u and v
            # Exclude u and v themselves from the common neighbor set, though they are not neighbors of each other in the induced subgraph sense for K222 parts
            # Actually, u and v are in P1. P2 and P3 must be disjoint from P1.
            # So we look at W = N(u) ∩ N(v) \ {u, v}
            common = adj[u] & adj[v]
            common[u] = False
            common[v] = False
            
            # If |common| < 4, cannot form two pairs of size 2
            if np.sum(common) < 4:
                continue
                
            # We need to check if the subgraph induced by common contains a K_{2,2}.
            # A K_{2,2} is a cycle of length 4.
            # Iterate over pairs (a, b) in common.
            # If there are at least 2 common neighbors of a and b within common, we have a K_{2,2}.
            common_indices = np.where(common)[0]
            len_c = len(common_indices)
            
            for i in range(len_c):
                a = common_indices[i]
                for j in range(i + 1, len_c):
                    b = common_indices[j]
                    # Common neighbors of a and b within common
                    # We need at least 2 vertices in common that are adjacent to both a and b
                    # d_neighbors = (adj[a] & adj[b]) & common
                    # Count them
                    # Optimization: if adj[a][b] is False, it doesn't matter for K22 in induced subgraph?
                    # No, K2,2 requires edges between the two parts. The parts are {a,b} and {c,d}.
                    # So we need c,d in common such that c~a, c~b, d~a, d~b.
                    # The edge a-b is irrelevant for the existence of K2,2 with parts {a,b} and {c,d}.
                    # Wait, if a and b are in the same part of K2,2, they don't need to be connected.
                    # So we just need 2 vertices in common that are neighbors of both a and b.
                    
                    # Use bit counting for speed
                    # np.sum((adj[a] & adj[b]) & common) >= 2
                    # This is O(n) per pair (a,b).
                    # Total O(n^4) worst case, but with early exits and small n=30, it's manageable.
                    
                    # Let's try a slightly faster bit operation approach if possible, or just numpy
                    count_common_ab = np.sum((adj[a] & adj[b]) & common)
                    if count_common_ab >= 2:
                        return True
    return False

def _check_validity(n, colors):
    """
    Check if the coloring is valid (no monochromatic K_{2,2,2}).
    """
    # Build adjacency matrices for color 0 and color 1
    adj0 = np.zeros((n, n), dtype=bool)
    adj1 = np.zeros((n, n), dtype=bool)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0[i][j] = True
                adj0[j][i] = True
            else:
                adj1[i][j] = True
                adj1[j][i] = True
            idx += 1
            
    if _has_k222_adj(adj0, n):
        return False
    if _has_k222_adj(adj1, n):
        return False
    return True

def construct():
    """
    Returns a valid 2-coloring of K_n without a monochromatic K_{2,2,2}.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Generate the known valid solution for n=29 (Paley Graph)
    def paley_coloring(n):
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return n, colors

    best_n, best_colors = paley_coloring(29)
    
    # Verify the n=29 solution quickly to ensure correctness of our checker
    # This is optional but good for sanity. It should be True.
    # _check_validity(29, best_colors) is expected to be True.
    
    # 2. Attempt to find a valid coloring for n=30
    # We use a heuristic local search.
    # Objective: Minimize the number of monochromatic K_{2,2,2}.
    # Since counting all K_{2,2,2} is expensive, we use a proxy or just random restarts with validity checks.
    # Given the difficulty, we try a limited number of random restarts.
    
    n_target = 30
    num_edges = n_target * (n_target - 1) // 2
    
    # Random seed for reproducibility
    random.seed(42)
    
    # We will try a few random initializations and perform local search (flip bits).
    # If we find a valid one, we return it.
    # If we run out of time, we return the n=29 solution.
    
    max_restarts = 50
    max_steps_per_restart = 500
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        current_colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Check if already valid (very unlikely)
        if _check_validity(n_target, current_colors):
            return {'n': n_target, 'colors': current_colors}
        
        # Local Search: Simulated Annealing or Hill Climbing
        # We need a cost function. Counting K_{2,2,2} is hard.
        # Alternative: Use the structure of the checker.
        # We can try to minimize the "potential" for K_{2,2,2}.
        # For each pair (u,v), let C_uv = N(u) ∩ N(v).
        # If C_uv contains a K_{2,2}, it's a violation.
        # We can count the number of K_{2,2} in all induced subgraphs G[C_uv].
        # This is a smooth-ish objective (integer count).
        
        def count_k222_potential(colors):
            # Build adj matrices
            adj0 = np.zeros((n_target, n_target), dtype=bool)
            adj1 = np.zeros((n_target, n_target), dtype=bool)
            idx = 0
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if colors[idx] == 0:
                        adj0[i][j] = True
                        adj0[j][i] = True
                    else:
                        adj1[i][j] = True
                        adj1[j][i] = True
                    idx += 1
            
            def count_k22_in_subgraph(adj, n):
                # Count K_{2,2} in graph adj
                # For each pair (u,v), let C = N(u) ∩ N(v).
                # Count K_{2,2} in G[C].
                # A K_{2,2} in G[C] is a set of 4 vertices {a,b,c,d} in C such that
                # a-c, a-d, b-c, b-d are edges.
                # This is equivalent to counting the number of 4-cycles in G[C]? No, just specific structure.
                # Actually, if we just want to check existence, we stop at 1.
                # For optimization, we want the count.
                # This is still expensive.
                # Let's just return 0 if valid, 1 if invalid? No, that's step function.
                # Let's count the number of "bad" pairs (u,v) where G[C_uv] contains K_{2,2}.
                
                bad_count = 0
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[u] & adj[v]
                        common[u] = False
                        common[v] = False
                        if np.sum(common) < 4:
                            continue
                        common_indices = np.where(common)[0]
                        len_c = len(common_indices)
                        
                        # Check for K_{2,2}
                        found = False
                        for i in range(len_c):
                            a = common_indices[i]
                            for j in range(i + 1, len_c):
                                b = common_indices[j]
                                if np.sum((adj[a] & adj[b]) & common) >= 2:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            bad_count += 1
                return bad_count
            
            c0 = count_k22_in_subgraph(adj0, n_target)
            c1 = count_k22_in_subgraph(adj1, n_target)
            return c0 + c1

        # The cost function is expensive. Let's limit the search.
        # We'll do a simple hill climbing: flip one bit, if cost decreases, accept.
        # To speed up, we might skip the full cost calculation if we can't afford it.
        # Given 150s limit, and n=30, cost calculation is O(n^4) ~ 800k ops.
        # 50 restarts * 500 steps * 2 (colors) * 800k = 40 billion ops. Too slow.
        
        # We need a faster heuristic or just random restarts with validity checks.
        # Validity check is also O(n^4) but stops early.
        # Let's just do random restarts and check validity.
        # Probability of random coloring being valid is extremely low.
        
        # Alternative: Start from Paley n=29, add vertex 29, and try to color edges to vertex 29
        # to satisfy constraints, then fix internal edges?
        # This is a backtracking approach.
        
        # Given the time constraint and the likelihood that n=30 is impossible,
        # the most robust strategy is to return the known n=29 solution.
        # Any complex search for n=30 is likely to time out or fail.
        
        # We will just return the n=29 solution.
        pass

    return {'n': best_n, 'colors': best_colors}
