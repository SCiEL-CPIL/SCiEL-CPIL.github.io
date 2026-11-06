import time
import random
import sys
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy:
    1. Start with a random graph on n=21.
    2. Use Simulated Annealing to minimize the number of regular induced subgraphs.
    3. The cost function is the count of regular induced subgraphs of size >= 6.
    4. Use vectorized NumPy operations for fast incremental cost updates.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit (evaluator kills at 150s)
    
    # Precompute all subsets of size >= 6 for n=21
    # C(21, 6) = 54264
    # C(21, 7) = 116280
    # ...
    # Total subsets is around 1.2 million.
    # Storing them as a list of tuples is memory heavy but manageable?
    # 1.2M * 6 * 8 bytes ~ 57MB. Plus overhead. Might be tight.
    # Alternative: Generate subsets on the fly? No, too slow.
    # Better: Precompute for n=20 first (C(20,6)=38760, total ~ 600k subsets).
    # If we find a valid n=20 graph, try to extend to 21.
    # But the directive suggests a structural change. Let's try to optimize n=21 directly if possible,
    # or n=20 if n=21 is too hard.
    
    # Let's try n=21.
    n = 21
    
    # Precompute subsets
    # To save memory and time, we can store subsets as integer bitmasks?
    # 21 bits fits in an int32.
    # We need to check regularity.
    # For a subset S (bitmask), the degree of vertex v in S is the popcount of adj[v] & S.
    # This is very fast with numpy if we can vectorize the popcount?
    # numpy doesn't have a direct popcount for 2D arrays of bitmasks easily without loops.
    # However, we can represent the adjacency matrix as a list of bitmasks (one int per vertex).
    # Then for a subset S (bitmask), deg(v) = (adj_mask[v] & S).bit_count().
    # In Python, int.bit_count() is C-speed.
    # If we have 1.2M subsets, and for each we check 21 vertices... 25M bit_count calls.
    # That might be too slow in Python loop.
    
    # Let's stick to NumPy matrices for the graph, but optimize the check.
    # Actually, the "penalty" approach requires evaluating the cost of a move.
    # If we flip edge (u, v), which subsets become regular or irregular?
    # Only subsets containing BOTH u and v are affected in their degree sequence?
    # No, flipping (u,v) changes degree of u and v in any subset containing both.
    # It changes the regularity status of those subsets.
    # Number of subsets containing both u and v: sum_{k=6}^n C(n-2, k-2).
    # For n=21, k=6..21. C(19, 4) + ...
    # C(19,4) = 3876.
    # This is small! We can precompute for each edge (u,v) the list of subsets containing both.
    
    # Memory check:
    # 21*20/2 = 210 edges.
    # Each edge has ~5000-10000 subsets?
    # Total subset references: 210 * 10000 = 2.1M references. Manageable.
    
    # Precomputation
    subsets_by_size = []
    for size in range(6, n + 1):
        subs = list(combinations(range(n), size))
        subsets_by_size.append(subs)
        
    # Map each subset to an index? Or just keep the list.
    # We need to know, for each edge (u,v), which subsets contain both.
    # Let's create a list of lists: affected_subsets[edge_idx] = [subset_indices...]
    # But checking regularity requires the subset vertices.
    
    # Alternative Strategy:
    # Just do a fast random search with a very fast verifier.
    # The current program's verifier is slow because it loops in Python.
    # Can we vectorize the verifier?
    
    def is_valid_fast(adj, n):
        """
        Checks validity using vectorized operations.
        adj: n x n numpy array of 0/1.
        """
        # For each size k from 6 to n:
        # Generate all combinations as indices.
        # We can't easily vectorize "extract submatrix" for all combos at once
        # without huge memory or fancy indexing that might be slow.
        
        # However, we can check regularity condition:
        # A subset S is regular if deg_S(u) = deg_S(v) for all u,v in S.
        # deg_S(u) = sum_{v in S} adj[u,v].
        
        # Let's try a different angle:
        # Random graphs are likely valid.
        # The probability that a random k-subset is regular is low.
        # We can sample subsets? No, must be exact.
        
        # Let's optimize the loop verifier.
        # The bottleneck is `is_regular_subset_fast`.
        # `sub_adj.sum(axis=1)` is fast.
        # The loop `for S in combs` is the bottleneck.
        
        # Can we use a C extension? No.
        # Can we use Numba? No, standard lib only.
        
        # Let's try to reduce the number of checks.
        # If a graph has a regular induced subgraph of size K, it might have one of size K-1?
        # No.
        
        # Let's try the Simulated Annealing with the "affected subsets" idea, but implemented carefully.
        # If we flip edge (u,v):
        # Only subsets containing u and v change their degree vector.
        # Specifically, deg(u) and deg(v) change by +/-1.
        # A subset S containing u,v was regular with degree d.
        # After flip:
        # If we add edge: deg(u) becomes d+1, deg(v) becomes d+1.
        # If all other degrees were d, they remain d.
        # So u and v are now d+1, others d. Not regular (unless d+1==d, impossible).
        # Wait, if the subset was regular with degree d, and we add edge (u,v):
        # deg(u) -> d+1, deg(v) -> d+1.
        # Other vertices in S have degree d.
        # So the new degrees are {d, d, ..., d+1, d+1}.
        # This is NOT regular (unless size=2, but size>=6).
        # So adding an edge to a regular subset makes it irregular.
        
        # What if the subset was NOT regular?
        # It might become regular.
        # This makes incremental cost update hard because we need to know the state of "almost regular" subsets.
        
        # Let's go back to brute force but optimized.
        # The current program scored 1.0, which means it found n=20 (value 20, seed 20? No, score is value/seed.
        # If seed program returns n=20, score is 1.0.
        # Inspiration 1 scored 3.4. 3.4 * 20 = 68? No.
        # "search score is value/seed".
        # If seed program returns n=20, score = 20/20 = 1.0?
        # Or is the "seed" the baseline value?
        # "1.0 equals the seed program".
        # So if I return n=21, and seed is 20, score = 21/20 = 1.05.
        # If I return n=22, score = 22/20 = 1.1.
        
        # The goal is to maximize n.
        # Known: n=20 exists. n=21 is the target.
        
        # Let's try to find n=21.
        # Random search for n=21 might be too sparse in solution space.
        # Local search is better.
        
        # Let's implement a fast local search that minimizes the number of regular subsets.
        # Cost = sum_{S, |S|>=6} [S is regular]
        
        # To speed up the initial cost calculation and updates, we will use a precomputed list of subsets
        # and a fast check function.
        
        # Optimization:
        # Instead of checking all subsets every time, we can just do a few random restarts
        # and a fast local search.
        
        def count_regular_subsets(adj, n):
            count = 0
            for size in range(6, n + 1):
                combs = list(combinations(range(n), size))
                for S in combs:
                    sub_adj = adj[np.ix_(S, S)]
                    degs = sub_adj.sum(axis=1)
                    if np.all(degs == degs[0]):
                        count += 1
            return count

        # This is still slow.
        
        # Let's try a different representation for the check.
        # Use bitmasks.
        # adj_mask[i] is an integer with bits set for neighbors of i.
        
        def is_regular_mask(adj_masks, S):
            # S is a tuple of indices
            mask_S = 0
            for v in S:
                mask_S |= (1 << v)
            
            degrees = []
            for v in S:
                # Count neighbors of v that are in S
                deg = (adj_masks[v] & mask_S).bit_count()
                degrees.append(deg)
            
            return all(d == degrees[0] for d in degrees)

        # This is also slow in Python loop.
        
        # Let's use NumPy for the bit_count equivalent?
        # No, NumPy doesn't do bit_count on integers well.
        
        # What if we use `np.bitwise_count`? It exists in newer NumPy?
        # `np.bitwise_count` is available in NumPy 2.0?
        # Assuming standard library and recent NumPy.
        # If not, we can use `np.unpackbits`? Too slow.
        
        # Let's assume we can use a fast C-like loop via `functools` or similar? No.
        
        # Let's try the "Inspiration 1" strategy which scored 3.4.
        # "Randomized greedy construction".
        # Maybe it constructs the graph vertex by vertex, ensuring no regular subset is formed?
        
        # Let's try a Greedy Construction:
        # Start with empty graph.
        # Add vertices one by one.
        # For each new vertex, decide edges to previous vertices.
        # Constraint: No regular induced subgraph of size >= 6.
        
        # This is a CSP.
        
        # Let's try a simple heuristic:
        # Generate random graphs on n=21.
        # Use a very fast filter:
        # Check only subsets of size 6?
        # If a graph has a regular induced subgraph of size K>6, does it necessarily have one of size 6?
        # No.
        # But if it has NO regular induced subgraph of size 6, it might still have one of size 7.
        
        # However, the density of regular subsets drops as size increases?
        # Actually, for random graphs, the expected number of regular induced subgraphs of size k is:
        # E_k = C(n, k) * P(regular).
        # P(regular) is roughly (k-1)! / k^k? No.
        # Number of regular graphs on k vertices is small compared to all graphs.
        
        # For k=6, P(regular) is not tiny.
        # For k=21, P(regular) is extremely tiny.
        
        # So checking size 6 is the most critical.
        # If we eliminate all regular 6-subsets, the probability of having a regular 7-subset is lower.
        
        # Let's try to find a graph with NO regular 6-subsets.
        # Then verify the rest.
        
        def has_regular_6(adj, n):
            for S in combinations(range(n), 6):
                sub_adj = adj[np.ix_(S, S)]
                degs = sub_adj.sum(axis=1)
                if np.all(degs == degs[0]):
                    return True
            return False

        # This is C(21,6) = 54k checks.
        # Each check is fast (6x6 matrix sum).
        # 54k * ~10us = 0.5s.
        # This is fast enough to do many random trials!
        
        best_adj = None
        best_n = 0
        
        # Try to find a valid n=21 graph
        # Strategy: Random search with "No Regular 6" filter.
        # If passes, full verify.
        
        for trial in range(1000):
            if time.time() > deadline - 5:
                break
                
            random.seed(trial)
            n_try = 21
            adj = np.zeros((n_try, n_try), dtype=np.int8)
            # Random graph with p=0.5
            mask = np.random.rand(n_try, n_try) < 0.5
            adj = np.triu(mask, 1).astype(np.int8)
            adj = adj + adj.T
            
            # Fast filter: Check for regular 6-subsets
            valid_6 = True
            for S in combinations(range(n_try), 6):
                sub_adj = adj[np.ix_(S, S)]
                degs = sub_adj.sum(axis=1)
                if np.all(degs == degs[0]):
                    valid_6 = False
                    break
            
            if not valid_6:
                continue
                
            # Full verification
            valid = True
            for size in range(7, n_try + 1):
                for S in combinations(range(n_try), size):
                    sub_adj = adj[np.ix_(S, S)]
                    degs = sub_adj.sum(axis=1)
                    if np.all(degs == degs[0]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_adj = adj
                best_n = n_try
                break
                
        if best_adj is None:
            # Try n=20
            for trial in range(1000):
                if time.time() > deadline - 5:
                    break
                random.seed(1000 + trial)
                n_try = 20
                adj = np.zeros((n_try, n_try), dtype=np.int8)
                mask = np.random.rand(n_try, n_try) < 0.5
                adj = np.triu(mask, 1).astype(np.int8)
                adj = adj + adj.T
                
                valid_6 = True
                for S in combinations(range(n_try), 6):
                    sub_adj = adj[np.ix_(S, S)]
                    degs = sub_adj.sum(axis=1)
                    if np.all(degs == degs[0]):
                        valid_6 = False
                        break
                
                if not valid_6:
                    continue
                
                valid = True
                for size in range(7, n_try + 1):
                    for S in combinations(range(n_try), size):
                        sub_adj = adj[np.ix_(S, S)]
                        degs = sub_adj.sum(axis=1)
                        if np.all(degs == degs[0]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_adj = adj
                    best_n = n_try
                    break
        
        if best_adj is None:
            # Fallback
            return {'n': 5, 'edges': []}
            
        # Convert to edges
        edges = []
        for i in range(best_n):
            for j in range(i):
                if best_adj[i, j] == 1:
                    edges.append([j, i])
                    
        return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
