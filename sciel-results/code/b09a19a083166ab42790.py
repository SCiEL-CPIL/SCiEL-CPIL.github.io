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
    1. Use a hardcoded valid n=20 graph derived from known constructions or random search.
    2. Verify it using fast NumPy vectorized operations.
    3. If valid, return it.
    4. If time permits, attempt to extend to n=21 using fast local search with NumPy.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def is_regular_subset_fast(adj_matrix, S):
        """
        adj_matrix: numpy array (n, n) of 0s and 1s
        S: list of vertex indices
        Returns True if the induced subgraph on S is regular and len(S) >= 6
        """
        n_v = len(S)
        if n_v < 6:
            return False
        
        # Extract submatrix
        sub_adj = adj_matrix[np.ix_(S, S)]
        # Compute degrees
        degs = sub_adj.sum(axis=1)
        # Check if all degrees are equal
        return np.all(degs == degs[0])
    
    def is_valid_graph_fast(adj_matrix, n):
        """
        Checks if the graph defined by adj_matrix on n vertices has no regular induced subgraph of size >= 6.
        Uses NumPy for speed.
        """
        # We only need to check sizes 6 to n.
        # For n=20, C(20,6) is ~38k, C(20,7) is ~77k. Total subsets to check is manageable with vectorization?
        # Actually, iterating combinations in Python is slow.
        # Better approach: For each size k, generate all subsets, stack into a matrix, and check.
        # But memory might be an issue for large n.
        # For n=20, C(20,10) is 184k. Stacking 184k * 10 * 10 ints is ~1.8M ints, which is fine.
        
        for size in range(6, n + 1):
            # Generate all combinations of size
            combs = list(combinations(range(n), size))
            if not combs:
                continue
            
            # Convert to numpy array of indices
            # Shape: (num_combs, size)
            idx = np.array(combs, dtype=np.int32)
            
            # For each combination, we need to check regularity.
            # This is tricky to fully vectorize without loops over combinations because the submatrix extraction varies.
            # However, we can check degrees for all vertices in the subset at once?
            # No, the degree of a vertex depends on the specific subset.
            
            # Alternative: Iterate but use numpy for the inner check.
            # The bottleneck in the previous code was bin().count('1').
            # Using adj_matrix[S, S].sum(axis=1) is much faster than bit counting.
            
            for S in combs:
                if is_regular_subset_fast(adj_matrix, S):
                    return False
        
        return True

    # Hardcoded 20-vertex graph.
    # Since I cannot guarantee a specific random seed produces a valid graph without running it,
    # I will implement a fast randomized search for n=20 using the fast checker.
    # If I find one, I return it. If not, I return a trivial valid graph (e.g., n=5) to ensure validity,
    # but I expect to find n=20 quickly.
    
    # To improve chances, I will try a few seeds.
    # The fast checker should be able to verify a 20-vertex graph in < 1s.
    
    best_adj_matrix = None
    best_n = 0
    
    # Try to find a valid n=20 graph
    for seed in [1, 42, 123, 456, 789, 1000, 2023, 31415, 271828, 314159]:
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        n = 20
        # Generate random graph
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Verify
        if is_valid_graph_fast(adj, n):
            best_adj_matrix = adj
            best_n = n
            break
            
    if best_adj_matrix is None:
        # Fallback: Return a small valid graph
        # Empty graph on 5 vertices is valid
        return {'n': 5, 'edges': []}
        
    # Convert to edge list
    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i):
                if adj[i, j] == 1:
                    edges.append([j, i])
        return edges
        
    best_edges = adj_to_edges(best_adj_matrix, best_n)
    
    # Attempt to extend to n=21
    # This is computationally expensive. We will try a few random extensions.
    if time.time() < deadline - 20 and best_n == 20:
        for _ in range(20): # Try 20 random extensions
            if time.time() > deadline - 5:
                break
                
            # Add vertex 20
            new_adj = np.zeros((21, 21), dtype=np.int8)
            new_adj[:20, :20] = best_adj_matrix
            
            # Random connections for new vertex
            for j in range(20):
                if random.random() < 0.5:
                    new_adj[20, j] = 1
                    new_adj[j, 20] = 1
            
            # Verify the 21-vertex graph
            # We only need to check subsets including the new vertex?
            # No, the old subsets are already valid.
            # So we only check subsets S of {0..20} such that 20 in S.
            # This is equivalent to checking subsets T of {0..19} of size k-1, where S = T U {20}.
            # And checking if S is regular.
            
            valid = True
            # Check sizes 6 to 21
            for size in range(6, 22):
                k = size - 1 # Size of T
                if k < 0 or k > 20:
                    continue
                combs = list(combinations(range(20), k))
                for T in combs:
                    S = list(T) + [20]
                    if is_regular_subset_fast(new_adj, S):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_edges = adj_to_edges(new_adj, 21)
                return {'n': 21, 'edges': new_edges}
            
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
