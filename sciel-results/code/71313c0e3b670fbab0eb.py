"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a K4-free graph on 18 vertices that minimizes the size of the
    largest triangle-free induced subgraph.
    
    Approach: Use spectral optimization. We represent the graph by a continuous
    adjacency matrix and optimize via gradient descent to:
    1. Penalize K4 subgraphs (to maintain K4-free property)
    2. Minimize the spectral radius of triangle-free subgraph indicators
    
    Then round to the nearest valid K4-free graph and verify.
    """
    n = 18
    np.random.seed(42)
    
    # Initialize with a random symmetric matrix
    A = np.random.uniform(-1, 1, (n, n))
    np.fill_diagonal(A, 0)
    A = (A + A.T) / 2.0  # Symmetrize
    
    # Learning rate schedule
    lr = 0.01
    
    # Number of iterations
    n_iters = 500
    
    for it in range(n_iters):
        # Current discrete approximation
        A_disc = np.sign(A)
        np.fill_diagonal(A_disc, 0)
        A_disc = np.maximum(A_disc, 0)  # Only keep positive edges
        
        # Compute gradient components
        
        # 1. K4 penalty gradient
        # For each potential K4 (4 vertices), check if all 6 edges are present
        # Penalty = sum over all 4-sets of (product of 6 edge indicators)
        # Gradient w.r.t. each edge is sum over K4s containing that edge of (product of other 5 edges)
        
        grad_k4 = np.zeros((n, n))
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            # Product of all 6 edge values
            prod = 1.0
            for u, v in edges:
                prod *= A_disc[u, v]
            if prod > 0.5:  # K4 present
                # Add penalty gradient: for each edge in this K4, add product of others
                for i, (u, v) in enumerate(edges):
                    prod_others = 1.0
                    for j, (x, y) in enumerate(edges):
                        if i != j:
                            prod_others *= A_disc[x, y]
                    grad_k4[u, v] += prod_others
                    grad_k4[v, u] += prod_others
        
        # 2. Triangle-free subgraph spectral radius penalty
        # We want to minimize the maximum over all subsets S that are triangle-free of lambda_max(A[S,S])
        # Approximate by: for each triangle-free set, compute spectral radius
        # Gradient: push eigenvalues down
        
        # Find all triangle-free induced subgraphs (small ones for efficiency)
        # For each set of vertices, check if triangle-free, compute spectral radius
        
        grad_spec = np.zeros((n, n))
        
        # Check subsets of size up to 8 (larger ones are harder to enumerate)
        for size in range(3, 9):
            for subset in combinations(range(n), size):
                # Check if triangle-free
                is_tri_free = True
                for tri in combinations(subset, 3):
                    a, b, c = tri
                    if A_disc[a, b] > 0.5 and A_disc[b, c] > 0.5 and A_disc[a, c] > 0.5:
                        is_tri_free = False
                        break
                if not is_tri_free:
                    continue
                
                # Compute spectral radius of induced subgraph
                A_sub = A_disc[list(subset), list(subset)]
                if size > 1:
                    eigvals = np.linalg.eigvalsh(A_sub)
                    lam_max = eigvals[-1]
                    if lam_max > 0.1:
                        # Gradient: direction to decrease lambda_max
                        # Use eigenvector corresponding to lambda_max
                        eigvecs = np.linalg.eigh(A_sub)
                        v_max = eigvecs[1, -1]
                        
                        # Map gradient back to full matrix
                        for i, vi in enumerate(subset):
                            for j, vj in enumerate(subset):
                                grad_spec[vi, vj] += lam_max * v_max[i] * v_max[j]
        
        # Combine gradients
        grad = grad_k4 + grad_spec
        
        # Gradient descent step
        A -= lr * grad
        
        # Symmetrize and zero diagonal
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0)
        
        # Project to [-1, 1]
        A = np.clip(A, -1, 1)
    
    # Round to discrete graph
    A_final = np.sign(A)
    np.fill_diagonal(A_final, 0)
    A_final = np.maximum(A_final, 0)
    
    # Ensure K4-free by removing edges from any K4
    # Iterate until no K4s remain
    changed = True
    while changed:
        changed = False
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            all_present = all(A_final[u, v] > 0.5 for u, v in edges)
            if all_present:
                # Remove one edge (the one with smallest value in A)
                min_idx = min(range(6), key=lambda i: A[edges[i][0], edges[i][1]])
                u, v = edges[min_idx]
                A_final[u, v] = 0
                A_final[v, u] = 0
                A[u, v] = -1  # Push away in continuous space
                A[v, u] = -1
                changed = True
    
    # Extract edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A_final[i, j] > 0.5:
                edges.append([i, j])
    
    # Verify K4-free
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (A_final[a,b] > 0.5 and A_final[a,c] > 0.5 and A_final[a,d] > 0.5 and
            A_final[b,c] > 0.5 and A_final[b,d] > 0.5 and A_final[c,d] > 0.5):
            # Should not happen after cleanup, but just in case
            pass
    
    return edges
# EVOLVE-BLOCK-END
