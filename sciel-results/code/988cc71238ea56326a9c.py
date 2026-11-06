"""
Strategy: Spectral Relaxation via Power Method.
Instead of discrete local search or simulated annealing, this approach uses the 
spectral relaxation of the Max-Cut problem. The Max-Cut problem can be formulated 
as maximizing x^T A x subject to x_i in {-1, 1}. The continuous relaxation 
maximizes x^T A x subject to ||x||^2 = n. The solution to this continuous problem 
is the top eigenvector of the adjacency matrix A.
We use the Power Method to approximate the top eigenvector, which is computationally 
efficient (matrix-vector products) and globally convergent. The final partition is 
obtained by rounding the signs of the converged eigenvector. This leverages the 
global structure of the graph rather than local neighborhood improvements.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Spectral Relaxation (Power Method on Adjacency Matrix) to find 
    the top eigenvector, then rounds to 0/1 labels.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    # 1. Spectral Initialization via Power Method
    # We want to find the principal eigenvector of A.
    # x_{k+1} = A x_k / ||A x_k||
    # We initialize with a random vector to avoid zero vector issues, 
    # but use a fixed seed for reproducibility if possible, or just random.
    # Since the graph is undirected and connected (likely), the top eigenvector 
    # is unique up to sign.
    
    # Check if A is all zeros
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    # Random start vector
    np.random.seed(42) # Deterministic start for consistency
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    # Power Iteration
    # The number of iterations depends on the spectral gap. 
    # For n=100, even 100-500 iterations are very fast.
    # We run until convergence or time limit.
    
    max_iterations = 1000
    tol = 1e-10
    
    for i in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
            
        # Matrix-vector multiplication
        Ax = adj @ x
        
        # Norm
        norm = np.linalg.norm(Ax)
        
        if norm < 1e-12:
            # Graph might be disconnected or zero, fallback
            break
            
        x_new = Ax / norm
        
        # Check convergence
        # We can check the angle between x and x_new
        # Or simply check change in x
        # Since x and x_new can differ by sign, we check |x - x_new| + |x + x_new|
        # Or just dot product absolute value
        diff = np.sum(np.abs(x - x_new)) + np.sum(np.abs(x + x_new))
        x = x_new
        
        if diff < tol:
            break
            
    # 2. Rounding to 0/1 labels
    # The sign of the eigenvector determines the partition.
    # x > 0 -> 1, x <= 0 -> 0
    # Note: The top eigenvector of a connected graph has all non-zero entries.
    
    # To handle potential numerical issues where x is very close to 0,
    # we can just use the sign.
    
    labels = (x > 0).astype(np.int8)
    
    # 3. Optional: Simple Greedy Polish
    # Spectral solution is usually very good, but a few greedy steps 
    # can fix any local imperfections introduced by rounding.
    # This is a very lightweight polish, not the main driver.
    
    def greedy_polish(labels, time_limit):
        adj_float = adj
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            # Compute gains for flipping each vertex
            # Gain = 2 * (same_side_neighbors) - degree
            # same_side_neighbors[i] = sum(adj[i] * (labels == labels[i]))
            
            # Vectorized calculation
            # labels shape (n,), adj shape (n,n)
            # We want for each i: count of j such that adj[i,j]=1 and labels[j]==labels[i]
            
            # Create mask of same labels
            # same_mask[i, j] = (labels[i] == labels[j])
            # This is O(n^2) memory and compute, which is fine for n=100.
            
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj_float * same_mask, axis=1)
            degrees = np.sum(adj_float, axis=1)
            
            gains = 2 * same_counts - degrees
            
            max_gain = np.max(gains)
            if max_gain > 1e-9:
                flip_idx = np.argmax(gains)
                labels[flip_idx] = 1 - labels[flip_idx]
                improved = True
                
        return labels

    # Run polish with a short time budget (e.g., 10 seconds)
    # The spectral part is usually done in milliseconds.
    if time.time() < deadline - 5.0:
        labels = greedy_polish(labels, deadline - 2.0)
        
    return labels
# EVOLVE-BLOCK-END
