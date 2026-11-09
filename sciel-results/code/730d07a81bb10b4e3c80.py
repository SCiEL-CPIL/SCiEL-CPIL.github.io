import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # 1. Compute Degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # 2. Compute Core Numbers (k-core decomposition)
    # This is O(n + m)
    def compute_core_numbers(adj):
        n = adj.shape[0]
        # Use numpy for neighbor lists for speed in the loop if needed, 
        # but standard list of lists is fine for n=200
        neighbors = [np.where(adj[i])[0] for i in range(n)]
        degs = adj.sum(axis=1).astype(int).copy()
        core = np.zeros(n, dtype=int)
        
        # Standard k-core algorithm
        # Bucket by degree
        max_deg = int(degs.max())
        buckets = [[] for _ in range(max_deg + 1)]
        for i in range(n):
            buckets[degs[i]].append(i)
            
        k = max_deg
        remaining = np.ones(n, dtype=bool)
        count_remaining = n
        
        while k >= 0 and count_remaining > 0:
            # Process all vertices with degree == k
            # Note: degrees might drop below k during processing
            # We need to check if deg[i] == k
            # Efficient way: iterate through bucket[k]
            # But degrees change.
            # Standard approach:
            # While there is a vertex with degree <= k, remove it.
            # We can just iterate vertices and check condition.
            # For n=200, simple loop is fast enough.
            
            # Let's use a queue-like approach or just repeated scanning
            # Since n is small, we can do:
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if remaining[i] and degs[i] <= k:
                        # Remove i
                        core[i] = k
                        remaining[i] = False
                        count_remaining -= 1
                        for u in neighbors[i]:
                            if remaining[u]:
                                degs[u] -= 1
                        changed = True
            k -= 1
            
        return core

    core_numbers = compute_core_numbers(adjacency)
    
    # 3. Compute Principal Eigenvector of Adjacency Matrix
    # adjacency is symmetric
    # Use eigh for symmetric matrices
    # For n=200, this is very fast
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
        # The principal eigenvector corresponds to the largest eigenvalue
        # eigh returns eigenvalues in ascending order
        idx_max = np.argmax(eigenvalues)
        principal_eigenvector = np.abs(eigenvectors[:, idx_max])
    except:
        # Fallback if eigh fails (shouldn't happen for symmetric)
        principal_eigenvector = degrees / (degrees.max() + 1e-6)

    # 4. Normalize features to [0, 1]
    def normalize(x):
        min_x = np.min(x)
        max_x = np.max(x)
        if max_x - min_x < 1e-9:
            return np.zeros_like(x)
        return (x - min_x) / (max_x - min_x)

    norm_deg = normalize(degrees)
    norm_core = normalize(core_numbers.astype(float))
    norm_eig = normalize(principal_eigenvector)

    # 5. Construct Score
    # We want HIGH score for vertices that are:
    # - Low Degree (1 - norm_deg)
    # - Low Core Number (1 - norm_core)
    # - Low Eigenvector Centrality (1 - norm_eig)
    
    # Weights: 
    # Eigenvector centrality is a strong indicator of "hubs" which are bad for IS.
    # Core number is a strong indicator of "deep structure".
    # Degree is a simple local measure.
    
    # Let's try a weighted sum.
    # Since we are scoring for a greedy decoder that picks highest first,
    # we want the "best" IS candidates to have the highest scores.
    
    w_eig = 5.0
    w_core = 3.0
    w_deg = 1.0
    
    scores = w_eig * (1.0 - norm_eig) + w_core * (1.0 - norm_core) + w_deg * (1.0 - norm_deg)
    
    # Optional: Add a small random tie-breaker to ensure distinct scores if needed, 
    # but the task says "fixed greedy decoder", implying it handles ties or we should provide distinct scores.
    # Usually, distinct scores are safer to avoid undefined tie-breaking behavior in the evaluator.
    # Add tiny noise based on index to ensure uniqueness without changing order significantly.
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-6, n)
    scores += noise
    
    return scores
