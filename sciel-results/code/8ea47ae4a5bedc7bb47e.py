<<<<<<< SEARCH
    def compute_loads(routing_binary):
        """Compute edge loads given binary routing decisions."""
        # routing_binary: array of shape (D,) with 0 or 1
        # Load on edge e = sum over d of weight[d] * (A[d, 0, e] if routing_binary[d]==0 else A[d, 1, e])
        # = sum_d weight[d] * (A[d, 0, e] * (1 - routing_binary[d]) + A[d, 1, e] * routing_binary[d])
        # = sum_d weight[d] * A[d, 0, e] - sum_d weight[d] * A[d, 0, e] * routing_binary[d] + sum_d weight[d] * A[d, 1, e] * routing_binary[d]
        # = base_load_e + sum_d weight[d] * (A[d, 1, e] - A[d, 0, e]) * routing_binary[d]
        
        # Precompute base loads (all CW)
        base_loads = np.zeros(n_edges)
        for d in range(D):
            base_loads += weights[d] * A[d, 0, :]
        
        # Difference matrix: diff[d, e] = weight[d] * (A[d, 1, e] - A[d, 0, e])
        diff = np.zeros((D, n_edges))
        for d in range(D):
            diff[d, :] = weights[d] * (A[d, 1, :] - A[d, 0, :])
        
        # Loads = base_loads + diff @ routing_binary
        loads = base_loads + diff @ routing_binary
        return loads
=======
    # Precompute base loads and difference matrix once outside the function for speed
    base_loads = np.zeros(n_edges)
    for d in range(D):
        base_loads += weights[d] * A[d, 0, :]
    
    diff_matrix = np.zeros((D, n_edges))
    for d in range(D):
        diff_matrix[d, :] = weights[d] * (A[d, 1, :] - A[d, 0, :])
    
    def compute_loads(routing_binary):
        """Compute edge loads given binary routing decisions."""
        # routing_binary: array of shape (D,) with 0 or 1
        # Loads = base_loads + diff_matrix @ routing_binary
        if len(routing_binary) != D:
            return np.zeros(n_edges) # Safety fallback
        loads = base_loads + diff_matrix @ routing_binary
        return loads
>>>>>>> REPLACE

