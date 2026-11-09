<<<<<<< SEARCH
    if n == 0 or fast_memory_budget <= 0:
        return []
    
    # Build children map for efficiency
=======
    if n == 0 or fast_memory_budget <= 0:
        return []
    
    # If budget is large enough to hold all vertices, just compute all and store outputs
    # This is optimal (min transfers = load inputs + store outputs)
    if fast_memory_budget >= n:
        ops = []
        # Load all inputs (vertices with no parents)
        for v in range(n):
            parents = get_parents(v)
            if not parents:
                ops.append(['load', v])
        # Compute all vertices in topological order
        for v in range(n):
            ops.append(['compute', v])
        # Store outputs
        for v in sorted(outputs):
            ops.append(['store', v])
        return ops

    # Build children map for efficiency
>>>>>>> REPLACE
