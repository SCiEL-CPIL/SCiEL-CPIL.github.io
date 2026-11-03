def search(available):
    if not available: return 0
    # Heuristic: pick v in available with max degree in induced subgraph of 'available'
    # This is expensive to compute exactly.
    # Approximation: pick v with max global degree? Or min degree?
    # For MIS, picking high degree vertices first is usually good for pruning.
