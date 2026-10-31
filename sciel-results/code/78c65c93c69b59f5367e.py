def construct():
    N = 400
    # Start with all ones
    h = np.ones(N)
    
    # Try to add a perturbation
    # v should be orthogonal to 1
    # Use a random vector, project out the mean
    np.random.seed(42)
    v = np.random.randn(N)
    v -= np.mean(v)
    
    best_h = h.copy()
    best_val = _score(h)
    
    # Grid search on epsilon
    # Score(epsilon) = max|a(1 + eps v)| / (N + eps sum v)^2
    # sum v = 0, so denominator is N^2
    # Numerator is max|a(1) + 2 eps a(1,v) + eps^2 a(v)|
    
    # a(1) is triangle. Peak N.
    # a(v) is autoconv of v.
    # a(1,v) is cross conv.
    
    # Let's just evaluate for a few epsilons
    for eps in np.linspace(-0.5, 0.5, 100):
        h_test = h + eps * v
        val = _score(h_test)
        if val < best_val:
            best_val = val
            best_h = h_test
    return best_h
