def _objective_wrapper(x, n, m, p, R):
    size_u = n * m * R
    size_v = m * p * R
    size_w = p * n * R
    
    U = x[:size_u].reshape(n * m, R)
    V = x[size_u:size_u + size_v].reshape(m * p, R)
    W = x[size_u + size_v:].reshape(p * n, R)
    
    return _compute_error_fast(U, V, W, n, m, p)
