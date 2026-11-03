def residual(x, rank, penalty_weight):
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    res_tensor = (T - T_recon).flatten()
    
    # Penalty for non-half-integers
    # We want x to be close to 0.5 * integer
    # penalty = sum((x - 0.5*round(2*x))^2)
    x_half = 0.5 * np.round(2 * x)
    res_penalty = (x - x_half).flatten()
    
    return np.concatenate([res_tensor, penalty_weight * res_penalty])
