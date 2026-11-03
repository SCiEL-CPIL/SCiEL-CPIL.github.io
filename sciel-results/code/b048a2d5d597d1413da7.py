def _energy_and_grad(c):
    c = c.reshape(N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d2_pairs = d2[mask]
    
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    e1 = 1.0 / (d2_pairs + eps)
    E1 = np.sum(e1)
    
    d4 = (d2 + eps)**2
    np.fill_diagonal(d4, np.inf)
    coeff = -2.0 / d4
    grad_pairs = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    abs_c = np.abs(c)
    dist_w = 0.5 - abs_c
    e2_terms = 1.0 / (dist_w**2 + eps)
    E2 = np.sum(e2_terms)
    
    sign_c = np.sign(c)
    grad_w = 2.0 * sign_c / (dist_w**3 + eps)
    
    E = E1 + E2
    grad = grad_pairs + grad_w
    
    return E, grad
