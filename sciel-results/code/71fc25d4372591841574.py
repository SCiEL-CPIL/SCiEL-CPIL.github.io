def _compute_gradient(sim, params, H):
    """Compute gradient of energy w.r.t. all parameters."""
    n = sim.n
    layers = sim.layers
    dim = sim.dim
    
    # Forward pass to get final state
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Store intermediate states for backward pass
    # Actually, let's do forward-mode AD: track derivative of state w.r.t. each param
    
    # Better: compute gradient for each param separately using parameter-shift
    # For Ry(θ), parameter shift rule: dE/dθ = (E(θ+π/2) - E(θ-π/2))/2
    # But that's 2*L*N function evaluations. With L=3, N=4 or 6, that's 24-36 evals.
    # Each eval is fast (dim=16 or 64), so this might be OK.
    
    # Actually, let's just do central differences for simplicity and speed
    # With dim=16 or 64, each circuit eval is very fast.
    
    grad = np.zeros((layers, n))
    eps = 1e-7
    
    for l in range(layers):
        for q in range(n):
            params_plus = params.copy()
            params_plus[l, q] += eps
            params_minus = params.copy()
            params_minus[l, q] -= eps
            
            sp = sim.run(params_plus)
            sm = sim.run(params_minus)
            
            ep = np.real(np.vdot(sp, H @ sp))
            em = np.real(np.vdot(sm, H @ sm))
            
            grad[l, q] = (ep - em) / (2 * eps)
    
    return grad
