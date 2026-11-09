<<<<<<< SEARCH
def _objective(params, n, coupling, field, H, ground_energy):
    """Objective: (circuit_energy - ground_energy) / n."""
    energy = _circuit_energy(params, n, coupling, field, H)
    return (energy - ground_energy) / n

def _objective_grad(params, n, coupling, field, H, ground_energy):
    """Objective with finite-difference gradient."""
    energy = _circuit_energy(params, n, coupling, field, H)
    obj = (energy - ground_energy) / n
    
    eps = 1e-6
    grad = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        orig = params[idx]
        params[idx] = orig + eps
        e_plus = _circuit_energy(params, n, coupling, field, H)
        params[idx] = orig - eps
        e_minus = _circuit_energy(params, n, coupling, field, H)
        params[idx] = orig
        grad[idx] = (e_plus - e_minus) / (2.0 * eps) / n
    
    return obj, grad
=======
def _objective(params, n, coupling, field, H, ground_energy):
    """Objective: (circuit_energy - ground_energy) / n.
    Accepts 1D or 2D params, reshapes to (layers, n) internally.
    """
    layers = 3 # Fixed by task contract
    # Reshape if 1D
    if params.ndim == 1:
        params_2d = params.reshape((layers, n))
    else:
        params_2d = params
    
    energy = _circuit_energy(params_2d, n, coupling, field, H)
    return (energy - ground_energy) / n
>>>>>>> REPLACE

