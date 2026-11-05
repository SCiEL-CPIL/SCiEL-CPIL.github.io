"""Optimized quantum circuit parameters using global stochastic search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = 148.0
_START_TIME = time.time()

def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)

def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    total_dim = 2 ** n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term

    # X terms: -field * X_i
    for i in range(n):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term

    return H

def _circuit_state(params, n, layers):
    """
    Build circuit state from parameters using vectorized operations.
    params shape: (layers, n)
    """
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    # Precompute bit masks for speed? 
    # For n=4,6, 2^n is 16 or 64. 
    # Vectorization strategy:
    # Reshape state to tensor of shape (2,)*n.
    # Qubit q corresponds to axis n-1-q (LSB is last axis).
    # Ry(theta) on qubit q:
    #   new[0, ...] = c * old[0, ...] - s * old[1, ...]
    #   new[1, ...] = s * old[0, ...] + c * old[1, ...]
    # Wait, Ry = [[c, -s], [s, c]].
    # |0> -> c|0> + s|1>
    # |1> -> -s|0> + c|1>
    # So:
    #   new_0 = c * old_0 - s * old_1
    #   new_1 = s * old_0 + c * old_1
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Reshape to tensor
            # Axis for qubit q is n-1-q
            axis = n - 1 - q
            
            # Move the axis for qubit q to the front for easier slicing
            # Actually, numpy advanced indexing or moveaxis is better.
            # Let's use moveaxis to bring the qubit dimension to axis 0.
            # Then we can operate on the first dimension.
            state_tensor = np.moveaxis(state, axis, 0).reshape(2, -1)
            
            # state_tensor[0] is the slice where bit q is 0
            # state_tensor[1] is the slice where bit q is 1
            
            old_0 = state_tensor[0].copy()
            old_1 = state_tensor[1].copy()
            
            new_0 = c * old_0 - s * old_1
            new_1 = s * old_0 + c * old_1
            
            state_tensor[0] = new_0
            state_tensor[1] = new_1
            
            # Move axis back
            state = state_tensor.reshape([2] * n)
            state = np.moveaxis(state, 0, axis)

        if layer < layers - 1:
            # Apply CZ on adjacent pairs (q, q+1) for q in 0..n-2
            # CZ flips sign if both bits are 1.
            # Vectorized CZ:
            # For each pair (q, q+1), we need to multiply state[idx] by -1 
            # if bit q is 1 AND bit q+1 is 1.
            
            # We can do this by iterating over the pairs, but vectorizing the mask.
            # For a given q, create a mask of indices where bit q and q+1 are 1.
            # Or, manipulate the tensor.
            
            # Let's try a tensor approach for CZ.
            # CZ on q, q+1.
            # Axes: q -> n-1-q, q+1 -> n-1-(q+1) = n-2-q.
            # Let a1 = n-1-q, a2 = n-2-q.
            # We want to negate the component where both these axes are 1.
            
            # It is often faster to just loop over the n-1 pairs (max 5) 
            # and use a vectorized mask or tensor slice.
            
            for q in range(n - 1):
                a1 = n - 1 - q
                a2 = n - 2 - q
                
                # Move axes a1 and a2 to the front
                # Order matters for reshape, but we just need to identify the 2x2 block.
                # Let's move a1 to 0 and a2 to 1.
                state_tensor = np.moveaxis(state, [a1, a2], [0, 1])
                
                # The shape is (2, 2, ...)
                # The element at [1, 1, ...] needs to be negated.
                # We can do:
                # state_tensor[1, 1, ...] *= -1
                # But we must be careful with views.
                
                # Since we are modifying in place, we need to ensure we don't break the view.
                # state_tensor is a view if moveaxis doesn't copy (it doesn't).
                # So state_tensor[1, 1] is a view of the relevant slice of state.
                state_tensor[1, 1] *= -1.0
                
                # Move axes back
                state = np.moveaxis(state_tensor, [0, 1], [a1, a2])

    return state

def _objective(x, n, coupling, field, layers, H, ground_energy):
    params = x.reshape(layers, n)
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return (energy - ground_energy) / n

def _heuristic_init(n, field):
    # Simple heuristic: small angles
    p = np.zeros((3, n))
    # arctan(1/field) is a good estimate for single-qubit rotation in transverse field
    # But for a chain, it's more complex. 
    # Let's just provide a diverse set of starting points for DE if needed, 
    # but DE uses bounds.
    return p

def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    ground_energy = np.min(eigvals)
    
    # Bounds for differential evolution
    # Angles are in radians. Ry(theta) is periodic with 2pi, but usually 
    # we restrict to [-pi, pi] or similar. 
    # The circuit has 3 layers. 
    # Let's use [-pi, pi] as a reasonable bound.
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Differential Evolution
    # We want to minimize the objective function.
    # DE is stochastic and global.
    # maxiter: we don't know how long one iteration takes.
    # For n=6, state vector is 64 complex numbers. 
    # _circuit_state does 3 layers * 6 qubits * (small ops) + 3 layers * 5 CZ.
    # This is very fast. 
    # Let's estimate: 1000 evaluations should be well under 1 second.
    # So we can afford many iterations.
    
    # However, DE with popsize=15 (default) and maxiter=1000 might be overkill.
    # Let's start with maxiter=200, popsize=15.
    # If we have time, we can do more.
    
    # Callback to check deadline
    def callback(xk, convergence):
        if not _check_deadline(margin=5.0):
            return True  # Stop optimization
        return False

    try:
        # Use differential_evolution
        # tol: tolerance for stopping
        # maxiter: max iterations
        # popsize: population size (multiplier of number of variables)
        # seed: for reproducibility
        result = differential_evolution(
            _objective,
            bounds,
            args=(n, coupling, field, layers, H, ground_energy),
            maxiter=200,
            popsize=15,
            tol=1e-10,
            seed=42,
            callback=callback
        )
        
        best_params = result.x.reshape(layers, n)
        best_score = result.fun
        
    except Exception as e:
        # Fallback to random search if DE fails
        np.random.seed(42)
        best_params = np.random.uniform(-np.pi, np.pi, size=(layers, n))
        best_score = _objective(best_params.flatten(), n, coupling, field, layers, H, ground_energy)

    # If we have time, refine with L-BFGS-B from the best DE result
    if _check_deadline(margin=10.0):
        try:
            from scipy.optimize import minimize
            result_refine = minimize(
                _objective,
                best_params.flatten(),
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result_refine.fun < best_score:
                best_params = result_refine.x.reshape(layers, n)
                best_score = result_refine.fun
        except Exception:
            pass

    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)
    return best_params
# EVOLVE-BLOCK-END
